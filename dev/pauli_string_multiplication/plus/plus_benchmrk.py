import os
import importlib
import re
import json
import csv
import time
import random
import sys
from datetime import datetime
from typing import List, Tuple
from utils.config import get_config

# Ensure project root is on sys.path so absolute imports like
# 'dev.pauli_string_multiplication.letter_replace_13.utils.llm_{backend}' work
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Resolve a path relative to project root if not absolute
_definitely_abs = lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(_PROJECT_ROOT, p))

def _print_runtime_paths(prompt_template_path: str, records_base_dir: str) -> None:
    print("\n📁 Paths (resolved):")
    print("-" * 30)
    print(f"{'✅' if os.path.exists(prompt_template_path) else '⚠️ '} prompt_template: {prompt_template_path}")
    print(f"{'✅' if os.path.isdir(records_base_dir) else '⚠️ '} records_base_dir: {records_base_dir}")

# --- Integer addition utilities ---

def random_integer(L: int) -> int:
    """Generate a random L-digit positive integer."""
    if L == 1:
        return random.randint(1, 9)
    return random.randint(10**(L-1), 10**L - 1)

def generate_questions_and_answers(L: int, batch_size: int) -> Tuple[str, List[str]]:
    """Generate random L-digit addition problems and their answers."""
    problems = []
    answers = []
    for _ in range(batch_size):
        a = random_integer(L)
        b = random_integer(L)
        result = a + b
        problems.append(f"{a}+{b}")
        answers.append(str(result))
    
    questions_block = "\n".join(
        [f"({i+1}) Input: $ {p} $\n    Your Answer: " for i, p in enumerate(problems)]
    )
    return questions_block, answers

# --- Configurable parameters ---
L = 4                 # number of digits in each integer
batch_size = 10       # questions per prompt
num_iterations = 3    # number of prompt rounds
SAVE_LLM_RESPONSE = False
TEMPERATURE = 0.0

# --- Model configuration (set before running) ---
LLM_BACKEND = None  # "openai" | "gemini" | "claude" | "grok"
MODEL_NAME = None   # model id string

# --- Load prompt template ---
def load_prompt_template(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()

# --- Extract Python list of answers from LLM response ---
ANSWER_LIST_PATTERN = r'<answers>\s*```python\s*LLM_answers\s*=\s*\[(.*?)\]\s*```\s*</answers>'

def extract_llm_answers(response: str) -> List[str]:
    match = re.search(ANSWER_LIST_PATTERN, response, re.DOTALL)
    if not match:
        # fallback to any LLM_answers list
        match = re.search(r'LLM_answers\s*=\s*\[(.*?)\]', response, re.DOTALL)
        if not match:
            raise ValueError('Could not find LLM_answers list in response.')
    list_str = match.group(1)
    # extract quoted strings
    answers = re.findall(r'"([^"]*)"|\'([^\']*)\'', list_str)
    # answers is list of tuples because of two alternatives; take non-empty part
    cleaned = [(a if a != '' else b) for a, b in answers]
    if not cleaned:
        raise ValueError('Could not extract quoted answers from LLM_answers list.')
    return cleaned

# --- Save helpers ---
def save_experiment_record(record, filename, model_dir):
    with open(os.path.join(model_dir, filename), 'w') as f:
        json.dump(record, f, indent=2)

def append_accuracy_csv(row, filename, model_dir):
    file_path = os.path.join(model_dir, filename)
    write_header = not os.path.exists(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp', 'model', 'L', 'batch_size', 'temperature', 'iteration', 'accuracy', 'parsing_success', 'input_tokens', 'output_tokens', 'total_tokens'])
        writer.writerow(row)

# --- Main experiment ---
def main():
    if LLM_BACKEND is None or MODEL_NAME is None:
        raise ValueError("LLM_BACKEND and MODEL_NAME must be set before running the script.")

    config = get_config()

    try:
        API_KEY = config.get_api_key(LLM_BACKEND)
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        return

    # dynamic import of LLM caller
    llm_module = importlib.import_module(f"dev.pauli_string_multiplication.plus.utils.llm_{LLM_BACKEND}")
    query_llm = llm_module.query_llm

    # paths
    records_base_dir = _definitely_abs(config.get_path('records_base_dir'))
    os.makedirs(records_base_dir, exist_ok=True)
    model_dir = os.path.join(records_base_dir, MODEL_NAME)
    os.makedirs(model_dir, exist_ok=True)

    prompt_template_path = _definitely_abs(config.get_path('prompt_template'))

    # print resolved path status (exclude unused prompt_irrelevant)
    _print_runtime_paths(prompt_template_path, records_base_dir)

    prompt_template = load_prompt_template(prompt_template_path)

    all_accuracies = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for it in range(num_iterations):
        print(f"[Iteration {it+1}] Querying LLM ({LLM_BACKEND}, {MODEL_NAME})...")
        # Generate fresh tasks per attempt
        while True:
            try:
                questions_block, gt_answers = generate_questions_and_answers(L, batch_size)
                full_prompt = prompt_template.replace('{{QUESTIONS_BLOCK}}', questions_block)
                llm_response, token_metadata = query_llm(full_prompt, MODEL_NAME, API_KEY, temperature=TEMPERATURE)
                time.sleep(8)
                break
            except Exception as e:
                print(f"[Iteration {it+1}] ❌ API call failed: {e}")
                print(f"[Iteration {it+1}] 🔄 Retrying in 30s...")
                time.sleep(30)

        # Parse and validate
        parsing_success = False
        try:
            llm_answers = extract_llm_answers(llm_response)
            parsing_success = True
            # align sizes
            k = min(len(llm_answers), len(gt_answers))
            llm_answers = llm_answers[:k]
            gt_answers = gt_answers[:k]

            is_correct = [la == ga for la, ga in zip(llm_answers, gt_answers)]
            acc = sum(is_correct) / len(is_correct) if is_correct else 0.0
            error_message = None
        except Exception as e:
            llm_answers = []
            is_correct = [False] * len(gt_answers)
            acc = 0.0
            error_message = str(e)

        record = {
            'timestamp': timestamp,
            'model': MODEL_NAME,
            'L': L,
            'batch_size': batch_size,
            'iteration': it+1,
            'inputs': None,  # inputs are implicit in prompt; can be added if needed
            'llm_answers': llm_answers,
            'correct_answers': gt_answers,
            'is_correct_list': is_correct,
            'accuracy': acc,
            'parsing_success': parsing_success,
            'error_message': error_message,
            'input_tokens': token_metadata.get('input_tokens'),
            'output_tokens': token_metadata.get('output_tokens'),
            'total_tokens': token_metadata.get('total_tokens'),
            'temperature': TEMPERATURE,
        }
        if SAVE_LLM_RESPONSE:
            record['llm_response'] = llm_response

        print(f"L = {L}, batch_size = {batch_size}, temperature = {TEMPERATURE}, iteration = {it+1}, accuracy = {acc}, parsing_success = {parsing_success}")
        save_experiment_record(record, f'record_L{L}_batch{batch_size}_temp{TEMPERATURE}_iter{it+1}_{timestamp}.json', model_dir)
        append_accuracy_csv([
            timestamp, MODEL_NAME, L, batch_size, TEMPERATURE, it+1, acc, parsing_success,
            token_metadata.get('input_tokens'), token_metadata.get('output_tokens'), token_metadata.get('total_tokens')
        ], 'accuracy_summary.csv', model_dir)
        all_accuracies.append(acc)

    print(f"Average accuracy over {num_iterations} iterations: {sum(all_accuracies)/len(all_accuracies) if all_accuracies else 0}")

if __name__ == '__main__':
    main()
