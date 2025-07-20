import os
import importlib.util
import importlib
import re
import json
import csv
from datetime import datetime
from typing import List
import pyclifford as pc
import mcp_server as mcp
import numpy
from math import ceil

# --- Pauli string question/answer generation utilities ---
def sample_random_pauli_string(N):
    """Sample a random Pauli operator string with random phase for N qubits.
    Returns a string like '-iIXIIYZXIZ'.
    """
    paulis = ['I', 'X', 'Y', 'Z']
    phases = ['', 'i', '-', '-i']
    pauli_str = ''.join(numpy.random.choice(paulis) for _ in range(N))
    phase = numpy.random.choice(phases)
    return f'{phase}{pauli_str}'

def sample_question_and_answer(N):
    pauli_str1 = sample_random_pauli_string(N)
    pauli_str2 = sample_random_pauli_string(N)
    op1 = pc.pauli(pauli_str1)
    op2 = pc.pauli(pauli_str2)
    str1 = mcp.PauliTerm.from_obj(op1).text
    str2 = mcp.PauliTerm.from_obj(op2).text
    sol = mcp.PauliTerm.from_obj(op1 @ op2)
    return f'({str1}) ({str2})', sol

def generate_questions_and_answers(N_max, iterations):
    questions = []
    answers = []
    for i in range(iterations):
        q, a = sample_question_and_answer(N_max)
        questions.append(q)
        answers.append(a)

    questions_block = "\n".join(
        [f"({i+1}) Input: $ {q} $\n    Your Answer: " for i, q in enumerate(questions)]
    )

    return questions_block, answers

# --- Configurable parameters ---
N = 1  # Length of Pauli strings
batch_size = 10  # Number of questions per prompt
num_iterations = 3  # Number of prompt rounds
PROMPT_TEMPLATE_PATH = 'dev/pauli_string_multiplication/direct_multiplication/utils/prompt_template.txt'  # Path to prompt description file
RECORDS_BASE_DIR = 'dev/pauli_string_multiplication/direct_multiplication/records'
SAVE_LLM_RESPONSE = False  # Set to True to save the full LLM response in the record
L_irr = 0  # Number of words of irrelevant text to append to the prompt (0 = no irrelevant text)
PROMPT_IRRELEVANT_PATH = 'dev/pauli_string_multiplication/direct_multiplication/utils/prompt_irrelevant.txt'  # Optional irrelevant text file

# --- Model configuration (must be set before running) ---
LLM_BACKEND = None  # Must be set to "openai", "claude", or "gemini"
MODEL_NAME = None   # Must be set to specific model name

# --- Load prompt template ---
def load_prompt_template(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()

# --- Extract Python list of answers from LLM response ---
def extract_llm_answers(response: str) -> List[str]:
    # First, try to find the answer block with XML-like tags and Python code block
    answers_pattern = r'<answers>\s*```python\s*LLM_answers\s*=\s*\[(.*?)\]\s*```\s*</answers>'
    answers_match = re.search(answers_pattern, response, re.DOTALL)
    
    if answers_match:
        list_str = answers_match.group(1)
        print("Debug: Found <answers> block with Python code")
        print(f"Debug: List string preview: {list_str[:200]}...")
    else:
        # Fallback: look for any LLM_answers list
        print("Debug: No <answers> block found, using fallback search")
        match = re.search(r'LLM_answers\s*=\s*\[(.*?)\]', response, re.DOTALL)
        if not match:
            print("Debug: No LLM_answers pattern found at all")
            print("Debug: Response preview:")
            print("-" * 50)
            print(response[-1000:])  # Show last 1000 characters
            print("-" * 50)
            raise ValueError('Could not find LLM_answers list in response.')
        list_str = match.group(1)
        print("Debug: Using fallback pattern match")
    
    # Extract each string in the list
    answers = []
    
    # Try different quote patterns
    quoted_patterns = [
        r"'([^']*)'",  # Single quotes
        r'"([^"]*)"',  # Double quotes
    ]
    
    for pattern in quoted_patterns:
        found_answers = re.findall(pattern, list_str)
        if found_answers:
            answers = found_answers
            print(f"Debug: Found {len(found_answers)} answers using pattern {pattern}")
            break
    
    if not answers:
        print("Debug: Could not extract quoted answers")
        print(f"Debug: List string content: {list_str}")
        raise ValueError('Could not extract quoted answers from LLM_answers list.')
    
    print(f"Debug: Extracted answers: {answers[:3]}...")  # Show first 3 answers
    return answers

# --- Ensure records directory exists ---
def ensure_records_dir(model_name):
    model_dir = os.path.join(RECORDS_BASE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

# --- Save experiment record as JSON ---
def save_experiment_record(record, filename, model_dir):
    with open(os.path.join(model_dir, filename), 'w') as f:
        json.dump(record, f, indent=2)

# --- Append accuracy data to CSV ---
def append_accuracy_csv(row, filename, model_dir):
    file_path = os.path.join(model_dir, filename)
    write_header = not os.path.exists(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp', 'model', 'N', 'batch_size', 'L_irr', 'iteration', 'accuracy'])
        writer.writerow(row)

# --- Main experiment loop ---
def main():
    # Check if model configuration is set
    if LLM_BACKEND is None or MODEL_NAME is None:
        raise ValueError("LLM_BACKEND and MODEL_NAME must be set before running the script.")
    
    # Set API keys based on backend
    if LLM_BACKEND == "openai":
        os.environ['OPENAI_API_KEY'] = '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'
        API_KEY = os.getenv('OPENAI_API_KEY')
    elif LLM_BACKEND == "claude":
        os.environ['ANTHROPIC_API_KEY'] = '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'
        API_KEY = os.getenv('ANTHROPIC_API_KEY')
    elif LLM_BACKEND == "gemini":
        os.environ['GOOGLE_API_KEY'] = 'AIzaSyDQzt4GTNNaXMsBHjofJkhb5u8fOZhPE1g'
        API_KEY = os.getenv('GOOGLE_API_KEY')
    else:
        raise ValueError(f"Invalid LLM backend: {LLM_BACKEND}")
    
    # Import the LLM loader
    llm_module = importlib.import_module(f"dev.pauli_string_multiplication.direct_multiplication.utils.llm_{LLM_BACKEND}")
    query_llm = llm_module.query_llm

    model_dir = ensure_records_dir(MODEL_NAME)
    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    # Try to load irrelevant text if available
    irrelevant_text = ""
    if os.path.exists(PROMPT_IRRELEVANT_PATH) and L_irr > 0:
        with open(PROMPT_IRRELEVANT_PATH, 'r') as f:
            all_irrelevant_words = f.read().split()
            # Use up to L_irr words
            selected_words = all_irrelevant_words[:L_irr]
            irrelevant_text = (
                "\n\n<irrelevant>\n"
                "The remainder of the prompt is irrelevant to the current task. Please ignore them.\n\n"
                + ' '.join(selected_words) + 
                "\n</irrelevant>"
            )
    all_accuracies = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for it in range(num_iterations):
        # Generate questions and answers
        prompt, gt_answers = generate_questions_and_answers(N, batch_size)
        # Compose full prompt
        questions_block = prompt.split('Questions (Compute and give your answer in the same format as above):')[-1].strip()
        full_prompt = prompt_template.replace('{{QUESTIONS_BLOCK}}', questions_block)
        # Append irrelevant text if any
        if irrelevant_text:
            full_prompt += irrelevant_text
        # Query LLM
        print(f"[Iteration {it+1}] Querying LLM ({LLM_BACKEND}, {MODEL_NAME})...")
        llm_response = query_llm(full_prompt, MODEL_NAME, API_KEY)
        # Try to extract answers from the LLM response. If the LLM fails to pack answers as instructed (e.g., does not provide a Python list),
        # mark all answers as incorrect, set accuracy to 0.0, and record the error message for later analysis.
        try:
            llm_answers = extract_llm_answers(llm_response)
            if not llm_answers:
                raise ValueError("No answers found in LLM_answers list")
            
            # Process answers and calculate accuracy
            LLM_answers_pc = []
            is_correct_list = []
            
            # Initialize all answers as incorrect
            is_correct_list = [False] * len(gt_answers)
            
            # Process only the answers that follow the format
            for i in range(min(len(llm_answers), len(gt_answers))):
                try:
                    llm_term = mcp.PauliTerm(text=llm_answers[i])
                    is_correct_list[i] = (llm_term == gt_answers[i])
                except Exception as e:
                    print(f"Failed to parse answer {i+1}: {e}")
                    # Keep is_correct_list[i] as False
            
            # Calculate accuracy
            acc = sum(is_correct_list) / len(is_correct_list)
            
            error_message = None
            record = {
                'timestamp': timestamp,
                'model': MODEL_NAME,
                'N': N,
                'batch_size': batch_size,
                'iteration': it+1,
                'llm_answers': llm_answers,
                'correct_answers': [str(a) for a in gt_answers],
                'is_correct_list': is_correct_list,
                'accuracy': acc,
                'error_message': error_message,
                'irrelevant_text_length': L_irr
            }
            if SAVE_LLM_RESPONSE:
                record['llm_response'] = llm_response
                
        except Exception as e:
            print(f"Failed to extract answers: {e}")
            llm_answers = []
            is_correct_list = [False] * len(gt_answers)
            acc = 0.0
            error_message = str(e)
            record = {
                'timestamp': timestamp,
                'model': MODEL_NAME,
                'N': N,
                'batch_size': batch_size,
                'iteration': it+1,
                'llm_answers': llm_answers,
                'correct_answers': [str(a) for a in gt_answers],
                'is_correct_list': is_correct_list,
                'accuracy': acc,
                'error_message': error_message,
                'irrelevant_text_length': L_irr,
                'llm_response': llm_response
            }
        all_accuracies.append(acc)
        print(f"N = {N}, batch_size = {batch_size}, L_irr = {L_irr}, iteration = {it+1}, accuracy = {acc}")
        # Save detailed record for this iteration
        save_experiment_record(record, f'record_N{N}_batch{batch_size}_L_irr{L_irr}_iter{it+1}_{timestamp}.json', model_dir)
        # Save accuracy to CSV
        append_accuracy_csv([timestamp, MODEL_NAME, N, batch_size, L_irr, it+1, acc], 'accuracy_summary.csv', model_dir)
    print(f"Average accuracy over {num_iterations} iterations: {sum(all_accuracies)/len(all_accuracies) if all_accuracies else 0}")

if __name__ == '__main__':
    main()
