import os
import importlib.util
import importlib
import re
import json
import csv
import time
from datetime import datetime
from typing import List
import pyclifford as pc
import mcp_server as mcp
import numpy
from math import ceil
from utils.config import get_config

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
SAVE_LLM_RESPONSE = False  # Set to True to save the full LLM response in the record
L_irr = 0  # Number of words of irrelevant text to append to the prompt (0 = no irrelevant text)
TEMPERATURE = 0.0  # Temperature for LLM sampling (0.0 = deterministic, higher = more random)

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
            writer.writerow(['timestamp', 'model', 'N', 'batch_size', 'L_irr', 'temperature', 'iteration', 'accuracy', 'input_tokens', 'output_tokens', 'total_tokens'])
        writer.writerow(row)

# --- Main experiment loop ---
def main():
    # Check if model configuration is set
    if LLM_BACKEND is None or MODEL_NAME is None:
        raise ValueError("LLM_BACKEND and MODEL_NAME must be set before running the script.")
    
    # Load configuration and get API key
    config = get_config()
    config.print_config_status()  # Show config status for debugging
    
    try:
        API_KEY = config.get_api_key(LLM_BACKEND)
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("💡 Please check your config.json file or environment variables")
        return
    
    # Import the LLM loader
    llm_module = importlib.import_module(f"dev.pauli_string_multiplication.direct_multiplication.utils.llm_{LLM_BACKEND}")
    query_llm = llm_module.query_llm

    # Use configured paths
    records_base_dir = config.get_path('records_base_dir')
    model_dir = os.path.join(records_base_dir, MODEL_NAME)
    os.makedirs(model_dir, exist_ok=True)
    
    prompt_template = load_prompt_template(config.get_path('prompt_template'))
    
    # Try to load irrelevant text if available
    irrelevant_text = ""
    irrelevant_path = config.get_path('prompt_irrelevant')
    if os.path.exists(irrelevant_path) and L_irr > 0:
        with open(irrelevant_path, 'r') as f:
            all_irrelevant_words = f.read().split()
            # Use up to L_irr words
            selected_words = all_irrelevant_words[:L_irr]
            irrelevant_text = (
                # "\n\n<irrelevant>\n"
                # "The remainder of the prompt is irrelevant to the current task. Please ignore them.\n\n"
                # + ' '.join(selected_words) + 
                # "\n</irrelevant>"
                "\n\nAdditional context loaded from knowledge base: "
                + ' '.join(selected_words) + 
                "\n\nNow proceed with the primary task:"
            )
    all_accuracies = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for it in range(num_iterations):
        # Query LLM with retry (generate fresh questions each attempt)
        print(f"[Iteration {it+1}] Querying LLM ({LLM_BACKEND}, {MODEL_NAME})...")
        
        while True:
            try:
                # Generate fresh questions for each attempt
                prompt, gt_answers = generate_questions_and_answers(N, batch_size)
                # Compose full prompt
                questions_block = prompt.split('Questions (Compute and give your answer in the same format as above):')[-1].strip()
                full_prompt = prompt_template.replace('{{QUESTIONS_BLOCK}}', questions_block)
                # Append irrelevant text if any
                if irrelevant_text:
                    full_prompt += irrelevant_text
                
                llm_response, token_metadata = query_llm(full_prompt, MODEL_NAME, API_KEY, temperature=TEMPERATURE)
                print(f"[Iteration {it+1}] Token usage: {token_metadata['input_tokens']} in, {token_metadata['output_tokens']} out")
                time.sleep(1)  # Default delay after successful call
                break  # Success, exit loop
            except Exception as e:
                print(f"[Iteration {it+1}] ❌ API call failed: {e}")
                print(f"[Iteration {it+1}] 🔄 Generating new questions and retrying in 15s...")
                time.sleep(15)
        
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
                'irrelevant_text_length': L_irr,
                'input_tokens': token_metadata['input_tokens'],
                'output_tokens': token_metadata['output_tokens'],
                'total_tokens': token_metadata['total_tokens'],
                'temperature': TEMPERATURE
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
                'input_tokens': token_metadata['input_tokens'],
                'output_tokens': token_metadata['output_tokens'],
                'total_tokens': token_metadata['total_tokens'],
                'llm_response': llm_response,
                'temperature': TEMPERATURE
            }
        all_accuracies.append(acc)
        print(f"N = {N}, batch_size = {batch_size}, L_irr = {L_irr}, temperature = {TEMPERATURE}, iteration = {it+1}, accuracy = {acc}")
        # Save detailed record for this iteration
        save_experiment_record(record, f'record_N{N}_batch{batch_size}_L_irr{L_irr}_temp{TEMPERATURE}_iter{it+1}_{timestamp}.json', model_dir)
        # Save accuracy and token usage to CSV
        append_accuracy_csv([timestamp, MODEL_NAME, N, batch_size, L_irr, TEMPERATURE, it+1, acc, 
                           token_metadata['input_tokens'], token_metadata['output_tokens'], token_metadata['total_tokens']], 
                          'accuracy_summary.csv', model_dir)
    print(f"Average accuracy over {num_iterations} iterations: {sum(all_accuracies)/len(all_accuracies) if all_accuracies else 0}")

if __name__ == '__main__':
    main()
