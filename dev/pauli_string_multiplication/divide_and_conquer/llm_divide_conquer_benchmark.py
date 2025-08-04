import os
import sys
import re
import json
import time
import csv
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import pyclifford as pc
import mcp_server as mcp

# === CONFIGURATION ===
# Model configuration
MODEL_BACKEND = "gemini"  # Options: "gemini", "openai", "claude"
MODEL_NAME = "gemini-2.5-flash-preview-04-17-thinking"
SAVE_LLM_RESPONSES = True
RECORDS_BASE_DIR = 'records'

# Phase computation strategy
USE_PHASE_ACCUMULATION = True  # True: accumulate in Phase 2, False: compute all phases in Phase 3

# === CSV TRACKING ===
def append_accuracy_csv(row, filename, model_dir):
    """Append accuracy data to CSV file."""
    file_path = os.path.join(model_dir, filename)
    write_header = not os.path.exists(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp', 'model', 'N', 'chunk_size', 'batch_size', 'iteration', 'use_phase_accumulation', 'workflow_success', 'success_rate', 'duration_seconds', 'llm_calls'])
        writer.writerow(row)

# === API KEY MANAGEMENT ===
def get_api_key_for_backend(backend: str) -> str:
    """Get the correct API key for the specified backend."""
    if backend == "gemini":
        return "AIzaSyDQzt4GTNNaXMsBHjofJkhb5u8fOZhPE1g"
    elif backend == "openai":
        return "sk-proj-your-openai-key-here"  # Replace with your OpenAI API key
    elif backend == "claude":
        return "sk-ant-your-claude-key-here"  # Replace with your Claude API key
    else:
        raise ValueError(f"Unknown backend: {backend}")

# === PAULI STRING UTILITIES ===

def generate_random_pauli_strings(N: int) -> Tuple[str, str]:
    """Generate two random Pauli strings of length N."""
    paulis = ['I', 'X', 'Y', 'Z']
    # Use '+' for positive identity phase instead of empty string for clarity
    phases = ['+', 'i', '-', '-i']
    
    str1_ops = ''.join(np.random.choice(paulis) for _ in range(N))
    str2_ops = ''.join(np.random.choice(paulis) for _ in range(N))
    
    phase1 = np.random.choice(phases)
    phase2 = np.random.choice(phases)
    
    return f'{phase1}{str1_ops}', f'{phase2}{str2_ops}'

def multiply_pauli_strings(str1: str, str2: str) -> str:
    """Multiply two Pauli strings using PyClifford."""
    try:
        op1 = pc.pauli(str1)
        op2 = pc.pauli(str2)
        result = op1 @ op2
        return mcp.PauliTerm.from_obj(result).text
    except Exception as e:
        print(f"Error multiplying Pauli strings: {e}")
        return "ERROR"

def semantic_pauli_equal(llm_result: str, expected: str) -> bool:
    """Check if two Pauli string representations are semantically equal."""
    try:
        llm_term = mcp.PauliTerm(text=llm_result)
        expected_term = mcp.PauliTerm(text=expected)
        return llm_term == expected_term
    except Exception:
        return False

# === LLM INTERFACE ===

def get_llm_response(prompt: str, model: str, backend: str) -> str:
    """Get LLM response using the appropriate backend module."""
    api_key = get_api_key_for_backend(backend)
    
    if backend == "gemini":
        import llm_gemini
        return llm_gemini.query_llm(prompt, model, api_key)
    elif backend == "openai":
        import llm_openai
        return llm_openai.query_llm(prompt, model, api_key)
    elif backend == "claude":
        import llm_claude
        return llm_claude.query_llm(prompt, model, api_key)
    else:
        raise ValueError(f"Unknown backend: {backend}")

# === PARSING FUNCTIONS ===

def extract_batch_decomposition(response: str) -> Dict[str, Any]:
    """Extract batch decomposition from LLM response using new simple format."""
    result = {'success': False, 'chunking_strategy': None, 'chunk_ranges': [], 'problems': []}
    
    # Extract content within XML tags
    pattern = r'<batch_decomposition>(.*?)</batch_decomposition>'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return result
    
    content = match.group(1)
    
    # Extract chunking strategy
    strategy_match = re.search(r'STRATEGY\s*=\s*"([^"]+)"', content)
    if strategy_match:
        result['chunking_strategy'] = strategy_match.group(1).strip()
    
    # Extract chunk ranges using simple pattern
    ranges_match = re.search(r'CHUNK_RANGES\s*=\s*\[([^\]]+)\]', content)
    if ranges_match:
        ranges_str = ranges_match.group(1)
        result['chunk_ranges'] = [r.strip(' "\'') for r in ranges_str.split(',')]
    
    # Extract problem chunks using new flat format
    # Pattern: PROBLEM_N_STR1_CHUNKS = ["chunk1", "chunk2", ...]
    problem_data = {}
    problem_pattern = r'PROBLEM_(\d+)_STR([12])_CHUNKS\s*=\s*\[([^\]]+)\]'
    
    for match in re.finditer(problem_pattern, content):
        prob_num = int(match.group(1))
        str_num = match.group(2)
        chunks_str = match.group(3)
        
        # Parse chunk list with better handling of quoted strings
        chunks = []
        # Split by ", " but handle quotes properly
        if '", "' in chunks_str:
            raw_chunks = chunks_str.split('", "')
        else:
            raw_chunks = [chunks_str]
        
        for chunk in raw_chunks:
            clean_chunk = chunk.strip(' "\'')
            if clean_chunk:
                chunks.append(clean_chunk)
        
        if prob_num not in problem_data:
            problem_data[prob_num] = {'problem_number': prob_num, 'chunks': []}
        
        # Store chunks by str number
        for i, chunk_text in enumerate(chunks):
            chunk_num = i + 1
            # Find or create chunk entry
            chunk_entry = None
            for existing_chunk in problem_data[prob_num]['chunks']:
                if existing_chunk['chunk_number'] == chunk_num:
                    chunk_entry = existing_chunk
                    break
            
            if not chunk_entry:
                chunk_entry = {
                    'chunk_number': chunk_num,
                    'str1': '',
                    'str2': '',
                    'definition': ''
                }
                problem_data[prob_num]['chunks'].append(chunk_entry)
            
            # Set the appropriate string
            chunk_entry[f'str{str_num}'] = chunk_text
    
    # Convert to expected format and create definitions
    for prob_num, prob_data in problem_data.items():
        for chunk in prob_data['chunks']:
            if chunk['str1'] and chunk['str2']:
                chunk['definition'] = f"({chunk['str1']}) ({chunk['str2']})"
        
        # Sort chunks by chunk_number
        prob_data['chunks'].sort(key=lambda x: x.get('chunk_number', 0))
        result['problems'].append(prob_data)
    
    # Sort problems by problem_number
    result['problems'].sort(key=lambda x: x.get('problem_number', 0))
    
    if result['problems'] and result['chunk_ranges']:
        result['success'] = True
    
    return result

def extract_batch_chunk_results(response: str, chunk_number: int) -> List[Dict]:
    """Extract batch chunk computation results from LLM response using new simple format."""
    results = []
    
    # Extract content within XML tags
    pattern = r'<batch_chunk_results>(.*?)</batch_chunk_results>'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return results
    
    content = match.group(1)
    
    # Extract chunk results list using new format
    # Pattern: CHUNK_N_RESULTS = ["PROBLEM_1: result PHASE_INFO: phase", ...]
    results_pattern = f'CHUNK_{chunk_number}_RESULTS\\s*=\\s*\\[([^\\]]+)\\]'
    results_match = re.search(results_pattern, content, re.DOTALL)
    if not results_match:
        return results
    
    results_content = results_match.group(1)
    
    # Parse each result line with new simple format
    # Pattern: "PROBLEM_X: result PHASE_INFO: phase"
    line_pattern = r'"PROBLEM_(\d+):\s*([^P]+?)\s*PHASE_INFO:\s*([^"]+)"'
    
    for line_match in re.finditer(line_pattern, results_content):
        prob_num = int(line_match.group(1))
        result_str = line_match.group(2).strip()
        phase_info = line_match.group(3).strip()
        
        # Create result with unified field names for compatibility
        result_dict = {
            'problem_number': prob_num,
            'result': result_str,
            'phase_info': phase_info  # Unified field name
        }
        
        # Add strategy-specific field for backward compatibility
        if USE_PHASE_ACCUMULATION:
            result_dict['new_accumulated_phase'] = phase_info
        else:
            result_dict['chunk_phase'] = phase_info
        
        results.append(result_dict)
    
    return results

def extract_batch_final_results(response: str) -> List[Dict]:
    """Extract batch final results from LLM response using new simple format."""
    results = []
    
    # Extract content within XML tags
    pattern = r'<batch_combination>(.*?)</batch_combination>'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return results
    
    content = match.group(1)
    
    # Extract final answers list using new format
    # Pattern: FINAL_ANSWERS = ["PROBLEM_1: result", "PROBLEM_2: result", ...]
    answers_pattern = r'FINAL_ANSWERS\s*=\s*\[([^\]]+)\]'
    answers_match = re.search(answers_pattern, content, re.DOTALL)
    if not answers_match:
        return results
    
    answers_content = answers_match.group(1)
    
    # Parse each answer line with new simple format
    # Pattern: "PROBLEM_X: final_result"
    line_pattern = r'"PROBLEM_(\d+):\s*([^"]+)"'
    
    for line_match in re.finditer(line_pattern, answers_content):
        prob_num = int(line_match.group(1))
        final_result = line_match.group(2).strip()
        
        results.append({
            'problem_number': prob_num,
            'result': final_result
        })
    
    return results

# === CORE WORKFLOW ===

def run_batch_divide_conquer_workflow(problems: List[Dict], chunk_size: int, N: int) -> Dict[str, Any]:
    """
    Run the three-phase divide-and-conquer workflow:
    Phase 1: Decompose all problems into chunks
    Phase 2: Compute each chunk for all problems  
    Phase 3: Combine chunk results into final answers
    """
    batch_size = len(problems)
    print(f"  Starting divide-and-conquer for {batch_size} problems, N={N}, chunk_size={chunk_size}")
    
    workflow_result = {
        'success': False,
        'llm_calls': 0,
        'decomposition': None,
        'chunk_results': {},
        'final_results': [],
        'error_message': None
    }
    
    if SAVE_LLM_RESPONSES:
        workflow_result.update({
            'phase1_prompt': None,
            'phase1_response': None,
            'phase2_prompts_and_responses': [],
            'phase3_prompt': None,
            'phase3_response': None
        })
    
    try:
        # === PHASE 1: Decomposition ===
        print(f"=== PHASE 1: Decomposition ===")
        print(f"    Decomposing {batch_size} problems...")
        
        # Convert compact format to indexed format for LLM display
        def to_indexed_format(pauli_string):
            """Convert compact IXYZ string to indexed X_0 Y_1 Z_2 format."""
            # Extract phase - handle all four phase factors: +, i, -, -i
            phase = ""
            ops = pauli_string
            # Check in order to avoid prefix conflicts (longer first)
            for p in ['-i', '+', 'i', '-']:
                if pauli_string.startswith(p):
                    phase = p
                    ops = pauli_string[len(p):]
                    break
            
            # Add indices - show ALL operators (even I)
            # Use simple underscore format without braces to avoid formatting conflicts
            indexed_ops = []
            for i, op in enumerate(ops):
                indexed_ops.append(f"{op}_{i}")
            
            result = " ".join(indexed_ops)
            return (phase + " " + result).strip() if phase else result
        
        problems_block = "\n".join([
            f"Problem {i+1}:\n  String 1: {to_indexed_format(prob['str1'])}\n  String 2: {to_indexed_format(prob['str2'])}"
            for i, prob in enumerate(problems)
        ])
        
        with open("prompt_round1_decomposition.txt", "r") as f:
            decomposition_prompt = f.read().format(
                batch_size=batch_size,
                problems_block=problems_block,
                N=N,
                chunk_size=chunk_size
            )
        
        decomposition_response = get_llm_response(decomposition_prompt, MODEL_NAME, MODEL_BACKEND)
        workflow_result['llm_calls'] += 1
        
        if SAVE_LLM_RESPONSES:
            workflow_result['phase1_prompt'] = decomposition_prompt
            workflow_result['phase1_response'] = decomposition_response
        
        decomposition_result = extract_batch_decomposition(decomposition_response)
        if not decomposition_result['success']:
            raise Exception("Failed to extract batch decomposition")
        
        workflow_result['decomposition'] = decomposition_result
        num_chunks = len(decomposition_result['chunk_ranges'])
        print(f"    Decomposed into {num_chunks} chunks: {decomposition_result['chunk_ranges']}")
        
        # === PHASE 2: CHUNK COMPUTATION ===
        print("=== PHASE 2: CHUNK COMPUTATION ===")
        
        # Initialize phase tracking based on strategy
        if USE_PHASE_ACCUMULATION:
            accumulated_phases = {}
            for problem in decomposition_result['problems']:
                accumulated_phases[problem['problem_number']] = "1"
            print(f"    Phase 2: Computing {num_chunks} chunks with LLM phase accumulation...")
            chunk_template_file = "prompt_round2_chunk_calculation.txt"
        else:
            chunk_phases = {}  # Track individual chunk phases
            print(f"    Phase 2: Computing {num_chunks} chunks independently...")
            chunk_template_file = "prompt_round2_chunk_calculation_independent.txt"
        
        with open(chunk_template_file, "r") as f:
            chunk_prompt_template = f.read()
        
        for chunk_num in range(1, num_chunks + 1):
            print(f"      Computing chunk {chunk_num}...")
            
            # Calculate qubit range for this chunk
            start_qubit = (chunk_num - 1) * chunk_size
            end_qubit = min(start_qubit + chunk_size - 1, N - 1)
            qubit_range = f"{start_qubit}-{end_qubit}"
            
            # Format chunk problems
            chunk_problems_list = []
            for problem in decomposition_result['problems']:
                for chunk in problem['chunks']:
                    if chunk['chunk_number'] == chunk_num:
                        chunk_problems_list.append(
                            f"Problem {problem['problem_number']}: {chunk['definition']}"
                        )
                        break
            
            chunk_problems_block = "\n".join(chunk_problems_list)
            
            # Format prompt based on strategy
            if USE_PHASE_ACCUMULATION:
                # Format accumulated phases for this chunk
                accumulated_phases_lines = []
                accumulated_phases_lines.append("PREVIOUS ACCUMULATED PHASES:")
                for prob_num in sorted(accumulated_phases.keys()):
                    accumulated_phases_lines.append(f"  Problem {prob_num}: {accumulated_phases[prob_num]}")
                accumulated_phases_block = "\n".join(accumulated_phases_lines)
                
                chunk_prompt = chunk_prompt_template.format(
                    chunk_number=chunk_num,
                    batch_size=batch_size,
                    chunk_problems_block=chunk_problems_block,
                    qubit_range=qubit_range,
                    accumulated_phases_block=accumulated_phases_block
                )
            else:
                # Independent chunk calculation - no accumulated phases
                chunk_prompt = chunk_prompt_template.format(
                    chunk_number=chunk_num,
                    batch_size=batch_size,
                    chunk_problems_block=chunk_problems_block,
                    qubit_range=qubit_range
                )
            
            chunk_response = get_llm_response(chunk_prompt, MODEL_NAME, MODEL_BACKEND)
            workflow_result['llm_calls'] += 1
            
            if SAVE_LLM_RESPONSES:
                workflow_result['phase2_prompts_and_responses'].append({
                    'chunk_number': chunk_num,
                    'prompt': chunk_prompt,
                    'response': chunk_response
                })
            
            chunk_results = extract_batch_chunk_results(chunk_response, chunk_num)
            if not chunk_results:
                raise Exception(f"Failed to extract results for chunk {chunk_num}")
            
            # Update phase tracking based on strategy
            if USE_PHASE_ACCUMULATION:
                for result in chunk_results:
                    prob_num = result['problem_number']
                    new_phase = result['new_accumulated_phase']
                    print(f"      Problem {prob_num}: accumulated phase updated from {accumulated_phases[prob_num]} to {new_phase}")
                    accumulated_phases[prob_num] = new_phase
            else:
                for result in chunk_results:
                    prob_num = result['problem_number']
                    chunk_phase = result['chunk_phase']
                    if prob_num not in chunk_phases:
                        chunk_phases[prob_num] = []
                    chunk_phases[prob_num].append(chunk_phase)
                    print(f"      Problem {prob_num}: chunk {chunk_num} phase = {chunk_phase}")
            
            workflow_result['chunk_results'][chunk_num] = chunk_results
        
        # === PHASE 3: COMBINATION ===
        print("=== PHASE 3: COMBINATION ===")
        
        # Format all chunk results
        all_chunk_results_lines = []
        for chunk_num in range(1, num_chunks + 1):
            all_chunk_results_lines.append(f"Chunk {chunk_num} Results:")
            for result in workflow_result['chunk_results'][chunk_num]:
                all_chunk_results_lines.append(
                    f"  Problem {result['problem_number']}: {result['result']}"
                )
            all_chunk_results_lines.append("")
        
        all_chunk_results_block = "\n".join(all_chunk_results_lines)
        
        # Format Phase 3 prompt based on strategy
        if USE_PHASE_ACCUMULATION:
            # Format final accumulated phases for Phase 3
            accumulated_phases_lines = []
            accumulated_phases_lines.append("FINAL ACCUMULATED PHASES:")
            for prob_num in sorted(accumulated_phases.keys()):
                phase = accumulated_phases[prob_num]
                accumulated_phases_lines.append(f"  Problem {prob_num}: {phase}")
            accumulated_phases_block = "\n".join(accumulated_phases_lines)
            
            print(f"    Final accumulated phases: {accumulated_phases}")
            
            with open("prompt_round3_combination.txt", "r") as f:
                combination_prompt = f.read().format(
                    num_chunks=num_chunks,
                    batch_size=batch_size,
                    original_problems_block=problems_block,
                    all_chunk_results_block=all_chunk_results_block,
                    accumulated_phases_block=accumulated_phases_block
                )
        else:
            print(f"    Final chunk phases: {chunk_phases}")
            
            with open("prompt_round3_combination_final_calc.txt", "r") as f:
                combination_prompt = f.read().format(
                    num_chunks=num_chunks,
                    batch_size=batch_size,
                    original_problems_block=problems_block,
                    all_chunk_results_block=all_chunk_results_block
                )
        
        combination_response = get_llm_response(combination_prompt, MODEL_NAME, MODEL_BACKEND)
        workflow_result['llm_calls'] += 1
        
        if SAVE_LLM_RESPONSES:
            workflow_result['phase3_prompt'] = combination_prompt
            workflow_result['phase3_response'] = combination_response
        
        final_results = extract_batch_final_results(combination_response)
        if not final_results:
            raise Exception("Failed to extract final batch results")
        
        workflow_result['final_results'] = final_results
        workflow_result['success'] = True
        
        print(f"    Workflow completed! Total LLM calls: {workflow_result['llm_calls']}")
        
    except Exception as e:
        workflow_result['error_message'] = str(e)
        print(f"    Workflow failed: {e}")
    
    return workflow_result

# === EXPERIMENT MANAGEMENT ===

def run_experiment(N_list, chunk_size_list, batch_size_list, num_iterations_list):
    """Run experiments with given parameters and save results."""
    
    print("=== Divide-and-Conquer Pauli String Multiplication Benchmark ===")
    print(f"Problem sizes: {N_list}")
    print(f"Chunk sizes: {chunk_size_list}")
    print(f"Batch sizes: {batch_size_list}")
    print(f"Iterations: {num_iterations_list}")
    print(f"Model: {MODEL_NAME}")
    print()
    
    # Ensure records directory exists
    model_dir = os.path.join(RECORDS_BASE_DIR, MODEL_NAME)
    os.makedirs(model_dir, exist_ok=True)
    
    all_results = []
    
    for N in N_list:
        for chunk_size in chunk_size_list:
            for batch_size in batch_size_list:
                for num_iterations in num_iterations_list:
                    if chunk_size >= N:
                        print(f"Skipping N={N}, chunk_size={chunk_size} (chunk too large)")
                        continue
                        
                    print(f"--- Testing N={N}, chunk_size={chunk_size}, batch_size={batch_size}, iterations={num_iterations} ---")
                    
                    for iteration in range(num_iterations):
                        print(f"Iteration {iteration + 1}/{num_iterations}")
                        
                        # Generate test problems
                        problems = []
                        for _ in range(batch_size):
                            str1, str2 = generate_random_pauli_strings(N)
                            problems.append({
                                'str1': str1,
                                'str2': str2,
                                'expected_result': multiply_pauli_strings(str1, str2)
                            })
                        
                        # Create record ID
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        record_id = f"batch={batch_size}_N={N}_chunk={chunk_size}_iter={iteration+1}_{timestamp}"
                        
                        print(f"  Record ID: {record_id}")
                        print(f"  Problems:")
                        for i, prob in enumerate(problems):
                            print(f"    {i+1}. {prob['str1']} × {prob['str2']} = {prob['expected_result']}")
                        
                        # Run workflow
                        start_time = time.time()
                        workflow_result = run_batch_divide_conquer_workflow(problems, chunk_size, N)
                        end_time = time.time()
                        
                        # Handle None workflow_result (workflow failed completely)
                        if workflow_result is None:
                            workflow_result = {
                                'success': False, 
                                'llm_calls': 0, 
                                'error_message': 'Workflow failed completely',
                                'decomposition': None,
                                'chunk_results': {},
                                'final_results': []
                            }
                        
                        # Check results
                        correct_count = 0
                        workflow_success = workflow_result['success']
                        
                        if workflow_success:
                            print(f"  Results:")
                            for result in workflow_result['final_results']:
                                prob_num = result['problem_number']
                                llm_result = result['result']
                                expected = problems[prob_num - 1]['expected_result']
                                
                                is_correct = semantic_pauli_equal(llm_result, expected)
                                if is_correct:
                                    correct_count += 1
                                
                                status = "✓" if is_correct else "✗"
                                print(f"    {prob_num}. {llm_result} {status} (expected: {expected})")
                        else:
                            print(f"  Workflow failed: {workflow_result.get('error_message', 'Unknown error')}")
                        
                        # Calculate success rate: 0 for failed workflows, actual rate for successful ones
                        success_rate = correct_count / batch_size if workflow_success and batch_size > 0 else 0
                        
                        # Save accuracy to CSV for tracking
                        timestamp_str = timestamp
                        append_accuracy_csv([timestamp_str, MODEL_NAME, N, chunk_size, batch_size, iteration + 1, USE_PHASE_ACCUMULATION, workflow_success, success_rate, end_time - start_time, workflow_result['llm_calls']], 'accuracy_summary.csv', model_dir)
                        
                        # Create problems_detailed format for easy debugging
                        problems_detailed = []
                        for i, prob in enumerate(problems):
                            prob_num = i + 1
                            
                            # Get decomposition for this problem
                            prob_decomposition = None
                            if workflow_result['success'] and 'decomposition' in workflow_result:
                                for decomp_prob in workflow_result['decomposition']['problems']:
                                    if decomp_prob['problem_number'] == prob_num:
                                        prob_decomposition = decomp_prob
                                        break
                            
                            # Extract LLM decomposition
                            decomposition_data = workflow_result.get('decomposition') or {}
                            llm_decomposition = {
                                "str1_chunks": [],
                                "str2_chunks": [],
                                "chunk_definitions": [],
                                "chunk_ranges": decomposition_data.get('chunk_ranges', [])
                            }
                            
                            if prob_decomposition:
                                for chunk in prob_decomposition['chunks']:
                                    llm_decomposition["str1_chunks"].append(chunk['str1'])
                                    llm_decomposition["str2_chunks"].append(chunk['str2'])
                                    llm_decomposition["chunk_definitions"].append(chunk['definition'])
                            
                            # Extract LLM chunk results for this problem
                            llm_computed_results = []
                            if workflow_result['success'] and 'chunk_results' in workflow_result:
                                for chunk_num in sorted(workflow_result['chunk_results'].keys()):
                                    for chunk_result in workflow_result['chunk_results'][chunk_num]:
                                        if chunk_result['problem_number'] == prob_num:
                                            llm_computed_results.append(chunk_result['result'])
                                            break
                            
                            # Extract LLM final result for this problem
                            llm_final_result = None
                            if workflow_result['success'] and 'final_results' in workflow_result:
                                for final_result in workflow_result['final_results']:
                                    if final_result['problem_number'] == prob_num:
                                        llm_final_result = final_result['result']
                                        break
                            
                            # Check correctness
                            is_correct = semantic_pauli_equal(llm_final_result or "", prob['expected_result'])
                            
                            problem_detail = {
                                "problem_number": prob_num,
                                "original_problem": f"Input: ({prob['str1']}) * ({prob['str2']})",
                                "expected_result": prob['expected_result'],
                                "llm_decomposition": llm_decomposition,
                                "llm_computed_results": llm_computed_results,
                                "llm_combined_result": llm_final_result,
                                "verification_summary": {
                                    "is_correct": is_correct,
                                    "expected": prob['expected_result'],
                                    "actual": llm_final_result or "None"
                                }
                            }
                            
                            problems_detailed.append(problem_detail)
                        
                        # Create new record format
                        record = {
                            'record_id': record_id,
                            'timestamp': timestamp,
                            'configuration': {
                                'N': N,
                                'chunk_size': chunk_size,
                                'batch_size': batch_size,
                                'iteration': iteration + 1,
                                'model_name': MODEL_NAME,
                                'model_backend': MODEL_BACKEND,
                                'use_phase_accumulation': USE_PHASE_ACCUMULATION
                            },
                            'problems_detailed': problems_detailed,
                            'workflow_summary': {
                                'total_llm_calls': workflow_result['llm_calls'],
                                'success_rate': success_rate,
                                'correct_count': correct_count,
                                'total_count': batch_size,
                                'duration_seconds': end_time - start_time
                            }
                        }
                        
                        # Add raw LLM prompts and responses if saving them
                        if SAVE_LLM_RESPONSES:
                            record['raw_llm_interactions'] = {}
                            if 'phase1_prompt' in workflow_result and 'phase1_response' in workflow_result:
                                record['raw_llm_interactions']['phase1_decomposition'] = {
                                    'prompt': workflow_result['phase1_prompt'],
                                    'response': workflow_result['phase1_response']
                                }
                            if 'phase2_prompts_and_responses' in workflow_result:
                                record['raw_llm_interactions']['phase2_chunk_calculations'] = workflow_result['phase2_prompts_and_responses']
                            if 'phase3_prompt' in workflow_result and 'phase3_response' in workflow_result:
                                record['raw_llm_interactions']['phase3_combination'] = {
                                    'prompt': workflow_result['phase3_prompt'],
                                    'response': workflow_result['phase3_response']
                                }
                        
                        # Save to file
                        filename = f"{'full' if SAVE_LLM_RESPONSES else 'light'}_{record_id}.json"
                        filepath = os.path.join(model_dir, filename)
                        
                        with open(filepath, 'w') as f:
                            json.dump(record, f, indent=2)
                        
                        print(f"  Record saved: {filename}")
                        
                        # Add to results
                        all_results.append({
                            'N': N,
                            'chunk_size': chunk_size,
                            'batch_size': batch_size,
                            'iteration': iteration + 1,
                            'success_rate': success_rate,
                            'duration': end_time - start_time,
                            'llm_calls': workflow_result['llm_calls'],
                            'workflow_success': workflow_success,
                            'record_id': record_id
                        })
                        
                        print(f"  Duration: {end_time - start_time:.1f}s")
                        print(f"  LLM calls: {workflow_result['llm_calls']}")
                        print(f"  Workflow Success: {workflow_success}")
                        if workflow_success:
                            print(f"  Success Rate: {success_rate:.1%} ({correct_count}/{batch_size})")
                        else:
                            print(f"  Success Rate: 0.0% (workflow failed)")
                        print()
    
    return all_results

# === MAIN FUNCTION ===

def main():
    """Test new parsing approach for improved workflow success rate."""
    global USE_PHASE_ACCUMULATION
    
    # Test parameters: N=7, chunk_size=2, batch_size=3, 2 iterations each
    N_list = [7]
    chunk_size_list = [2] 
    batch_size_list = [3]
    num_iterations_list = [5]  # Testing 5 iterations to get better statistics
    
    print("=== Testing NEW PARSING APPROACH ===")
    print("Parameters: N=7, chunk_size=2, batch_size=3, 5 iterations each")
    print("Expected chunks: [0-1], [2-3], [4-5], [6-6] (last chunk has only 1 qubit)")
    print("Goal: Achieve 100% workflow success rate with improved parsing")
    print()
    
    # Test both strategies
    for strategy_name, use_accumulation in [("Phase_Accumulation", True), ("Independent_Phase", False)]:
        USE_PHASE_ACCUMULATION = use_accumulation
        print(f"\n=== Testing {strategy_name} Strategy ===")
        run_experiment(N_list, chunk_size_list, batch_size_list, num_iterations_list)

if __name__ == "__main__":
    main()
