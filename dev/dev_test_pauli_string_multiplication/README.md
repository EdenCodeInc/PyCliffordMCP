# LLM Pauli String Multiplication Benchmark

This project provides a modular framework to benchmark large language models (LLMs) on Pauli string multiplication tasks, evaluating their ability to follow quantum algebra rules. It is designed for easy extensibility, reproducibility, and analysis across different LLMs (OpenAI, Gemini, Claude, etc.).

## Features
- **Configurable LLM backend and model** via `config.json`
- **Built-in API key configuration** for OpenAI, Claude, and Gemini
- **External prompt template** for easy prompt engineering
- **Irrelevant text injection** to test robustness (L_irr parameter)
- **Randomized question generation** for robust testing
- **Dynamic LLM loader**: add new LLMs by dropping in a loader file
- **Robust answer extraction and error handling**
- **Per-model experiment records and accuracy summaries**
- **Ready-to-plot CSV output for comprehensive analysis**

## Directory Structure
```
dev/dev_test_pauli_string_multiplication/
├── llm_pauli_benchmark.py         # Main experiment script
├── llm_openai.py                  # OpenAI LLM loader
├── llm_claude.py                  # Claude LLM loader
├── llm_gemini.py                  # Gemini LLM loader
├── config.json                    # Config file for backend/model selection
├── prompt_template.txt            # Prompt template for LLM
├── prompt_irrelevant.txt          # Irrelevant text for robustness testing
├── test.ipynb                     # Jupyter notebook for analysis and plotting
├── records/                       # Experiment results (per model)
│   └── <model_name>/
│       ├── record_...json         # Per-run detailed records
│       └── accuracy_summary.csv   # Summary for plotting
└── ...
```

## Setup
1. **Install dependencies** (in your Python/conda environment):
   ```sh
   pip install openai anthropic google-generativeai numpy pandas matplotlib pyclifford
   # Or use conda for numpy/pandas/matplotlib if preferred
   ```

2. **Configure API keys** (built into the code):
   - Edit `llm_pauli_benchmark.py` to update your API keys in the "Set API keys" section
   - The current API keys are already configured for OpenAI, Claude, and Gemini

3. **Configure your experiment**:
   - Edit `config.json` to select the LLM backend and model:
     ```json
     {
       "llm_backend": "openai",
       "model_name": "gpt-3.5-turbo"
     }
     ```
   - Edit `prompt_template.txt` to change the prompt instructions/examples.

## Running Experiments
### Basic Usage
You can run experiments from a Jupyter notebook or Python script:

```python
import sys, os
os.chdir('/Users/hwanda/Desktop/EdenCode/projects/PyCliffordMCP-main')
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import and configure
import dev.dev_test_pauli_string_multiplication.llm_pauli_benchmark as benchmark

# Set parameters
benchmark.N = 3                    # Pauli string length
benchmark.batch_size = 10          # Questions per prompt
benchmark.num_iterations = 3       # Number of prompt rounds
benchmark.L_irr = 0               # Words of irrelevant text (0 = none)

# Run experiment
benchmark.main()
```

### Parameter Sweeps
To sweep over parameter ranges:
```python
from itertools import product

N_values = [1, 2, 3, 4, 5]
batch_sizes = [10, 20]
num_iterations_list = [5]
L_irr_values = [0, 50, 100]  # Test robustness with irrelevant text

for N, batch_size, num_iterations, L_irr in product(N_values, batch_sizes, num_iterations_list, L_irr_values):
    benchmark.N = N
    benchmark.batch_size = batch_size
    benchmark.num_iterations = num_iterations
    benchmark.L_irr = L_irr
    benchmark.main()
```

## Plotting Results
### Multi-Model Comparison
Compare different models across various parameters:

```python
import os
import pandas as pd
import matplotlib.pyplot as plt

# Model comparison
model_names = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o", "claude-3-opus-20240229", "claude-sonnet-4-20250514"]
records_base = "dev/dev_test_pauli_string_multiplication/records"

plt.figure(figsize=(8, 5))

for model_name in model_names:
    model_dir = os.path.join(records_base, model_name)
    csv_path = os.path.join(model_dir, "accuracy_summary.csv")
    if not os.path.isfile(csv_path):
        print(f"Warning: No data for model {model_name}")
        continue
    
    df = pd.read_csv(csv_path)
    batch_sizes = sorted(df['batch_size'].unique())
    N_values = sorted(df['N'].unique())
    L_irr_values = sorted(df['L_irr'].unique()) if 'L_irr' in df.columns else [0]
    
    # Pick specific parameters to plot
    batch_size = batch_sizes[0]
    L_irr_value = L_irr_values[0]
    
    # Average over all iterations for each N
    means = []
    stds = []
    for N in N_values:
        accs = df[(df['batch_size'] == batch_size) & (df['L_irr'] == L_irr_value) & (df['N'] == N)]['accuracy']
        means.append(accs.mean() if not accs.empty else float('nan'))
        stds.append(accs.std() if not accs.empty else 0)
    
    plt.errorbar(N_values, means, yerr=stds, marker='o', capsize=4, 
                label=f"{model_name} (batch_size={batch_size}, L_irr={L_irr_value})")

plt.xlabel("N (Pauli string length)")
plt.ylabel("Mean Accuracy (averaged over all iterations)")
plt.title("LLM Pauli Benchmark Comparison")
plt.legend(title="Models:")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

### Standard Error Analysis
For more accurate error bars using standard error of the mean:

```python
# Use standard error instead of standard deviation
for N in N_values:
    accs = df[(df['batch_size'] == batch_size) & (df['N'] == N)]['accuracy']
    num_iterations = len(accs)
    effective_sample_size = batch_size * num_iterations
    
    mean_acc = accs.mean() if not accs.empty else float('nan')
    std_acc = accs.std() if not accs.empty else 0
    std_error = std_acc / np.sqrt(effective_sample_size) if effective_sample_size > 0 else 0
    
    means.append(mean_acc)
    stds.append(std_error)  # Standard error of the mean
```

## Key Parameters
- **N**: Length of Pauli strings (complexity of the problem)
- **batch_size**: Number of questions per LLM prompt
- **num_iterations**: Number of independent experimental runs
- **L_irr**: Number of words of irrelevant text added to test robustness (0 = no irrelevant text)

## Output Data Structure
Each experiment generates:
- **JSON records**: Detailed per-iteration results with questions, answers, and metadata
- **CSV summary**: Compact format with columns: `timestamp`, `model`, `N`, `batch_size`, `iteration`, `accuracy`, `L_irr`

## Extending to Other LLMs
1. Add a new loader file (e.g., `llm_newmodel.py`) with a `query_llm` function matching the interface
2. Add API key configuration in the "Set API keys" section of `llm_pauli_benchmark.py`
3. Update `config.json` to use the new backend/model

## Error Handling
- If the LLM fails to return answers in the expected format, all answers are marked incorrect
- Full LLM responses are saved on extraction failures for debugging
- Error messages are logged in the JSON records for later analysis

---

**For questions or contributions, please open an issue or pull request!** 