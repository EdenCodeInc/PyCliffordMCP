# LLM Pauli String Multiplication Benchmark

This project provides a modular framework to benchmark large language models (LLMs) on Pauli string multiplication tasks, evaluating their ability to follow quantum algebra rules. It is designed for easy extensibility, reproducibility, and analysis across different LLMs (OpenAI, Gemini, Claude, etc.).

## Features
- **Configurable LLM backend and model** via `config.json`
- **External prompt template** for easy prompt engineering
- **Randomized question generation** for robust testing
- **Dynamic LLM loader**: add new LLMs by dropping in a loader file
- **Robust answer extraction and error handling**
- **Per-model experiment records and accuracy summaries**
- **Ready-to-plot CSV output for analysis**

## Directory Structure
```
dev/dev_test_pauli_string_multiplication/
├── llm_pauli_benchmark.py         # Main experiment script
├── llm_openai.py                  # OpenAI LLM loader (add more for other LLMs)
├── config.json                    # Config file for backend/model selection
├── prompt_template.txt            # Prompt template for LLM
├── records/                       # Experiment results (per model)
│   └── <model_name>/
│       ├── record_...json         # Per-run detailed records
│       └── accuracy_summary.csv   # Summary for plotting
└── ...
```

## Setup
1. **Install dependencies** (in your Python/conda environment):
   ```sh
   pip install openai numpy pandas matplotlib
   # Or use conda for numpy/pandas/matplotlib if preferred
   ```
2. **Set your OpenAI API key** (for OpenAI models):
   - In your notebook:
     ```python
     import os
     os.environ['OPENAI_API_KEY'] = 'sk-...your-key-here...'
     ```
   - Or in your terminal:
     ```sh
     export OPENAI_API_KEY='sk-...your-key-here...'
     ```
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
You can run experiments from a Jupyter notebook or Python script. Example for a notebook:

```python
import sys, os
os.chdir('/Users/hwanda/Desktop/EdenCode/projects/PyCliffordMCP-main')
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import llm_pauli_benchmark

# Set parameters
llm_pauli_benchmark.N = 3
llm_pauli_benchmark.batch_size = 5
llm_pauli_benchmark.num_iterations = 2
llm_pauli_benchmark.main()
```

To sweep over parameter ranges:
```python
from itertools import product
N_values = [1, 5, 10]
batch_sizes = [10]
num_iterations_list = [3]
for N, batch_size, num_iterations in product(N_values, batch_sizes, num_iterations_list):
    llm_pauli_benchmark.N = N
    llm_pauli_benchmark.batch_size = batch_size
    llm_pauli_benchmark.num_iterations = num_iterations
    llm_pauli_benchmark.main()
```

## Plotting Results
After running experiments, plot the results with:
```python
import os
import pandas as pd
import matplotlib.pyplot as plt

records_base = "dev/dev_test_pauli_string_multiplication/records"
batch_size = 10
num_iterations = 3
N_values = [1, 5, 10]

plt.figure(figsize=(8, 5))
for model_name in os.listdir(records_base):
    model_dir = os.path.join(records_base, model_name)
    csv_path = os.path.join(model_dir, "accuracy_summary.csv")
    if not os.path.isfile(csv_path):
        continue
    df = pd.read_csv(csv_path)
    df = df[(df['batch_size'] == batch_size) & (df['iteration'] == num_iterations)]
    means = []
    stds = []
    for N in N_values:
        accs = df[df['N'] == N]['accuracy']
        means.append(accs.mean() if not accs.empty else float('nan'))
        stds.append(accs.std() if not accs.empty else 0)
    plt.errorbar(N_values, means, yerr=stds, marker='o', capsize=4, label=model_name)
plt.xlabel("N (Pauli string length)")
plt.ylabel("Mean Accuracy")
plt.title(f"LLM Pauli Benchmark (batch_size={batch_size}, num_iterations={num_iterations})")
plt.legend(title="LLM Model")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

## Extending to Other LLMs
- Add a new loader file (e.g., `llm_gemini.py`) with a `query_llm` function matching the interface.
- Update `config.json` to use the new backend/model.

## Notes
- All experiment records are saved for reproducibility and later analysis.
- If the LLM fails to return answers in the expected format, all answers are marked incorrect and the error is logged.
- You can adjust the prompt, parameters, and LLM backend without changing the core experiment code.

---

**For questions or contributions, please open an issue or pull request!** 