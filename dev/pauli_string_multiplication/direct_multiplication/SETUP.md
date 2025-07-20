# LLM Benchmarking Setup Guide

## Quick Setup

### 1. Create Configuration File
```bash
cd dev/pauli_string_multiplication/direct_multiplication/utils/
cp config.json.template config.json
```

### 2. Add Your API Keys
Edit `utils/config.json` and replace the placeholder keys with your actual API keys:

```json
{
  "api_keys": {
    "openai": "sk-proj-your-actual-openai-key",
    "anthropic": "sk-ant-your-actual-claude-key", 
    "google": "your-actual-google-api-key"
  }
}
```

### 3. Run Benchmarks
```python
import llm_pauli_benchmark

# Set your model configuration
llm_pauli_benchmark.LLM_BACKEND = "gemini"
llm_pauli_benchmark.MODEL_NAME = "gemini-2.5-pro"

# Run the benchmark
llm_pauli_benchmark.main()
```

## Configuration Options

### API Keys
- **OpenAI**: Get from https://platform.openai.com/api-keys
- **Anthropic (Claude)**: Get from https://console.anthropic.com/
- **Google (Gemini)**: Get from https://aistudio.google.com/app/apikey

### Environment Variables (Alternative)
Instead of `utils/config.json`, you can set environment variables:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key" 
export GOOGLE_API_KEY="your-key"
```

### Benchmark Settings
```json
{
  "benchmark_settings": {
    "default_temperature": 0.0,      // Use 0.0 for deterministic research
    "default_max_tokens": null       // null = use model maximum
  }
}
```

### Paths (Optional)
```json
{
  "paths": {
    "prompt_template": "path/to/your/prompt.txt",
    "prompt_irrelevant": "path/to/irrelevant.txt", 
    "records_base_dir": "path/to/save/results"
  }
}
```

## Security Notes

- ✅ **utils/config.json** is gitignored - your API keys won't be committed
- ✅ Environment variables are used as fallbacks
- ✅ Hardcoded API keys have been removed from the code

## What You Get

### Token Usage Tracking
```
[Iteration 1] Token usage: 245 in, 67 out
```

### Enhanced Records
```json
{
  "accuracy": 0.8,
  "input_tokens": 245,
  "output_tokens": 67,
  "total_tokens": 312,
  ...
}
```

### CSV Summary with Tokens
```csv
timestamp,model,N,batch_size,L_irr,iteration,accuracy,input_tokens,output_tokens,total_tokens
```

## Troubleshooting

### "No API key found" Error
1. Check your `utils/config.json` has the correct key names
2. Verify API keys are valid and have sufficient credits
3. Try setting environment variables as backup

### "Config file not found" Warning
1. Copy `utils/config.json.template` to `utils/config.json`
2. Or rely on environment variables (warning is harmless)

### Token Usage Shows "null"
- Some API responses may not include token counts
- This is normal - just means the API didn't provide usage data 