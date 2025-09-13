import json
import os
from typing import Dict, Any, Optional

class BenchmarkConfig:
    """Configuration manager for LLM benchmarking."""
    
    def __init__(self, config_path: str = None):
        # Resolve default path relative to this file so it works regardless of CWD
        default_path = os.path.join(os.path.dirname(__file__), 'config.json')
        self.config_path = config_path or default_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file with environment variable fallbacks."""
        config = {}
        
        # Try to load from JSON file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                print(f"✅ Loaded configuration from {self.config_path}")
            except Exception as e:
                print(f"⚠️  Error loading config file: {e}")
        else:
            print(f"⚠️  Config file not found: {self.config_path}")
            print("💡 Copy config.json.template to config.json and add your API keys")
        
        # Fallback to environment variables and defaults
        self._set_defaults(config)
        return config
    
    def _set_defaults(self, config: Dict[str, Any]):
        """Set default values and environment variable fallbacks."""
        
        # API keys with environment variable fallbacks
        if 'api_keys' not in config:
            config['api_keys'] = {}
        
        api_keys = config['api_keys']
        api_keys['openai'] = api_keys.get('openai', os.getenv('OPENAI_API_KEY'))
        api_keys['anthropic'] = api_keys.get('anthropic', os.getenv('ANTHROPIC_API_KEY'))
        api_keys['google'] = api_keys.get('google', os.getenv('GOOGLE_API_KEY'))
        
        # Benchmark settings
        if 'benchmark_settings' not in config:
            config['benchmark_settings'] = {}
        
        benchmark = config['benchmark_settings']
        benchmark['default_temperature'] = benchmark.get('default_temperature', 0.0)
        benchmark['default_max_tokens'] = benchmark.get('default_max_tokens', None)
        
        # Paths
        if 'paths' not in config:
            config['paths'] = {}
        
        paths = config['paths']
        paths['prompt_template'] = paths.get('prompt_template', 
                                           'dev/pauli_string_multiplication/letter_replace_26/utils/prompt_template.txt')
        paths['records_base_dir'] = paths.get('records_base_dir',
                                            'dev/pauli_string_multiplication/letter_replace_26/records')
    
    def get_api_key(self, backend: str) -> Optional[str]:
        """Get API key for the specified backend."""
        # Normalize backend name to avoid casing/whitespace issues
        backend = (backend or '').strip().lower()
        backend_map = {
            'openai': 'openai',
            'claude': 'anthropic',
            'gemini': 'google'
        }
        
        key_name = backend_map.get(backend, backend)
        api_key = self._config['api_keys'].get(key_name)
        
        if not api_key:
            raise ValueError(f"No API key found for backend '{backend}'. Please set it in config.json or environment variable.")
        
        return api_key
    
    def get_benchmark_setting(self, setting: str, default=None):
        """Get benchmark setting with default fallback."""
        return self._config['benchmark_settings'].get(setting, default)
    
    def get_path(self, path_name: str) -> str:
        """Get configured path."""
        return self._config['paths'][path_name]
    
    def validate_backend_config(self, backend: str) -> bool:
        """Validate that the backend has proper configuration."""
        try:
            api_key = self.get_api_key(backend)
            return api_key is not None
        except ValueError:
            return False
    
    def print_config_status(self):
        """Print configuration status for debugging."""
        print("\n🔧 Configuration Status:")
        print("-" * 30)
        
        # API keys status
        for backend in ['openai', 'claude', 'gemini']:
            status = "✅" if self.validate_backend_config(backend) else "❌"
            print(f"{status} {backend}: {'configured' if status == '✅' else 'missing API key'}")
        
        # Paths status
        print(f"\n📁 Paths:")
        for path_name in ['prompt_template', 'records_base_dir']:
            path = self.get_path(path_name)
            exists = os.path.exists(path) if path_name != 'records_base_dir' else True
            status = "✅" if exists else "⚠️ "
            print(f"{status} {path_name}: {path}")

# Global config instance
_config_instance = None

def get_config(config_path: str = None) -> BenchmarkConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = BenchmarkConfig(config_path)
    return _config_instance 