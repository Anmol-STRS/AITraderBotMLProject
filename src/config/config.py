"""
Simple Configuration Loader
Everything in one file - clean and easy
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any

try:
    import tomli as toml
    USE_BINARY = True  # tomli needs binary mode
except ImportError:
    try:
        import tomllib as toml
        USE_BINARY = True  # tomllib also needs binary mode
    except ImportError:
        import toml
        USE_BINARY = False  # old toml package uses text mode

from dotenv import load_dotenv
from src.util.logging import get_logger

class Config:
    """Simple configuration loader - everything in one place."""
    
    def __init__(self, config_path: str = None):
        """Load configuration from single TOML file."""
        self.log = logging.getLogger("Config")
        load_dotenv()
        
        # Auto-find config.toml
        if config_path is None:
            possible_paths = [
                Path("config.toml"),
                Path("config.toml"),
                Path(__file__).parent / "config.toml",
                Path(__file__).parent.parent.parent / "config.toml",
            ]
            
            config_path = None
            for p in possible_paths:
                if p.exists():
                    config_path = p
                    self.log.info(f"Found config at: {config_path}")
                    break
        
        if config_path is None or not Path(config_path).exists():
            self.log.error(f"Config file not found!")
            self.log.error(f"Tried: {[str(p) for p in possible_paths]}")
            self.log.error(f"Current directory: {Path.cwd()}")
            self.config = {}
            return
        
        # Load TOML file
        try:
            config_path = Path(config_path)
            self.log.info(f"Loading config from: {config_path}")
            
            # Use correct mode based on library
            if USE_BINARY:
                with open(config_path, 'rb') as f:
                    self.config = toml.load(f)
            else:
                with open(config_path, 'r') as f:
                    self.config = toml.load(f)
            
            self.log.info(f"Loaded sections: {list(self.config.keys())}")
            
        except Exception as e:
            self.log.error(f"Error loading config: {e}")
            self.config = {}
            return
        
        # Merge API keys from .env
        self._merge_env_keys()
        
        self.log.info("Configuration loaded successfully")
    
    def _merge_env_keys(self):
        """Add API keys from environment variables."""
        env_keys = {
            'alpha_vantage': 'ALPHA_VANTAGE_KEY',
            'news_api': 'NEWS_API_KEY',
            'finnhub': 'FINNHUB_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'deepseek': 'DEEPSEEK_API_KEY',
        }
        
        # Add keys to api section
        for service, env_var in env_keys.items():
            key = os.getenv(env_var, '')
            if key and 'api' in self.config:
                if service in self.config['api']:
                    self.config['api'][service]['api_key'] = key
                    self.config['api'][service]['enabled'] = True
                    self.log.debug(f"Enabled {service} from .env")
            
            # Also add to llm section for openai, anthropic, deepseek
            if service in ['openai', 'anthropic', 'deepseek']:
                if key and 'llm' in self.config:
                    if service in self.config['llm']:
                        self.config['llm'][service]['api_key'] = key
                        self.config['llm'][service]['enabled'] = True
    
    # ========================================================================
    # MAIN SECTIONS
    # ========================================================================
    
    @property
    def api(self) -> Dict[str, Any]:
        """Get API configurations."""
        return self.config.get('api', {})
    
    @property
    def database(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.config.get('database', {})
    
    @property
    def llm(self) -> Dict[str, Any]:
        """Get LLM configurations."""
        return self.config.get('llm', {})
    
    @property
    def model_results_db(self) -> Dict[str, Any]:
        """
        Get model results database configuration.
        Falls back to project root/model_results.db if not defined.
        """
        default_path = Path(__file__).resolve().parents[2] / "model_results.db"
        return self.config.get('model_results_db', {'path': str(default_path)})
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.
        
        Examples:
            config.get('database.path')
            config.get('api.yfinance.rate_limit')
            config.get('llm.openai.model')
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def is_enabled(self, service: str) -> bool:
        """Check if a service is enabled."""
        if service in self.api:
            return self.api[service].get('enabled', False)
        
        if service in self.llm:
            return self.llm[service].get('enabled', False)
        
        return False
    
    def get_api_key(self, service: str) -> str:
        """Get API key for a service."""
        if service in self.api:
            return self.api[service].get('api_key', '')
        
        if service in self.llm:
            return self.llm[service].get('api_key', '')
        
        return ''


# Global config instance
_config = None

def get_config() -> Config:
    """Get global config instance (singleton)."""
    global _config
    if _config is None:
        _config = Config()
    return _config
