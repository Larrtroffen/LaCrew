import os
from typing import Any, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

class ConfigManager:
    """配置管理器，负责加载和管理所有配置"""
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self._load_env()
        self._load_default_config()
    
    def _load_env(self):
        """加载环境变量"""
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
    
    def _load_default_config(self):
        """加载默认配置"""
        self.config.update({
            'llm': {
                'provider': os.getenv('LLM_PROVIDER', 'openai'),
                'api_key': os.getenv('LLM_API_KEY', ''),
                'base_url': os.getenv('LLM_BASE_URL', ''),
                'model': os.getenv('LLM_MODEL', 'gpt-3.5-turbo'),
            },
            'web': {
                'mode': os.getenv('WEB_MODE', 'Browser Automation'),
                'search_engine': os.getenv('SEARCH_ENGINE', 'duckduckgo'),
                'proxy': os.getenv('PROXY_SERVER', ''),
                'timeout': int(os.getenv('REQUEST_TIMEOUT', '30')),
                'max_retries': int(os.getenv('MAX_RETRIES', '3')),
            },
            'crawler': {
                'max_pages': int(os.getenv('MAX_PAGES', '10')),
                'max_data_count': int(os.getenv('MAX_DATA_COUNT', '10')),
                'domain_restriction': os.getenv('DOMAIN_RESTRICTION', ''),
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'format': os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
                'file': os.getenv('LOG_FILE', 'app.log'),
            }
        })
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        current = self.config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    def update(self, config: Dict[str, Any]):
        """批量更新配置"""
        self.config.update(config)
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self.config.copy() 