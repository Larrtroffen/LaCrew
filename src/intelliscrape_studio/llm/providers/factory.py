from typing import Dict, Type, Optional, List

# Use relative imports within the providers package
from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider

# You might add other providers here as they are implemented
# Example: from .ollama_provider import OllamaProvider 

class LLMProviderFactory:
    """LLM 提供商工厂类，用于创建和管理不同的 LLM 提供商实例。"""
    
    # Registry of available providers
    _providers: Dict[str, Type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        # Add other registered providers here
        # "ollama": OllamaProvider,
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[LLMProvider]):
        """动态注册一个新的 LLM 提供商。
        
        Args:
            name: 提供商的名称 (小写)。
            provider_class: 实现 LLMProvider 接口的类。
        """
        if not issubclass(provider_class, LLMProvider):
             raise TypeError(f"Provider class {provider_class.__name__} must inherit from LLMProvider.")
        cls._providers[name.lower()] = provider_class
        print(f"Successfully registered LLM provider: {name.lower()}")
    
    @classmethod
    def create_provider(cls, 
                       provider_name: str,
                       api_key: str,
                       base_url: Optional[str] = None,
                       **kwargs) -> LLMProvider:
        """根据名称创建 LLM 提供商实例。
        
        Args:
            provider_name: 要创建的提供商名称 (不区分大小写)。
            api_key: 提供商的 API 密钥。
            base_url: 自定义 API 基础 URL (可选)。
            **kwargs: 传递给提供商构造函数的其他参数。
            
        Returns:
            LLMProvider 接口的实例。
            
        Raises:
            ValueError: 如果提供商名称无效或未注册。
            ImportError: 如果提供商所需的依赖库未安装。
            Exception: 如果在初始化提供商时发生其他错误。
        """
        provider_name_lower = provider_name.lower()
        if provider_name_lower not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: '{provider_name}'. Available: {cls.get_available_providers()}")
            
        provider_class = cls._providers[provider_name_lower]
        
        try:
            # Pass api_key, base_url, and any other relevant kwargs
            return provider_class(api_key=api_key, base_url=base_url, **kwargs)
        except ImportError as e:
             # Catch import errors specific to the provider\'s dependencies
             raise ImportError(f"Failed to create provider '{provider_name}'. Missing dependency for {provider_class.__name__}: {e}") from e
        except Exception as e:
             # Catch other initialization errors
             # Consider logging the error here
             raise RuntimeError(f"Failed to initialize LLM provider '{provider_name}': {e}") from e

    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """获取所有已注册的可用提供商名称列表。"""
        return sorted(list(cls._providers.keys())) 