from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

class LLMProvider(ABC):
    """LLM 提供商接口的抽象基类。"""
    
    @abstractmethod
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """初始化 LLM 提供商。
        
        Args:
            api_key: 提供商的 API 密钥。
            base_url: 自定义的 API 基础 URL (可选, 用于代理或自托管模型)。
            **kwargs: 传递给特定提供商 SDK 的其他参数。
        """
        self.api_key = api_key
        self.base_url = base_url
        # Store extra kwargs for potential use by subclasses
        self.init_kwargs = kwargs 
        # Consider adding a generic logger callback here if needed by providers
        # self.log_callback = kwargs.get('log_callback') 

    @abstractmethod
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = 1000,
                       response_format: Optional[Dict[str, str]] = None, # e.g., {'type': 'json_object'}
                       **kwargs # Allow extra provider-specific args
                       ) -> Union[str, Dict]: # Response can be string or structured (e.g., JSON)
        """执行聊天补全请求。
        
        Args:
            messages: 对话消息列表 (例如, [{'role': 'user', 'content': '...'}]).
            model: 要使用的模型名称。
            temperature: 控制随机性的温度参数 (0.0 到 2.0)。
            max_tokens: 生成响应的最大 token 数 (可选)。
            response_format: 指定响应格式 (例如 JSON) (可选, 取决于提供商支持)。
            **kwargs: 其他传递给 API 调用的参数。
            
        Returns:
            生成的响应，通常是字符串，但如果请求了特定格式 (如 JSON)，则可能是字典。
            
        Raises:
            NotImplementedError: 如果子类未实现。
            Exception: 如果 API 调用失败。
        """
        pass

    # Optional methods that could be useful
    # @abstractmethod
    # def count_tokens(self, text: str, model: str) -> int:
    #     """计算给定文本在特定模型下的 token 数。"""
    #     pass

    # @abstractmethod
    # def get_available_models(self) -> List[str]:
    #     """获取此提供商可用的模型列表。"""
    #     pass

    # Credentials validation might be better done implicitly during the first API call
    # @abstractmethod
    # def validate_credentials(self) -> bool:
    #     """验证 API 凭证是否有效。"""
    #     pass 