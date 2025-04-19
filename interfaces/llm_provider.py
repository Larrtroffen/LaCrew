from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMProvider(ABC):
    """LLM提供商接口"""
    
    @abstractmethod
    async def chat_completion(self,
                            messages: List[Dict[str, str]],
                            model: str,
                            temperature: float = 0.7,
                            max_tokens: int = 1000) -> str:
        """聊天完成"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        pass
    
    @abstractmethod
    def set_api_key(self, api_key: str):
        """设置API密钥"""
        pass
    
    @abstractmethod
    def set_base_url(self, base_url: Optional[str]):
        """设置基础URL"""
        pass 