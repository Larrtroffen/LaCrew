from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class Parser(ABC):
    """解析器接口"""
    
    @abstractmethod
    async def parse(self, url: str) -> Optional[str]:
        """解析页面内容"""
        pass
    
    @abstractmethod
    def get_text(self) -> str:
        """获取文本内容"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """获取元数据"""
        pass 