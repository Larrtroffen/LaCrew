from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class SearchEngine(ABC):
    """搜索引擎接口"""
    
    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """执行搜索"""
        pass
    
    @abstractmethod
    def set_proxy(self, proxy: Optional[str]):
        """设置代理"""
        pass
    
    @abstractmethod
    def set_timeout(self, timeout: int):
        """设置超时时间"""
        pass 