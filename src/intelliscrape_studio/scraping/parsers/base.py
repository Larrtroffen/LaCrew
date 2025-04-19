from typing import Any, Dict, Optional, List # Add List import
from abc import ABC, abstractmethod

class BaseParser(ABC):
    """基础解析器类，定义所有解析器需要实现的接口"""
    
    def __init__(self, content: str, url: Optional[str] = None):
        if not isinstance(content, str):
             raise TypeError("Parser content must be a string.")
        self.content = content
        self.url = url
        self._parsed_content = None # Cache for parsed structured data
        self._text_content = None   # Cache for extracted text
        self._links = None          # Cache for extracted links
        self._metadata = None       # Cache for extracted metadata
        self._title = None          # Cache for extracted title
    
    @abstractmethod
    def parse(self) -> Any:
        """核心解析逻辑，将原始content转换为内部表示（例如 BeautifulSoup对象、JSON dict等）。子类实现。"""
        pass
        
    def _ensure_parsed(self):
        """确保核心解析逻辑已执行。"""
        if self._parsed_content is None:
             self._parsed_content = self.parse()

    @abstractmethod
    def get_title(self) -> Optional[str]:
        """从解析后的内容中获取页面标题。子类实现。"""
        pass
        
    def get_cached_title(self) -> Optional[str]:
        """获取缓存的标题，如果未缓存则提取并缓存。"""
        if self._title is None:
             self._ensure_parsed()
             self._title = self.get_title()
        return self._title

    @abstractmethod
    def get_text(self) -> str:
        """从解析后的内容中提取纯文本。子类实现。"""
        pass
        
    def get_cached_text(self) -> str:
        """获取缓存的文本，如果未缓存则提取并缓存。"""
        if self._text_content is None:
             self._ensure_parsed()
             self._text_content = self.get_text()
        return self._text_content

    @abstractmethod
    def get_links(self) -> List[str]:
        """从解析后的内容中提取所有绝对URL链接。子类实现。"""
        pass
        
    def get_cached_links(self) -> List[str]:
         """获取缓存的链接列表，如果未缓存则提取并缓存。"""
         if self._links is None:
             self._ensure_parsed()
             self._links = self.get_links()
         return self._links

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """从解析后的内容中提取元数据（如 meta tags）。子类实现。"""
        pass
        
    def get_cached_metadata(self) -> Dict[str, Any]:
        """获取缓存的元数据，如果未缓存则提取并缓存。"""
        if self._metadata is None:
             self._ensure_parsed()
             self._metadata = self.get_metadata()
        return self._metadata

    # is_valid might be removed or re-evaluated based on parse() behavior
    # def is_valid(self) -> bool:
    #     """检查内容是否可以被解析"""
    #     return bool(self.content)
    
    # get_parsed_content might be less useful if specific getters are preferred
    # def get_parsed_content(self) -> Any:
    #     """获取解析后的内部表示"""
    #     self._ensure_parsed()
    #     return self._parsed_content
