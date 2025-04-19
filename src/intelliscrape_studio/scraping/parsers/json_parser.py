import json
from typing import Any, Dict, List, Optional, Union

# Use relative import for the base class
from .base import BaseParser

class JSONParser(BaseParser):
    """解析 JSON 字符串。"""
    
    def __init__(self, content: str, url: Optional[str] = None):
        # JSON content should always be a string for json.loads
        super().__init__(content, url)
        self._data: Optional[Union[Dict, List]] = None # Parsed JSON object/array

    def parse(self) -> Union[Dict, List]:
        """解析 JSON 字符串并返回 Python 字典或列表。"""
        if not isinstance(self.content, str):
            raise TypeError("JSONParser requires string content.")
            
        if self._data is None:
            try:
                self._data = json.loads(self.content)
            except json.JSONDecodeError as e:
                # Optionally log the error
                # print(f"JSON parsing failed: {e}")
                # Return empty dict or list depending on what's expected? 
                # Returning empty dict for now.
                self._data = {}
                raise ValueError(f"Invalid JSON content: {e}") from e
        return self._data

    def _ensure_parsed(self):
        """确保 JSON 数据已解析。"""
        if self._data is None:
            self.parse()
        # Check again in case parse failed and raised/returned empty
        if self._data is None:
             raise RuntimeError("JSON data could not be parsed.")

    def get_title(self) -> Optional[str]:
        """尝试从常见的键（如 'title', 'name'）获取标题。"""
        self._ensure_parsed()
        if isinstance(self._data, dict):
            for key in ['title', 'name', 'header']: # Common keys for a title
                if key in self._data and isinstance(self._data[key], str):
                    return self._data[key].strip()
        return None # No clear title found

    def get_text(self) -> str:
        """将解析后的 JSON 漂亮地格式化为字符串表示形式。"""
        self._ensure_parsed()
        try:
            return json.dumps(self._data, ensure_ascii=False, indent=2)
        except TypeError as e:
            # Handle potential serialization errors (e.g., non-serializable objects)
            # print(f"Error serializing JSON data to text: {e}")
            return str(self._data) # Fallback to simple string representation

    def get_links(self) -> List[str]:
        """递归地从解析后的 JSON 数据中提取所有有效的 URL。"""
        self._ensure_parsed()
        links = set()
        self._extract_links_recursive(self._data, links)
        return sorted(list(links))
    
    def _extract_links_recursive(self, data: Any, links: set):
        """递归辅助函数，用于查找链接。"""
        if isinstance(data, dict):
            for key, value in data.items():
                # Check common keys and if value is a string URL
                if isinstance(value, str) and key in ['url', 'link', 'href', 'uri', '@id']: 
                    if value.startswith(('http://', 'https://')):
                        links.add(value)
                # Recurse into nested structures
                elif isinstance(value, (dict, list)):
                    self._extract_links_recursive(value, links)
                # Optional: Check if any string value *looks* like a URL, even if key doesn't match
                # elif isinstance(value, str) and value.startswith(('http://', 'https://')) and '.' in value:
                #     links.add(value)
        elif isinstance(data, list):
            for item in data:
                self._extract_links_recursive(item, links)

    def get_metadata(self) -> Dict[str, Any]:
        """提取 JSON 数据中的潜在元数据。
           这里假设顶层字典键可能是元数据。
        """
        self._ensure_parsed()
        metadata = {}
        if isinstance(self._data, dict):
            # Collect top-level keys that are not complex types (dict/list)
            for key, value in self._data.items():
                if not isinstance(value, (dict, list)):
                    metadata[key] = value
            # Remove potentially large fields like 'content' or 'text'
            for key in ['content', 'text', 'body', 'html']:
                 metadata.pop(key, None)
        return metadata
        
    # Specific method to access data using JSON path (optional enhancement)
    # def get_path(self, path: str) -> Optional[Any]:
    #     """Get data using a simple dot-notation path."""
    #     self._ensure_parsed()
    #     try:
    #         parts = path.split('.')
    #         value = self._data
    #         for part in parts:
    #             if isinstance(value, list):
    #                 try:
    #                      part_index = int(part)
    #                      value = value[part_index]
    #                 except (ValueError, IndexError):
    #                      return None
    #             elif isinstance(value, dict):
    #                  value = value.get(part)
    #                  if value is None:
    #                       return None
    #             else:
    #                  return None
    #         return value
    #     except Exception:
    #          return None 