from typing import Optional, Type, Union
import json
import re

# Use relative imports within the parsers package
from .base import BaseParser
from .html_parser import HTMLParser
from .json_parser import JSONParser
from .xml_parser import XMLParser
# Import BeautifulSoup here if needed for detection, or handle missing import
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None # Define as None if not installed


class ParserFactory:
    """解析器工厂类，用于根据内容或MIME类型选择合适的解析器。"""
    
    # Mapping from simplified type name to Parser class
    _parser_map = {
        'html': HTMLParser,
        'json': JSONParser,
        'xml': XMLParser,
        # Add more parsers here, e.g., 'pdf': PDFParser
    }

    @staticmethod
    def create_parser(content: Union[str, bytes], url: Optional[str] = None, mime_type: Optional[str] = None) -> BaseParser:
        """根据内容、URL或MIME类型创建合适的解析器。
        
        Args:
            content: 要解析的内容 (字符串或字节)。
            url: 内容的URL（可选, 可用于推断类型或处理相对链接）。
            mime_type: 内容的MIME类型（可选, 优先于内容检测）。
            
        Returns:
            合适的解析器实例。
            
        Raises:
            ValueError: 如果无法确定解析器类型。
            TypeError: 如果内容不是字符串或字节。
        """
        if not isinstance(content, (str, bytes)):
             raise TypeError("Content must be a string or bytes.")
             
        # Convert bytes to string if necessary (assuming UTF-8, might need adjustment)
        if isinstance(content, bytes):
             try:
                 content_str = content.decode('utf-8')
             except UnicodeDecodeError:
                 # Fallback to latin-1 or handle error appropriately
                 try:
                    content_str = content.decode('latin-1')
                 except Exception as e:
                    raise ValueError(f"Could not decode byte content: {e}") from e
        else:
             content_str = content
             
        parser_type = None
        # 1. Use MIME type if provided
        if mime_type:
            mime_lower = mime_type.lower()
            if 'html' in mime_lower:
                parser_type = 'html'
            elif 'json' in mime_lower:
                parser_type = 'json'
            elif 'xml' in mime_lower or 'rss' in mime_lower or 'atom' in mime_lower:
                parser_type = 'xml'
            # Add more MIME type checks

        # 2. If no type from MIME, detect from content
        if parser_type is None:
            parser_type = ParserFactory._detect_content_type(content_str)
            
        # 3. Fallback or raise error
        if parser_type == 'unknown':
            # Default to HTML parser as a common fallback?
            # Or raise an error if type cannot be determined.
            # print(f"Warning: Unknown content type for URL {url}. Defaulting to HTML.")
            # parser_type = 'html' 
            raise ValueError(f"Could not determine parser type for content (URL: {url}, MIME: {mime_type}).")
            
        parser_class = ParserFactory._parser_map.get(parser_type)
        
        if parser_class:
            return parser_class(content_str, url)
        else:
             # This case should ideally not be reached if detection logic is sound
             raise ValueError(f"No parser registered for detected type: {parser_type}")

    
    @staticmethod
    def _detect_content_type(content: str) -> str:
        """根据字符串内容尝试检测类型 (HTML, JSON, XML)。"""
        stripped_content = content.strip()
        
        if not stripped_content:
            return 'unknown'
        
        # JSON check (more reliable)
        if stripped_content.startswith(('{', '[')):
            try:
                json.loads(stripped_content)
                return 'json'
            except json.JSONDecodeError:
                pass # Not valid JSON
        
        # XML check
        # Look for common XML declaration or root tag start
        if stripped_content.startswith('<?xml') or \
           (stripped_content.startswith('<') and not stripped_content.startswith('<!DOCTYPE') and ':' in stripped_content[:100]): # Heuristic for namespace
            # Optional: More robust check using XML parser if BS4 is available
            if BeautifulSoup:
                try:
                    # Try parsing with lxml or xml.etree for better validation
                    BeautifulSoup(stripped_content, 'xml') # Or 'lxml-xml' if installed
                    return 'xml'
                except Exception:
                     pass # Parsing failed
            else:
                 # Basic check passed, assume XML if BS4 not available
                 return 'xml'
                 
        # HTML check
        # Look for DOCTYPE or <html> tag, case-insensitive
        content_lower_snippet = stripped_content[:500].lower() # Check beginning
        if content_lower_snippet.startswith('<!doctype html') or '<html' in content_lower_snippet:
            return 'html'
            
        # Less reliable: check for any tag-like structures as HTML fallback
        if re.search(r'<([a-z][a-z0-9]*)\b[^>]*>', stripped_content, re.IGNORECASE):
            return 'html'
        
        return 'unknown' # Could not determine type 