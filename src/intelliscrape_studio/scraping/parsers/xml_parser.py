from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin
import re

try:
    # Requires beautifulsoup4 and potentially lxml for XML parsing
    from bs4 import BeautifulSoup, Comment, ProcessingInstruction
except ImportError:
    raise ImportError("XMLParser requires BeautifulSoup4. Please install it: pip install beautifulsoup4")

# Use relative import for the base class
from .base import BaseParser

class XMLParser(BaseParser):
    """使用 BeautifulSoup 解析 XML 内容。"""
    
    def __init__(self, content: str, url: Optional[str] = None):
        super().__init__(content, url)
        self.soup: Optional[BeautifulSoup] = None # Parsed soup object
        # Common XML namespaces that might be useful (can be expanded)
        self.namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'rss': 'http://purl.org/rss/1.0/',
            'content': 'http://purl.org/rss/1.0/modules/content/',
            'dc': 'http://purl.org/dc/elements/1.1/'
        }

    def parse(self) -> BeautifulSoup:
        """解析 XML 字符串并返回 BeautifulSoup 对象。"""
        if not isinstance(self.content, str):
             raise TypeError("XMLParser requires string content.")
             
        # Use 'lxml-xml' or 'xml' parser. 'lxml-xml' is generally preferred.
        parser_choice = 'lxml-xml'
        try:
             import lxml
        except ImportError:
             parser_choice = 'xml' # Fallback built-in parser
             # print("lxml not found, using Python's built-in xml parser. Install lxml for better XML support.")
             
        try:
            self.soup = BeautifulSoup(self.content, parser_choice)
            return self.soup
        except Exception as e: # Catch potential parsing errors
            # print(f"XML parsing failed: {e}")
            raise ValueError(f"Invalid XML content: {e}") from e

    def _ensure_soup(self):
        """确保 BeautifulSoup 对象已创建。"""
        if self.soup is None:
            self.parse()
        if self.soup is None:
             raise RuntimeError("BeautifulSoup object could not be created for XML content.")

    def get_title(self) -> Optional[str]:
        """尝试获取 XML 文档的标题 (e.g., <title>, RSS/Atom title)。"""
        self._ensure_soup()
        
        # Try common tags first
        title_tag = self.soup.find('title', recursive=False) # Check top level first
        if title_tag and title_tag.string:
            return title_tag.string.strip()
        
        # Check within potential feed structures (RSS/Atom)
        if self.soup.find('channel'): # RSS
            title_tag = self.soup.channel.find('title')
            if title_tag and title_tag.string: return title_tag.string.strip()
        elif self.soup.find('feed', xmlns=self.namespaces.get('atom')): # Atom
             title_tag = self.soup.feed.find('title', xmlns=self.namespaces.get('atom'))
             if title_tag and title_tag.string: return title_tag.string.strip()
             
        # Fallback: Find any tag named 'title' anywhere
        title_tag = self.soup.find('title')
        if title_tag and title_tag.string:
             return title_tag.string.strip()
             
        return None

    def get_text(self) -> str:
        """获取 XML 文档的主要文本内容，去除标签。"""
        self._ensure_soup()
        
        # Create a copy to avoid modifying the original soup during text extraction
        text_soup = BeautifulSoup(str(self.soup), self.soup.builder.NAME) 

        # Remove comments and processing instructions
        for element in text_soup.find_all(string=lambda text: isinstance(text, (Comment, ProcessingInstruction))):
            element.extract()
            
        # Get text, try to preserve some structure with spaces
        text = text_soup.get_text(separator=' ', strip=True)
        
        # Clean up excessive whitespace
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def get_links(self) -> List[str]:
        """获取 XML 文档中的所有链接 (href, src, RSS/Atom links)。"""
        self._ensure_soup()
        links = set()
        
        # General href/src attributes
        for tag in self.soup.find_all(attrs={'href': True}):
             links.add(tag['href'])
        for tag in self.soup.find_all(attrs={'src': True}):
             links.add(tag['src'])
             
        # RSS/Atom specific links
        if self.soup.find('channel'): # RSS
             for item in self.soup.channel.find_all('item'):
                  link_tag = item.find('link')
                  if link_tag and link_tag.string: links.add(link_tag.string)
        elif self.soup.find('feed', xmlns=self.namespaces.get('atom')): # Atom
             for entry in self.soup.feed.find_all('entry', xmlns=self.namespaces.get('atom')):
                  for link_tag in entry.find_all('link', xmlns=self.namespaces.get('atom'), href=True):
                       links.add(link_tag['href'])

        # Resolve relative URLs and filter
        absolute_links = set()
        for link in links:
            href = link.strip()
            if href and not href.startswith( ('#', 'javascript:', 'mailto:', 'tel:') ):
                try:
                    absolute_url = urljoin(self.url or '', href)
                    if absolute_url.startswith(('http://', 'https://')):
                        absolute_links.add(absolute_url)
                except ValueError:
                    pass # Ignore invalid URLs
                    
        return sorted(list(absolute_links))

    def get_metadata(self) -> Dict[str, Any]:
        """尝试提取 XML 文档的元数据 (e.g., RSS/Atom feed info)。"""
        self._ensure_soup()
        metadata = {}
        
        # Try extracting feed-level metadata for RSS/Atom
        if self.soup.find('channel'): # RSS
             channel = self.soup.channel
             for tag in ['language', 'pubDate', 'lastBuildDate', 'managingEditor', 'webMaster', 'description']:
                  elem = channel.find(tag, recursive=False)
                  if elem and elem.string: metadata[f"rss_{tag}"] = elem.string.strip()
        elif self.soup.find('feed', xmlns=self.namespaces.get('atom')): # Atom
            feed = self.soup.feed
            for tag in ['id', 'updated', 'author', 'subtitle']:
                 elem = feed.find(tag, xmlns=self.namespaces.get('atom'), recursive=False)
                 if elem:
                     # Author might be complex, take name if possible
                     if tag == 'author' and elem.find('name'):
                          metadata[f"atom_{tag}"] = elem.find('name').string.strip()
                     elif elem.string:
                          metadata[f"atom_{tag}"] = elem.string.strip()
                          
        # Add root element info
        root = self.soup.find()
        if root:
             metadata['root_element'] = root.name
             if root.namespace:
                 metadata['root_namespace'] = root.namespace
                 
        return metadata
        
    # Optional: Method to find elements by tag name, possibly with namespace
    def find_elements(self, tag_name: str, namespace_uri: Optional[str] = None) -> List[BeautifulSoup]:
        """Find all elements by tag name, optionally specifying a namespace URI."""
        self._ensure_soup()
        # Note: BS4's namespace handling with find_all might require specific syntax or setup
        # depending on the XML structure and parser used (lxml is better here).
        # Simple search by name for now:
        return self.soup.find_all(tag_name) 