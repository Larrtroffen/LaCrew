from typing import Optional, Dict, Any, List
from urllib.parse import urljoin
import re

try:
    from bs4 import BeautifulSoup, NavigableString, Comment
except ImportError:
    raise ImportError("HTMLParser requires BeautifulSoup4. Please install it: pip install beautifulsoup4")

try:
    # Optional: Use readability-lxml for better main content extraction
    from readability import Document
except ImportError:
    Document = None

# Use relative import for the base class
from .base import BaseParser


class HTMLParser(BaseParser):
    """使用 BeautifulSoup 解析 HTML 内容。"""
    
    def __init__(self, content: str, url: Optional[str] = None):
        super().__init__(content, url)
        self.soup: Optional[BeautifulSoup] = None # Parsed soup object

    def parse(self) -> BeautifulSoup:
        """解析 HTML 字符串并返回 BeautifulSoup 对象。"""
        # Ensure content is string
        if not isinstance(self.content, str):
             raise TypeError("HTMLParser requires string content.")
             
        # Use lxml if available for speed, fallback to html.parser
        parser_choice = 'lxml'
        try:
             import lxml
        except ImportError:
             parser_choice = 'html.parser'
             # print("lxml not found, using html.parser. Install lxml for faster parsing.")
             
        self.soup = BeautifulSoup(self.content, parser_choice)
        return self.soup

    def _ensure_soup(self):
        """确保 BeautifulSoup 对象已创建。"""
        if self.soup is None:
            self.parse()
        if self.soup is None: # Check again after parsing attempt
            raise RuntimeError("BeautifulSoup object could not be created.")

    def get_title(self) -> Optional[str]:
        """获取页面标题 (<title> 标签)。"""
        self._ensure_soup()
        if self.soup.title and self.soup.title.string:
            return self.soup.title.string.strip()
        # Fallback: Try to get from h1 tag
        h1 = self.soup.find('h1')
        if h1:
             return h1.get_text(strip=True)
        return None

    def get_text(self, main_content_only: bool = False) -> str:
        """获取页面的纯文本内容。
        
        Args:
            main_content_only: 如果为 True，尝试只提取主要内容区域的文本。
                               需要安装 readability-lxml 库才能获得最佳效果。
        Returns:
            提取的纯文本。
        """
        self._ensure_soup()

        if main_content_only and Document:
            try:
                # Use readability-lxml if available
                doc = Document(self.content)
                html_summary = doc.summary()
                # Parse the summary HTML back into soup to extract text
                summary_soup = BeautifulSoup(html_summary, 'lxml' if 'lxml' in globals() else 'html.parser')
                text = self._extract_text_from_soup(summary_soup)
                if text: # Return readability result only if it's not empty
                     return text
                # Fallback to full text if readability fails or returns empty
                # print("Readability extraction failed or returned empty, falling back to full text.")
            except Exception as e:
                 # print(f"Error during readability extraction: {e}. Falling back to full text.")
                 pass # Fall through to full text extraction
                 
        # Full text extraction (or fallback)
        return self._extract_text_from_soup(self.soup)
        
    def _extract_text_from_soup(self, soup: BeautifulSoup) -> str:
        """Helper function to extract and clean text from a soup object."""
        # Remove script, style, head, and other non-visible elements
        for element in soup(["script", "style", "head", "title", "meta", "[document]", "noscript", "link"]):
            element.decompose()
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
             comment.extract()
             
        # Get text, separating blocks with spaces
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up excessive whitespace
        text = re.sub(r'\s{2,}', ' ', text) 
        return text.strip()

    def get_links(self) -> List[str]:
        """获取页面中所有有效的绝对 URL 链接。"""
        self._ensure_soup()
        links = set() # Use set to avoid duplicates
        for a in self.soup.find_all('a', href=True):
            href = a['href'].strip()
            if href and not href.startswith( ('#', 'javascript:', 'mailto:', 'tel:') ):
                try:
                     # Resolve relative URLs using the base URL if available
                     absolute_url = urljoin(self.url or '', href)
                     # Basic check for valid http/https scheme
                     if absolute_url.startswith(('http://', 'https://')):
                         links.add(absolute_url)
                except ValueError: 
                     # Handle potential errors in urljoin (e.g., bad base URL)
                     pass # Ignore invalid URLs
                     
        return sorted(list(links))

    def get_metadata(self) -> Dict[str, Any]:
        """提取常见的元数据标签 (description, keywords, language, charset)。"""
        self._ensure_soup()
        metadata = {}
        
        # Description
        desc_tag = self.soup.find('meta', attrs={'name': re.compile(r'^description$', re.I)})
        if desc_tag and desc_tag.get('content'):
            metadata['description'] = desc_tag['content'].strip()
        
        # Keywords
        keys_tag = self.soup.find('meta', attrs={'name': re.compile(r'^keywords$', re.I)})
        if keys_tag and keys_tag.get('content'):
            keywords_str = keys_tag['content'].strip()
            metadata['keywords'] = [k.strip() for k in keywords_str.split(',') if k.strip()]
            
        # Language
        html_tag = self.soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag['lang'].strip()
            
        # Charset
        charset_tag = self.soup.find('meta', attrs={'charset': True})
        if charset_tag and charset_tag.get('charset'):
            metadata['charset'] = charset_tag['charset'].strip()
        else:
             # Fallback check for http-equiv content-type
             content_type_tag = self.soup.find('meta', attrs={'http-equiv': re.compile(r'^Content-Type$', re.I)})
             if content_type_tag and content_type_tag.get('content'):
                  match = re.search(r'charset=([\w-]+)', content_type_tag['content'], re.I)
                  if match:
                       metadata['charset'] = match.group(1).strip()
                       
        return metadata

    # Optional: Add method to extract specific elements by CSS selector
    def select(self, selector: str) -> List[BeautifulSoup]:
         """Find elements matching a CSS selector."""
         self._ensure_soup()
         return self.soup.select(selector)
         
    def select_one(self, selector: str) -> Optional[BeautifulSoup]:
         """Find the first element matching a CSS selector."""
         self._ensure_soup()
         return self.soup.select_one(selector) 