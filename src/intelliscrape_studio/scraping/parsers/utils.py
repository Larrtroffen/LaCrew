from typing import List, Dict, Any, Optional, Set
import re
from urllib.parse import urlparse, urljoin

try:
    from bs4 import BeautifulSoup
except ImportError:
    # Allow utils to be imported even if bs4 isn't installed, 
    # but methods requiring it will fail.
    BeautifulSoup = None

try:
    from readability import Document
except ImportError:
    Document = None


class ParserUtils:
    """提供用于解析和处理内容的静态工具方法。"""
    
    @staticmethod
    def clean_text(text: str, aggressive: bool = False) -> str:
        """清理文本，移除多余空白，以及可选地移除特殊字符。

        Args:
            text: 要清理的文本。
            aggressive: 如果为 True，移除所有非字母数字、非空白、非中文字符。
                        如果为 False，仅规范化空白。

        Returns:
            清理后的文本。
        """
        if not isinstance(text, str):
            return "" # Return empty for non-string input
            
        # Normalize whitespace (replace multiple spaces/newlines with single space)
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        
        if aggressive:
            # Keep letters, numbers, spaces, and common CJK characters
            # Adjust the regex range based on specific character needs
            cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?-]', '', cleaned_text) # Keep some basic punctuation
            # Remove extra spaces potentially introduced by character removal
            cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
            
        return cleaned_text
    
    @staticmethod
    def extract_links(html_content: str, base_url: Optional[str] = None) -> List[str]:
        """从 HTML 字符串中提取所有绝对 URL。
        
        Args:
            html_content: HTML 内容字符串。
            base_url: 用于解析相对链接的基础 URL。
            
        Returns:
            提取到的绝对 URL 列表 (去重并排序)。
        """
        if not BeautifulSoup:
             # print("Warning: BeautifulSoup not installed. Cannot extract links from HTML.")
             return []
             
        if not isinstance(html_content, str):
             return []
             
        # Use lxml if available
        parser = 'lxml' if 'lxml' in globals() else 'html.parser'
        soup = BeautifulSoup(html_content, parser)
        unique_links: Set[str] = set()
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            if href and not href.startswith( ('#', 'javascript:', 'mailto:', 'tel:') ):
                try:
                    absolute_url = urljoin(base_url or '', href)
                    if ParserUtils.is_valid_url(absolute_url):
                        unique_links.add(absolute_url)
                except ValueError:
                    pass # Ignore errors during urljoin
        
        return sorted(list(unique_links))
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """检查 URL 是否具有有效的方案 (http/https) 和网络位置。"""
        if not isinstance(url, str) or len(url) > 2048: # Basic checks
             return False
        try:
            result = urlparse(url)
            # Check for http or https scheme and a netloc (domain name)
            return result.scheme in ['http', 'https'] and bool(result.netloc)
        except ValueError: # Handle potential parsing errors for malformed URLs
            return False
    
    @staticmethod
    def normalize_url(url: str, keep_fragment: bool = False) -> str:
        """规范化 URL (小写方案和域名，移除片段等)。"""
        try:
            parsed = urlparse(url.strip())
            # Reconstruct with lowercase scheme and netloc
            # Handle potential missing parts gracefully
            scheme = parsed.scheme.lower() if parsed.scheme else ''
            netloc = parsed.netloc.lower() if parsed.netloc else ''
            path = parsed.path if parsed.path else '/' # Ensure path starts with /
            params = parsed.params
            query = parsed.query
            fragment = parsed.fragment if keep_fragment else ''
            
            # Ensure scheme is present if netloc exists
            if netloc and not scheme:
                 scheme = 'http' # Default to http if scheme missing but domain present?
            
            # Handle IDNA encoding for international domains
            try:
                 netloc = netloc.encode('idna').decode('ascii')
            except Exception:
                 pass # Ignore encoding errors
                 
            # Rebuild the URL
            normalized = f"{scheme}://{netloc}{path}"
            if params: normalized += f";{params}"
            if query: normalized += f"?{query}"
            if fragment: normalized += f"#{fragment}"
            
            return normalized
        except Exception:
            # Return original URL if normalization fails
            return url.strip()

    # extract_metadata and extract_main_content might be better placed
    # within the HTMLParser class itself, as they rely heavily on BS4 parsing.
    # Keeping them here mirrors the original structure but consider refactoring.

    @staticmethod
    def extract_metadata(html_content: str) -> Dict[str, Any]:
        """(依赖 BS4) 从 HTML 字符串中提取元数据。"""
        if not BeautifulSoup:
            # print("Warning: BeautifulSoup not installed. Cannot extract metadata.")
            return {}
        if not isinstance(html_content, str):
            return {}
            
        parser = 'lxml' if 'lxml' in globals() else 'html.parser'
        soup = BeautifulSoup(html_content, parser)
        metadata = {}
        
        # Title
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            metadata['title'] = ParserUtils.clean_text(title_tag.string)
        
        # Meta tags (name or property)
        for meta in soup.find_all('meta'):
            key = meta.get('name') or meta.get('property')
            content = meta.get('content')
            if key and content:
                metadata[key.lower()] = content.strip()
                
        # Language
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
             metadata['language'] = html_tag['lang'].strip()
             
        # Favicon
        icon_link = soup.find('link', rel=re.compile(r'icon', re.I), href=True)
        if icon_link:
             metadata['favicon'] = icon_link['href']
             
        # Canonical URL
        canonical_link = soup.find('link', rel='canonical', href=True)
        if canonical_link:
             metadata['canonical_url'] = canonical_link['href']
             
        return metadata
    
    @staticmethod
    def extract_main_content(html_content: str) -> Optional[str]:
        """(依赖 readability 或 BS4) 尝试提取主要文本内容。"""
        if Document:
            try:
                doc = Document(html_content)
                # Return text content of the summary HTML
                summary_html = doc.summary()
                if BeautifulSoup:
                     parser = 'lxml' if 'lxml' in globals() else 'html.parser'
                     summary_soup = BeautifulSoup(summary_html, parser)
                     # Clean the text extracted from summary
                     return ParserUtils.clean_text(summary_soup.get_text(separator=' '))
                else:
                     # Fallback if BS4 not available to parse summary
                     return ParserUtils.clean_text(summary_html) # Return raw summary HTML text
            except Exception:
                # Fallback if readability fails
                pass 
        
        # Fallback using BeautifulSoup basic cleaning if readability failed or not available
        if BeautifulSoup:
            if not isinstance(html_content, str):
                 return None
            parser = 'lxml' if 'lxml' in globals() else 'html.parser'
            soup = BeautifulSoup(html_content, parser)
            
            # Basic approach: remove script/style, get body text
            for element in soup(["script", "style", "head", "nav", "footer", "aside"]):
                element.decompose()
            body = soup.find('body')
            if body:
                return ParserUtils.clean_text(body.get_text(separator=' '))
            else: # If no body, try getting all text
                 return ParserUtils.clean_text(soup.get_text(separator=' '))
                 
        # Cannot extract content if neither library is available
        # print("Warning: Neither readability nor BeautifulSoup installed. Cannot extract main content.")
        return None 