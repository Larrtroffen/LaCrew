# -*- coding: utf-8 -*-
"""
搜索引擎集成模块。

包含各种搜索引擎的实现（例如 DuckDuckGo, Baidu）以及一个 Searcher 类来协调使用。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Type
import time
import requests
import functools
import os
import logging

# --- Dependency Imports ---
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None  # Mark as unavailable if not installed

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # Mark as unavailable if not installed

logger = logging.getLogger(__name__)  # Use standard logging

# --- Base Class ---
class BaseSearchEngine(ABC):
    """搜索引擎的抽象基类。"""
    
    def __init__(self, config: Dict[str, Any], api_keys: Dict[str, Any], proxy: Optional[str] = None):
        """Initializes the search engine.

        Args:
            config: Search engine specific configuration (e.g., settings for just this engine).
            api_keys: Dictionary containing relevant API keys.
            proxy: Optional proxy URL string.
        """
        self.config = config
        self.api_keys = api_keys
        self.proxy = proxy # Store passed proxy
        self.session = self._create_session() # Session creation will now use self.proxy

    def _create_session(self) -> requests.Session:
        """创建并配置 requests session。"""
        session = requests.Session()
        if self.proxy: # Use the stored proxy
            # Ensure proxy includes scheme (http:// or https:// or socks5://)
            if '://' not in self.proxy:
                logger.warning(f"Proxy string '{self.proxy}' might be missing scheme (e.g., http://). Assuming http.")
                proxies = {'http': f'http://{self.proxy}', 'https': f'http://{self.proxy}'}
            else:
                proxies = {'http': self.proxy, 'https': self.proxy}
            session.proxies = proxies
            logger.debug(f"Requests session configured with proxy: {proxies}")
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/108.0.0.0 Safari/537.36'
        })
        return session
    
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """执行搜索并返回结果。

        Args:
            query: 搜索查询字符串。
            max_results: 希望返回的最大结果数量。

        Returns:
            一个字典列表，每个字典包含 'title', 'url', 'snippet'。
            
        Raises:
            NotImplementedError: 如果子类未实现此方法。
            Exception: 如果搜索过程中发生错误。
        """
        pass

    def _make_request(self, url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 15) -> requests.Response:
        """使用配置好的 session 发送 GET 请求。"""
        try:
            response = self.session.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            logger.info(f"Request to {url} successful (Status: {response.status_code})")
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {url} failed: {e}")
            raise ConnectionError(f"Failed to connect to {url}: {e}") from e

# --- DuckDuckGo Implementation ---
class DuckDuckGoSearch(BaseSearchEngine):
    """使用 duckduckgo_search 库实现 DuckDuckGo 搜索。"""
    
    def __init__(self, config: Dict[str, Any], api_keys: Dict[str, Any], proxy: Optional[str] = None):
        super().__init__(config, api_keys, proxy) # Pass proxy to base class
        if DDGS is None:
            raise ImportError(
                "DuckDuckGo search requires the 'duckduckgo_search' library. "
                "Install it with: pip install duckduckgo_search"
            )
        self.timeout = config.get('timeout', 20)
        # Proxy is already stored in self.proxy by the base class

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """执行 DuckDuckGo 搜索。"""
        results = []
        last_exception = None
        max_retries = self.config.get('max_retries', 2)

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting DuckDuckGo search (Attempt {attempt + 1}/{max_retries}): {query}")
                
                # Use self.proxy stored by base class
                with DDGS(timeout=self.timeout, proxies=self.proxy) as ddgs:
                    # Use text search
                    raw_results = ddgs.text(
                        query,
                        max_results=max_results
                    )
                    
                    # Convert generator to list and format
                    results = []
                    for r in raw_results:
                         if isinstance(r, dict) and r.get('title') and r.get('href'):
                              results.append({
                                  'title': r['title'],
                                  'url': r['href'],
                                  'snippet': r.get('body', '')
                              })
                              if len(results) >= max_results:
                                   break  # Stop if we reached the desired number
                                   
                logger.info(f"DuckDuckGo search successful. Found {len(results)} results.")
                return results  # Return on success

            except Exception as e:
                last_exception = e
                logger.warning(f"DuckDuckGo search attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retrying
        
        # If all retries failed
        logger.error(f"DuckDuckGo search failed after {max_retries} attempts.")
        # Raise the last exception encountered or a generic error
        raise ConnectionError(f"DuckDuckGo search failed: {last_exception}") from last_exception

# --- Baidu Implementation (Kept for reference, but potentially fragile) ---
class BaiduSearch(BaseSearchEngine):
    """通过网页抓取实现百度搜索（可能不稳定）。"""

    def __init__(self, config: Dict[str, Any], api_keys: Dict[str, Any], proxy: Optional[str] = None):
        super().__init__(config, api_keys, proxy) # Pass proxy to base class
        if BeautifulSoup is None:
            raise ImportError(
                "Baidu search requires BeautifulSoup4. Install it with: pip install beautifulsoup4 lxml"
            )
        self.search_url_template = "https://www.baidu.com/s?wd={query}&rn={num}"  # rn controls number of results per page
        self.headers = {
             # Use headers similar to a real browser
             'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
             'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
             'Cache-Control': 'no-cache',
             'Pragma': 'no-cache'
         }
        # self.session is created by base class __init__ using the proxy

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """执行百度搜索并解析结果页面。"""
        results = []
        last_exception = None
        max_retries = self.config.get('max_retries', 2)
        # Baidu's rn parameter might not guarantee exact number
        results_to_request = max(max_results, 10) 
        
        search_url = self.search_url_template.format(query=requests.utils.quote(query), num=results_to_request)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting Baidu search (Attempt {attempt + 1}/{max_retries}): {query}")
                response = self._make_request(search_url, headers=self.headers)
                
                # Basic checks for Baidu blocks/verification
                if "验证" in response.text[:1000] or "安全验证" in response.text[:1000]:
                     raise ConnectionError("Baidu security verification detected. Check IP or use proxy.")
                if "百度一下" in response.text[:500] and "<title>百度搜索" not in response.text[:500]:
                     raise ConnectionError("Redirected to Baidu homepage, possibly blocked.")

                # Use BeautifulSoup to parse
                soup = BeautifulSoup(response.text, 'lxml' if 'lxml' in globals() else 'html.parser')
                results = self._extract_results_from_soup(soup)
                
                if not results:
                    # Maybe save debug page?
                    # logger.debug("Baidu page content:", response.text[:500])
                    raise ValueError("Could not extract any results from Baidu page. Structure might have changed.")

                logger.info(f"Baidu search successful. Found {len(results)} potential results.")
                return results[:max_results]  # Return up to max_results

            except (ConnectionError, ValueError, requests.exceptions.RequestException) as e:
                last_exception = e
                logger.warning(f"Baidu search attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying Baidu
            except Exception as e:  # Catch unexpected errors
                last_exception = e
                logger.error(f"Unexpected error during Baidu search attempt {attempt + 1}: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        logger.error(f"Baidu search failed after {max_retries} attempts.")
        raise ConnectionError(f"Baidu search failed: {last_exception}") from last_exception

    def _extract_results_from_soup(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """从 BeautifulSoup 对象中提取百度搜索结果。"""
        extracted_results = []
        # Baidu result containers (selectors might change)
        result_containers = soup.select('div.result.c-container, div.result-op.c-container')
        
        for container in result_containers:
            title = ''
            url = ''
            snippet = ''
            
            # Find title (usually within h3/a)
            title_elem = container.select_one('h3 > a')
            if title_elem:
                title = title_elem.get_text(strip=True)
                raw_url = title_elem.get('href')
                if raw_url and 'baidu.com/link?url=' in raw_url:
                    # Attempt to resolve redirect (basic attempt using requests)
                    try:
                        # HEAD request is faster, follow redirects
                        head_resp = self.session.head(raw_url, timeout=5, allow_redirects=True)
                        if head_resp.ok:
                            url = head_resp.url 
                            logger.debug(f"Resolved Baidu redirect: {raw_url} -> {url}")
                        else:
                            logger.warning(f"Failed to resolve Baidu redirect {raw_url}, status: {head_resp.status_code}")
                            url = raw_url  # Use original redirect url as fallback
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Error resolving Baidu redirect {raw_url}: {e}")
                        url = raw_url  # Fallback
                elif raw_url:
                    url = raw_url  # Assume it's a direct link

            # Find snippet
            snippet_elem = container.select_one('.c-abstract, .c-span-last .c-span18')
            if snippet_elem:
                snippet = snippet_elem.get_text(strip=True)
            
            # Simple URL validation before adding
            if title and url and url.startswith('http'):
                extracted_results.append({
                    'title': title,
                    'url': url,
                    'snippet': snippet
                })
        return extracted_results

# --- Search Engine Factory (Internal Use) ---
class _SearchEngineFactory:
    """Creates search engine instances (internal)."""
    
    _engine_map: Dict[str, Type[BaseSearchEngine]] = {
        'duckduckgo': DuckDuckGoSearch,
        'baidu': BaiduSearch
        # Add other engines like Google Custom Search, Bing etc.
    }

    @staticmethod
    def create_engine(engine_name: str, 
                      config: Dict[str, Any], 
                      api_keys: Dict[str, Any],
                      proxy: Optional[str] = None) -> BaseSearchEngine:
        """Creates a search engine instance based on name.

        Args:
            engine_name: Name of the engine (e.g., 'duckduckgo').
            config: Search engine specific configuration.
            api_keys: Dictionary of API keys.
            proxy: Optional proxy URL string.

        Returns:
            An instance of BaseSearchEngine.

        Raises:
            ValueError: If engine name is unsupported.
            ImportError: If required dependencies are missing.
        """
        engine_class = _SearchEngineFactory._engine_map.get(engine_name.lower())
        if not engine_class:
            raise ValueError(
                f"Unsupported search engine: '{engine_name}'. Available: {_SearchEngineFactory.get_available_engines()}"
            )
        
        try:
            # Pass config and api_keys to the engine's constructor
            return engine_class(config=config, api_keys=api_keys, proxy=proxy)
        except ImportError as e:
            raise ImportError(f"Failed to create engine '{engine_name}'. Missing dependency: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize search engine '{engine_name}': {e}") from e

    @staticmethod
    def get_available_engines() -> List[str]:
        """Returns a list of supported search engine names."""
        return list(_SearchEngineFactory._engine_map.keys())

# --- Main Searcher Class (Used by Orchestrator) ---
class Searcher:
    """Coordinates search operations using a configured search engine."""

    def __init__(self, config: Dict[str, Any], api_keys: Dict[str, Any]):
        """Initializes the Searcher.

        Args:
            config: Search configuration section (from scraping.search).
            api_keys: Dictionary containing all API keys.
        """
        self.config = config
        self.api_keys = api_keys
        self.default_provider = config.get('default_provider', 'duckduckgo')
        self.max_results = config.get('max_results', 5)
        self.proxy = config.get('proxy')
        self.engine: Optional[BaseSearchEngine] = None

        try:
            # Get engine-specific config (e.g., scraping.search.duckduckgo)
            provider_config = config.get(self.default_provider, {}).copy() # Use copy
            
            # --- Added: Merge global timeout if not set in provider_config ---
            if 'timeout' not in provider_config:
                global_timeout = config.get('timeout')
                if global_timeout is not None:
                    provider_config['timeout'] = global_timeout
                    logger.debug(f"Applying global search timeout ({global_timeout}s) to {self.default_provider} config.")
            # --- End Added ---
            
            # --- Added: Merge global max_retries if not set in provider_config ---
            if 'max_retries' not in provider_config:
                global_max_retries = config.get('max_retries')
                if global_max_retries is not None:
                    provider_config['max_retries'] = global_max_retries
                    logger.debug(f"Applying global search max_retries ({global_max_retries}) to {self.default_provider} config.")
            # --- End Added ---
            
            self.engine = _SearchEngineFactory.create_engine(
                engine_name=self.default_provider,
                config=provider_config,  # Pass potentially merged provider_config
                api_keys=api_keys,
                proxy=self.proxy
            )
            logger.info(f"Searcher initialized with engine: {self.default_provider}")
        except (ValueError, ImportError, RuntimeError) as e:
            logger.error(
                f"Failed to initialize search engine '{self.default_provider}': {e}. "
                "Search functionality will be disabled."
            )
            self.engine = None  # Disable search if initialization fails

    def search(self, query: str) -> List[Dict[str, str]]:
        """Performs a search using the configured engine.

        Args:
            query: The search query.

        Returns:
            A list of search result dictionaries, or an empty list on error.
        """
        if not self.engine:
            logger.error("Search engine not initialized. Cannot perform search.")
            return []
        
        try:
            logger.info(f"Performing search via {self.default_provider}: '{query[:50]}...'")
            results = self.engine.search(query, max_results=self.max_results)
            logger.info(f"Search returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Search failed using engine {self.default_provider}: {e}", exc_info=True)
            return []  # Return empty list on error

# Example Usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example Config (replace with actual loaded config)
    test_config = {
        'default_provider': 'duckduckgo',
        'max_results': 3,
        'duckduckgo': {  # Provider specific config
            'timeout': 15
        },
        'baidu': {  # Provider specific config
             'max_retries': 1
        }
    }
    test_api_keys = {}  # No API keys needed for DDG/Baidu here

    print("--- Testing DuckDuckGo Searcher ---")
    try:
        ddg_searcher = Searcher(config=test_config, api_keys=test_api_keys)
        ddg_results = ddg_searcher.search("Python web scraping libraries")
        print("DuckDuckGo Results:")
        for i, res in enumerate(ddg_results):
            print(f"  {i+1}. {res['title']} ({res['url']})")
    except Exception as e:
        print(f"Error testing DuckDuckGo: {e}")

    print("\n--- Testing Baidu Searcher (May fail due to blocking/structure changes) ---")
    baidu_test_config = test_config.copy()
    baidu_test_config['default_provider'] = 'baidu'
    try:
        baidu_searcher = Searcher(config=baidu_test_config, api_keys=test_api_keys)
        baidu_results = baidu_searcher.search("Python 网络爬虫库")
        print("Baidu Results:")
        for i, res in enumerate(baidu_results):
            print(f"  {i+1}. {res['title']} ({res['url']})")
    except Exception as e:
        print(f"Error testing Baidu: {e}")
