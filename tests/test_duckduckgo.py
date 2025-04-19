from duckduckgo_search import DDGS
import pytest
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_duckduckgo_search():
    proxy = os.getenv('PROXY_SERVER')
    
    # 创建 DDGS 实例，使用代理
    with DDGS(timeout=30, proxy=proxy) as ddgs:
        # 执行搜索 (使用应用中失败的查询)
        search_query = "Silicon Valley AI startups Series A funding"
        print(f"Testing search with query: '{search_query}'")
        results = list(ddgs.text(search_query, max_results=5))
        
        # 验证结果
        assert len(results) > 0, "搜索结果不应为空"
        assert all(isinstance(result, dict) for result in results), "所有结果应该是字典类型"
        assert all('title' in result for result in results), "每个结果都应该包含标题"
        assert all('href' in result for result in results), "每个结果都应该包含链接"
        assert all('body' in result for result in results), "每个结果都应该包含正文"
        
        # 打印第一个结果作为示例
        print("\n第一个搜索结果：")
        print(f"标题: {results[0]['title']}")
        print(f"链接: {results[0]['href']}")
        print(f"摘要: {results[0]['body'][:200]}...")

if __name__ == "__main__":
    test_duckduckgo_search() 