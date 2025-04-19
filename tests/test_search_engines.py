import pytest
from search_engines.duckduckgo import DuckDuckGoSearchEngine
import logging
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def search_engine():
    """创建搜索引擎实例的fixture"""
    return DuckDuckGoSearchEngine()

def test_duckduckgo_search(search_engine):
    """测试DuckDuckGo搜索引擎是否能正常返回搜索结果"""
    logger.info("开始测试基本搜索功能")
    
    # 执行搜索
    results = search_engine.search(
        query="Python programming",
        max_results=5,
        region="wt-wt",
        safesearch="moderate"
    )
    
    logger.info(f"搜索返回 {len(results)} 条结果")
    
    # 验证结果
    assert len(results) > 0, "搜索结果不应为空"
    assert len(results) <= 5, "结果数量不应超过max_results"
    
    # 验证每个结果的结构
    for i, result in enumerate(results):
        logger.info(f"检查第 {i+1} 条结果的结构")
        assert "title" in result, "结果应包含title字段"
        assert "url" in result, "结果应包含url字段"
        assert "snippet" in result, "结果应包含snippet字段"
        assert isinstance(result["title"], str), "title应为字符串"
        assert isinstance(result["url"], str), "url应为字符串"
        assert isinstance(result["snippet"], str), "snippet应为字符串"
        assert result["url"].startswith("http"), "url应以http开头"

def test_duckduckgo_search_with_region(search_engine):
    """测试不同地区的搜索结果"""
    logger.info("开始测试地区搜索功能")
    
    # 测试中文区域搜索
    logger.info("执行中文区域搜索")
    results_cn = search_engine.search(
        query="Python编程",
        max_results=5,
        region="cn-zh",
        safesearch="moderate"
    )
    
    logger.info(f"中文搜索返回 {len(results_cn)} 条结果")
    
    # 测试英文区域搜索
    logger.info("执行英文区域搜索")
    results_en = search_engine.search(
        query="Python programming",
        max_results=5,
        region="us-en",
        safesearch="moderate"
    )
    
    logger.info(f"英文搜索返回 {len(results_en)} 条结果")
    
    assert len(results_cn) > 0, "中文搜索结果不应为空"
    assert len(results_en) > 0, "英文搜索结果不应为空"
    
    # 验证结果结构
    for results in [results_cn, results_en]:
        for result in results:
            assert all(key in result for key in ["title", "url", "snippet"]), "结果应包含所有必要字段" 