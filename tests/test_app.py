import pytest
from llm_core import LLMCore
from web_interaction import WebInteraction
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@pytest.fixture
def llm_core():
    return LLMCore(
        provider_name="openai",
        api_key=os.getenv('OPENAI_API_KEY'),
        model_name="gpt-3.5-turbo"
    )

@pytest.fixture
def web_interaction():
    return WebInteraction()

@pytest.fixture
def sample_schema():
    return pd.DataFrame({
        'Field': ['Title', 'Description', 'URL'],
        'Type': ['Text', 'Text', 'URL'],
        'Required': [True, True, True]
    })

def test_app_with_adjustable_parameters(llm_core, web_interaction, sample_schema):
    """Test the application with adjustable parameters"""
    
    # Test parameters
    test_cases = [
        {
            "description": "Test basic search with default parameters",
            "task_description": "Find information about Python programming",
            "max_pages": 3,
            "domain_restriction": None,
            "expected_min_results": 1
        },
        {
            "description": "Test search with domain restriction",
            "task_description": "Find Python tutorials on python.org",
            "max_pages": 2,
            "domain_restriction": "python.org",
            "expected_min_results": 1
        },
        {
            "description": "Test search with limited pages",
            "task_description": "Find information about machine learning",
            "max_pages": 1,
            "domain_restriction": None,
            "expected_min_results": 1
        }
    ]

    for case in test_cases:
        print(f"\nRunning test case: {case['description']}")
        
        # Execute the task
        results = llm_core.execute_task(
            task_description=case['task_description'],
            data_schema=sample_schema,
            web_interaction=web_interaction,
            max_pages=case['max_pages'],
            domain_restriction=case['domain_restriction']
        )
        
        # Assertions
        assert not results.empty, f"Test failed: {case['description']} - No results returned"
        assert len(results) >= case['expected_min_results'], \
            f"Test failed: {case['description']} - Expected at least {case['expected_min_results']} results"
        
        # Verify schema compliance
        for field in sample_schema['Field']:
            assert field in results.columns, \
                f"Test failed: {case['description']} - Missing required field: {field}"
        
        print(f"Test passed: {case['description']}")

def test_app_with_invalid_parameters(llm_core, web_interaction, sample_schema):
    """Test the application with invalid parameters"""
    
    # Test invalid max_pages
    with pytest.raises(ValueError):
        llm_core.execute_task(
            task_description="Test invalid parameters",
            data_schema=sample_schema,
            web_interaction=web_interaction,
            max_pages=0
        )
    
    # Test empty task description
    with pytest.raises(ValueError):
        llm_core.execute_task(
            task_description="",
            data_schema=sample_schema,
            web_interaction=web_interaction,
            max_pages=1
        )