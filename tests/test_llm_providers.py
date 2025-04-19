import pytest
from llm_providers.openai_provider import OpenAIProvider
from llm_providers.anthropic_provider import AnthropicProvider
from llm_providers.gemini_provider import GeminiProvider

def test_openai_provider_initialization():
    provider = OpenAIProvider(api_key="test_key")
    assert provider.api_key == "test_key"
    assert provider.model == "gpt-3.5-turbo"

def test_anthropic_provider_initialization():
    provider = AnthropicProvider(api_key="test_key")
    assert provider.api_key == "test_key"
    assert provider.model == "claude-3-opus-20240229"

def test_gemini_provider_initialization():
    provider = GeminiProvider(api_key="test_key")
    assert provider.api_key == "test_key"
    assert provider.model == "gemini-pro" 