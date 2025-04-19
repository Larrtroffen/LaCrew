from typing import List, Dict, Any, Optional, Union

# Use relative import for the base class
from .base import LLMProvider

# Import the specific client library
try:
    from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError, OpenAIError
except ImportError:
    raise ImportError("OpenAIProvider requires the OpenAI library. Install it with: pip install openai")

class OpenAIProvider(LLMProvider):
    """使用 openai Python 库与 OpenAI API 交互的 LLM 提供商。"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """初始化 OpenAI 客户端。
        
        Args:
            api_key: OpenAI API 密钥。
            base_url: 自定义 API 基础 URL (可选, 用于代理或兼容的 API)。
            **kwargs: 传递给 OpenAI() 构造函数的其他参数 (例如 timeout, max_retries)。
        """
        super().__init__(api_key, base_url, **kwargs)
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                **self.init_kwargs # Pass extra args like timeout, max_retries
            )
            # Optional: Add a validation check here if desired, 
            # but often it's better to let the first call fail.
            # self.validate_credentials() 
        except Exception as e:
             raise RuntimeError(f"Failed to initialize OpenAI client: {e}") from e

    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = 1000,
                       response_format: Optional[Dict[str, str]] = None,
                       **kwargs
                       ) -> Union[str, Dict]:
        """使用 OpenAI API 执行聊天补全。"""
        try:
            completion_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                # Only include response_format if it's specified and not None
                **({"response_format": response_format} if response_format else {}),
                **kwargs # Pass any other provider-specific kwargs
            }
            # Filter out None values for arguments like max_tokens
            completion_params = {k: v for k, v in completion_params.items() if v is not None}
            
            # print(f"DEBUG: OpenAI request params: {completion_params}") # Debug logging
            
            response = self.client.chat.completions.create(**completion_params)
            
            # Handle different response formats
            choice = response.choices[0]
            if choice.message.content:
                 return choice.message.content
            # Handle cases where response might be structured (e.g., function calling - though not explicitly supported here yet)
            # For now, primarily expect text content. 
            # If JSON mode was requested, the content string itself should be the JSON.
            elif response_format and response_format.get('type') == 'json_object':
                 # The content *should* be a parsable JSON string in this case
                 # Returning the string itself, let the caller parse.
                 # Or potentially parse it here? Let's return string for now.
                 # print("Warning: JSON object requested, but message content is empty. Returning raw choice.")
                 # Potentially return choice.message object or empty string?
                 return choice.message.content if choice.message.content is not None else "{}" # Return empty JSON string if content is None
                 
            # Fallback or error if no content found
            # print(f"Warning: OpenAI response choice had no content. Finish reason: {choice.finish_reason}")
            return "" # Return empty string if no content

        except AuthenticationError as e:
            # print(f"OpenAI Authentication Error: {e}")
            raise PermissionError(f"OpenAI API Authentication Failed: {e}") from e
        except RateLimitError as e:
            # print(f"OpenAI Rate Limit Error: {e}")
            # Consider adding retry logic here or in the caller
            raise ConnectionAbortedError(f"OpenAI API Rate Limit Exceeded: {e}") from e
        except APIConnectionError as e:
             # print(f"OpenAI Connection Error: {e}")
             raise ConnectionError(f"Failed to connect to OpenAI API: {e}") from e
        except OpenAIError as e:
            # Handle other generic OpenAI errors
            # print(f"OpenAI API Error: {e} (Type: {e.type}, Code: {e.code})")
            raise RuntimeError(f"OpenAI API Error ({e.type}/{e.code}): {e}") from e
        except Exception as e:
            # Catch any other unexpected errors
            # print(f"Unexpected error during OpenAI chat completion: {e}")
            raise RuntimeError(f"OpenAI API call failed unexpectedly: {e}") from e

    # Getting available models dynamically might be better than a hardcoded list
    def get_available_models(self) -> List[str]:
        """获取可用的 OpenAI 模型列表 (通过 API 调用)。"""
        try:
            models_response = self.client.models.list()
            # Filter for models that support chat completion (heuristically)
            # This filtering logic might need adjustment based on OpenAI's model naming/capabilities
            available_models = [model.id for model in models_response.data 
                                if 'gpt' in model.id and 'instruct' not in model.id]
            return sorted(available_models)
        except Exception as e:
            # print(f"Failed to retrieve models from OpenAI API: {e}. Returning fallback list.")
            # Fallback to a predefined list if API call fails
            return sorted([ 
                "gpt-4-turbo", "gpt-4-turbo-preview", "gpt-4", "gpt-4-0613",
                "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0125"
            ])

    def validate_credentials(self) -> bool:
        """通过尝试列出模型来验证 OpenAI API 凭证。"""
        try:
            self.client.models.list()
            return True
        except AuthenticationError:
            return False
        except Exception:
            # Other errors (connection, etc.) don't necessarily mean invalid credentials
            # but we can count them as failure for this basic check.
            return False 