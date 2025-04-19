from typing import List, Dict, Any, Optional, Union

# Use relative import for the base class
from .base import LLMProvider

# Import the specific client library
try:
    import anthropic
    from anthropic import AuthenticationError, RateLimitError, APIConnectionError, AnthropicError
except ImportError:
    raise ImportError("AnthropicProvider requires the Anthropic library. Install it with: pip install anthropic")

class AnthropicProvider(LLMProvider):
    """使用 anthropic Python 库与 Anthropic API 交互的 LLM 提供商。"""

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """初始化 Anthropic 客户端。
        
        Args:
            api_key: Anthropic API 密钥。
            base_url: 自定义 API 基础 URL (可选)。
            **kwargs: 传递给 Anthropic() 构造函数的其他参数 (例如 timeout, max_retries)。
        """
        super().__init__(api_key, base_url, **kwargs)
        try:
            self.client = anthropic.Anthropic(
                api_key=self.api_key,
                base_url=self.base_url,
                **self.init_kwargs # Pass extra args like timeout, max_retries
            )
        except Exception as e:
             raise RuntimeError(f"Failed to initialize Anthropic client: {e}") from e

    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = 1000, # Required by Anthropic API
                       response_format: Optional[Dict[str, str]] = None, # Anthropic does not support JSON mode directly
                       **kwargs
                       ) -> Union[str, Dict]: # Always returns string for Anthropic currently
        """使用 Anthropic API 执行聊天补全 (Messages API)。"""
        if response_format:
             # print("Warning: Anthropic provider does not support response_format parameter.")
             pass # Ignore the parameter
             
        if max_tokens is None:
             # Anthropic requires max_tokens
             raise ValueError("max_tokens parameter is required for Anthropic API calls.")

        try:
            # Separate system prompt and messages
            system_prompt = None
            anthropic_messages = []
            if messages and messages[0]['role'] == 'system':
                system_prompt = messages[0]['content']
                user_assistant_msgs = messages[1:]
            else:
                user_assistant_msgs = messages
                
            # Validate message roles (must alternate user/assistant)
            last_role = None
            for i, msg in enumerate(user_assistant_msgs):
                role = msg.get('role')
                if role not in ['user', 'assistant']:
                    raise ValueError(f"Invalid role '{role}' in message {i}. Must be 'user' or 'assistant'.")
                if i > 0 and role == last_role:
                     # Merge consecutive messages or raise error? Anthropic expects alternation.
                     # Let's try merging for now, though API might still reject.
                     # print(f"Warning: Merging consecutive {role} messages for Anthropic.")
                     last_message = anthropic_messages[-1]
                     if isinstance(last_message['content'], list):
                          last_message['content'].append({"type": "text", "text": msg['content']})
                     else: # Assume previous was string
                          last_message['content'] = f"{last_message['content']}\n\n{msg['content']}"
                     continue # Skip adding this as a new message
                
                anthropic_messages.append({'role': role, 'content': msg['content']})
                last_role = role
                
            # The API requires the first message to be from the user role.
            if not anthropic_messages or anthropic_messages[0]['role'] != 'user':
                 raise ValueError("Anthropic conversation must start with a user message.")
                 
            # Prepare API call parameters
            completion_params = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                # Only include system prompt if it exists
                **({"system": system_prompt} if system_prompt else {}),
                **kwargs # Pass other API params
            }
            
            response = self.client.messages.create(**completion_params)

            # Extract text content from the response blocks
            response_text = ""
            if response.content:
                 for block in response.content:
                      if block.type == 'text':
                           response_text += block.text
            
            # Check stop reason
            if response.stop_reason == 'max_tokens':
                 # print("Warning: Anthropic response stopped due to max_tokens.")
                 pass # Content might be incomplete
                 
            return response_text

        except AuthenticationError as e:
            # print(f"Anthropic Authentication Error: {e}")
            raise PermissionError(f"Anthropic API Authentication Failed: {e}") from e
        except RateLimitError as e:
            # print(f"Anthropic Rate Limit Error: {e}")
            raise ConnectionAbortedError(f"Anthropic API Rate Limit Exceeded: {e}") from e
        except APIConnectionError as e:
             # print(f"Anthropic Connection Error: {e}")
             raise ConnectionError(f"Failed to connect to Anthropic API: {e}") from e
        except AnthropicError as e:
            # Handle other generic Anthropic errors
            # print(f"Anthropic API Error: {e}")
            raise RuntimeError(f"Anthropic API Error: {e}") from e
        except Exception as e:
            # print(f"Unexpected error during Anthropic chat completion: {e}")
            raise RuntimeError(f"Anthropic API call failed unexpectedly: {e}") from e

    # Anthropic does not provide a standard API endpoint to list models.
    # Rely on documentation or hardcoded list.
    def get_available_models(self) -> List[str]:
        """返回已知的可用 Anthropic 模型列表 (基于文档)。"""
        # This list should be updated based on current Anthropic documentation
        return sorted([ 
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ])

    def validate_credentials(self) -> bool:
        """通过尝试发送一个简短的测试消息来验证 Anthropic API 凭证。"""
        try:
            self.client.messages.create(
                # Use a cheaper/faster model for validation if possible
                model="claude-3-haiku-20240307", 
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except AuthenticationError:
            return False
        except Exception:
            # Other errors might occur, assume failure for validation
            return False 