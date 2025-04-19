from typing import List, Dict, Any, Optional, Union

# Use relative import for the base class
from .base import LLMProvider

# Import the specific client library
try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
    from google.generativeai.types import generation_types
except ImportError:
    raise ImportError("GeminiProvider requires the Google Generative AI library. Install it with: pip install google-generativeai")

class GeminiProvider(LLMProvider):
    """使用 google-generativeai Python 库与 Google Gemini API 交互的 LLM 提供商。"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """初始化 Gemini 客户端。
        
        Args:
            api_key: Google AI API 密钥。
            base_url: Google AI 不直接支持 base_url，此参数将被忽略。
            **kwargs: 其他 genai.configure 参数 (例如 transport)。
        """
        super().__init__(api_key, base_url, **kwargs)
        if base_url:
             print("Warning: GeminiProvider does not support the 'base_url' parameter.")
        try:
            # Configure the library with the API key and any other relevant kwargs
            genai.configure(
                 api_key=self.api_key,
                 **self.init_kwargs # Pass extra args like transport
            )
            self.client = genai
            # Validate credentials during init? Optional, but can catch issues early.
            # self.validate_credentials()
        except Exception as e:
             raise RuntimeError(f"Failed to configure Google Generative AI: {e}") from e

    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str,
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = 1000,
                       response_format: Optional[Dict[str, str]] = None, # Note: Gemini has limited JSON support
                       **kwargs
                       ) -> Union[str, Dict]:
        """使用 Gemini API 执行聊天补全。"""
        if response_format and response_format.get('type') == 'json_object':
             # print("Warning: Gemini API does not explicitly support a dedicated JSON object mode like OpenAI.")
             # Instructing the model via prompt is the typical way.
             # We will proceed, but the output might not be guaranteed JSON.
             pass 
             
        try:
            # Instantiate the model
            generative_model = self.client.GenerativeModel(model_name=model)
            
            # Convert messages to Gemini format (alternating user/model roles)
            # Handle system prompt (often prepended to the first user message)
            gemini_history = []
            system_prompt = None
            processed_messages = []
            
            # Find system prompt
            if messages and messages[0]['role'] == 'system':
                system_prompt = messages[0]['content']
                processed_messages = messages[1:]
            else:
                 processed_messages = messages
                 
            # Build history, prepending system prompt if needed
            current_role = None
            content_buffer = []
            first_user_message = True
            
            for msg in processed_messages:
                 role = msg['role']
                 content = msg['content']
                 
                 # Gemini expects alternating user/model roles
                 gemini_role = 'user' if role == 'user' else 'model'
                 
                 if first_user_message and gemini_role == 'user' and system_prompt:
                      content = f"{system_prompt}\n\n{content}" # Prepend system prompt
                      first_user_message = False
                      
                 if current_role is None: # First message
                      current_role = gemini_role
                      content_buffer.append(content)
                 elif role == current_role:
                      # Merge consecutive messages from the same role
                      content_buffer.append(content)
                 else:
                      # Role changed, add previous message block to history
                      gemini_history.append({'role': current_role, 'parts': ["\n".join(content_buffer)]})
                      # Start new block
                      current_role = gemini_role
                      content_buffer = [content]
                      
            # Add the last message block
            if current_role and content_buffer:
                 gemini_history.append({'role': current_role, 'parts': ["\n".join(content_buffer)]})
                 
            # The last message in the history must be from the 'user' role
            if not gemini_history or gemini_history[-1]['role'] != 'user':
                # This case might happen if the input ends with an assistant message.
                # Gemini API might error. We might need to append a dummy user message?
                # Or raise an error here?
                raise ValueError("Gemini chat history must end with a user message.")
                
            # Extract the last user message content for the generate_content call
            last_user_content = gemini_history.pop()['parts']
                
            # Start chat with history (excluding the last user message)
            chat = generative_model.start_chat(history=gemini_history)
            
            # Configure generation parameters
            generation_config = generation_types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )

            # Send the last user message to generate the response
            response = chat.send_message(
                content=last_user_content,
                generation_config=generation_config,
                **kwargs # Pass extra args if needed
            )

            # Extract the text content
            # Handle potential errors or empty responses
            if response.parts:
                return response.text # response.text concatenates parts
            else:
                 # Check finish reason if available
                 finish_reason = getattr(response, 'prompt_feedback', {}).get('block_reason', None)
                 if finish_reason:
                      # print(f"Warning: Gemini response blocked. Reason: {finish_reason}")
                      raise RuntimeError(f"Gemini response blocked due to: {finish_reason}")
                 else:
                      # print("Warning: Gemini response has no content parts.")
                      return "" # Return empty string if no parts

        except google_exceptions.PermissionDenied as e:
             # print(f"Gemini Permission Denied Error: {e}")
             raise PermissionError(f"Gemini API Permission Denied (check API key?): {e}") from e
        except google_exceptions.ResourceExhausted as e:
             # print(f"Gemini Rate Limit Error: {e}")
             raise ConnectionAbortedError(f"Gemini API Rate Limit Exceeded: {e}") from e
        except google_exceptions.InvalidArgument as e:
            # print(f"Gemini Invalid Argument Error: {e}")
            # Could be bad model name, invalid roles, etc.
            raise ValueError(f"Gemini API Invalid Argument: {e}") from e
        except Exception as e:
            # Catch other potential errors from the library or API
            # print(f"Unexpected error during Gemini chat completion: {e}")
            raise RuntimeError(f"Gemini API call failed unexpectedly: {e}") from e

    def get_available_models(self) -> List[str]:
        """获取可用的 Gemini 模型列表 (通过 API 调用)。"""
        try:
            models = self.client.list_models()
            # Filter for models supporting generateContent (chat-like interactions)
            available_models = [m.name for m in models 
                                if 'generateContent' in m.supported_generation_methods]
            # Further filter for common model name patterns if desired
            available_models = [m.split('/')[-1] for m in available_models if 'gemini' in m]
            return sorted(list(set(available_models))) # Get unique names
        except Exception as e:
            # print(f"Failed to retrieve models from Google AI API: {e}. Returning fallback list.")
            # Fallback list (might become outdated)
            return sorted([ 
                "gemini-1.0-pro", "gemini-1.5-flash", "gemini-1.5-pro", 
                # Add vision or other variants if needed explicitly
                # "gemini-pro-vision"
            ])

    def validate_credentials(self) -> bool:
        """通过尝试列出模型来验证 Google AI API 凭证。"""
        try:
            self.client.list_models()
            return True
        except google_exceptions.PermissionDenied:
            return False
        except Exception:
            # Other errors might not indicate invalid credentials, but fail validation here
            return False 