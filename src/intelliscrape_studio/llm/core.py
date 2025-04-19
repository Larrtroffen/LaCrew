import json
import logging
from typing import Dict, Any, Optional, Type, cast

from ..data.models import TaskState, OrchestratorAction, DataTableSchema, LLMResponse, Action
from .prompts import (
    SELECT_ACTION_SYSTEM_PROMPT,
    SELECT_ACTION_USER_PROMPT_TEMPLATE,
    EXTRACT_DATA_SYSTEM_PROMPT,
    EXTRACT_DATA_USER_PROMPT_TEMPLATE,
    format_schema_for_prompt,
    format_history_for_prompt
)
from .providers.base import LLMProvider
from .providers.factory import LLMProviderFactory
from pydantic import ValidationError, parse_obj_as

logger = logging.getLogger(__name__)

# Mapping from action type string to Pydantic model class
# This helps in parsing the LLM response into the correct Action subclass
ACTION_TYPE_MAP: Dict[str, Type[Action]] = {
    action_cls.__fields__['type'].default: action_cls
    for action_cls in Action.__subclasses__() # Action.__subclasses__() doesn't work as expected in some setups
    # if action_cls.__fields__.get('type') and action_cls.__fields__['type'].default
}
# Manual mapping if subclasses don't work reliably
if not ACTION_TYPE_MAP:
    from ..data.models import SearchAction, VisitURLAction, ClickElementAction, FillInputAction, ExtractDataAction, FillTableAction, FinishAction
    ACTION_TYPE_MAP = {
        'Search': SearchAction,
        'VisitURL': VisitURLAction,
        'ClickElement': ClickElementAction,
        'FillInput': FillInputAction,
        'ExtractData': ExtractDataAction,
        'FillTable': FillTableAction,
        'Finish': FinishAction
    }


class LLMCore:
    """Handles interaction with the LLM, including prompt formatting and response parsing."""

    def __init__(self, config: Dict[str, Any]):
        """Initializes the LLMCore.
        
        Args:
            config: Configuration dictionary for the LLM module (e.g., from config_loader).
        """
        self.config = config
        self.provider_name = config.get('default_provider', 'openai')
        self.model_name = config.get('default_model', 'gpt-4o-mini') # Adjust default as needed
        self.api_key = config.get('api_keys', {}).get(self.provider_name)
        self.request_timeout = config.get('request_timeout', 120)
        self.max_retries = config.get('max_retries', 3)
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2048)
        # Extract base_url from config, defaulting to None if not present
        self.base_url = config.get('base_url') 

        if not self.api_key:
            logger.warning(f"API key for provider '{self.provider_name}' not found in config.")
            # Decide if this should be a hard error or allow operation without API key (e.g., for dummy providers)
            # raise ValueError(f"API key for provider '{self.provider_name}' is required.")
            self.provider: Optional[LLMProvider] = None
        else:
            try:
                # Pass base_url explicitly to the factory
                self.provider = LLMProviderFactory.create_provider(
                    provider_name=self.provider_name,
                    api_key=self.api_key,
                    base_url=self.base_url # Pass extracted base_url
                )
                logger.info(f"LLM Provider '{self.provider_name}' initialized with model '{self.model_name}'" + (f" using base URL '{self.base_url}'." if self.base_url else "."))
            except ValueError as e:
                 logger.error(f"Failed to initialize LLM provider '{self.provider_name}': {e}")
                 self.provider = None
            except Exception as e:
                 logger.error(f"Unexpected error initializing LLM provider '{self.provider_name}': {e}", exc_info=True)
                 self.provider = None

    def _call_llm(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Internal helper to call the LLM provider with error handling and retries."""
        if not self.provider:
            return LLMResponse(error="LLM Provider not initialized.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        for attempt in range(self.max_retries):
            try:
                response_text = self.provider.chat_completion(
                messages=messages,
                model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.request_timeout
                    # Pass other provider-specific params if needed
                )
                if response_text:
                    cleaned_text = response_text.strip()
                    # Basic cleaning (remove potential markdown fences if LLM adds them)
                    if cleaned_text.startswith('```json'):
                        cleaned_text = cleaned_text[7:-3].strip()
                    elif cleaned_text.startswith('```'):
                        cleaned_text = cleaned_text[3:-3].strip()
                    # --- Added Cleaning: Handle potential extra outer braces --- 
                    elif cleaned_text.startswith('{{') and cleaned_text.endswith('}}'):
                        cleaned_text = cleaned_text[1:-1].strip() # Remove one layer of braces
                        logger.debug("Removed potential extra outer braces from LLM response.")
                    # --- End Added Cleaning ---
                    
                    return LLMResponse(content=cleaned_text)
                else:
                    raise ValueError("LLM returned an empty response.")

            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt + 1 == self.max_retries:
                    logger.error(f"LLM call failed after {self.max_retries} attempts.", exc_info=True)
                    return LLMResponse(error=f"LLM call failed after retries: {e}")
                # Optional: time.sleep(2**attempt) # Exponential backoff
        
        # Should not be reached if retries are configured
        return LLMResponse(error="LLM call failed after retries.") 

    def select_next_action(self, state: TaskState) -> Optional[OrchestratorAction]:
        """Asks the LLM to select the next action based on the current task state."""
        if not self.provider:
            logger.error("Cannot select action: LLM Provider not initialized.")
            return None # Or return a specific error action

        schema_desc = format_schema_for_prompt(state.target_schema)
        history_summary = format_history_for_prompt(state.history, limit=5) # Limit history length
        
        # Truncate page content snippet to avoid excessive prompt length
        snippet_limit = 2000 # Characters
        page_snippet = (state.page_content or "")[:snippet_limit]
        if len(state.page_content or "") > snippet_limit:
            page_snippet += "... (truncated)"

        user_prompt = SELECT_ACTION_USER_PROMPT_TEMPLATE.format(
            task_description=state.task_description,
            schema_description=schema_desc,
            current_url=state.current_url or "None",
            visited_urls=str(state.visited_urls),
            page_content_snippet=page_snippet,
            snippet_length=snippet_limit,
            history_summary=history_summary,
            history_limit=5
        )

        llm_response = self._call_llm(SELECT_ACTION_SYSTEM_PROMPT, user_prompt)

        if llm_response.error or not llm_response.content:
            logger.error(f"LLM failed to provide next action: {llm_response.error}")
            return None # Or return a FinishAction with error

        logger.debug(f"LLM action response (raw): {llm_response.content}")

        try:
            action_data = json.loads(llm_response.content)
            action_type_str = action_data.get('type')

            if not action_type_str:
                 raise ValueError("LLM response JSON missing 'type' field.")

            action_cls = ACTION_TYPE_MAP.get(action_type_str)
            if not action_cls:
                raise ValueError(f"Unknown action type received from LLM: '{action_type_str}'")

            # Parse using the specific Action subclass model
            parsed_action = parse_obj_as(action_cls, action_data)
            # parsed_action = action_cls(**action_data) # Alternative parsing
            
            # Cast for type hinting, though parse_obj_as should return correct type
            return cast(OrchestratorAction, parsed_action)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode LLM action response JSON: {e}. Response: {llm_response.content}")
            return None # Or return FinishAction
        except ValidationError as e:
            logger.error(f"Failed to validate LLM action against Pydantic model: {e}. Response: {llm_response.content}")
            return None # Or return FinishAction
        except ValueError as e:
             logger.error(f"Error processing LLM action response: {e}. Response: {llm_response.content}")
             return None # Or return FinishAction
        except Exception as e:
             logger.error(f"Unexpected error parsing LLM action: {e}. Response: {llm_response.content}", exc_info=True)
             return None # Or return FinishAction

    def extract_structured_data(self, 
                                schema: DataTableSchema, 
                                html_content: str, 
                                task_description: str) -> Optional[Dict[str, Any]]:
        """Asks the LLM to extract structured data from HTML based on the schema."""
        if not self.provider:
            logger.error("Cannot extract data: LLM Provider not initialized.")
            return None

        # Limit content length passed to LLM if necessary (check provider limits)
        max_content_length = 30000 # Example limit, adjust based on model context window
        if len(html_content) > max_content_length:
             logger.warning(f"HTML content length ({len(html_content)}) exceeds limit ({max_content_length}). Truncating.")
             html_content = html_content[:max_content_length]

        schema_desc = format_schema_for_prompt(schema)
        system_prompt = EXTRACT_DATA_SYSTEM_PROMPT.format(
            schema_description=schema_desc,
            task_description=task_description
        )
        user_prompt = EXTRACT_DATA_USER_PROMPT_TEMPLATE.format(html_content=html_content)

        llm_response = self._call_llm(system_prompt, user_prompt)

        if llm_response.error or not llm_response.content:
            logger.error(f"LLM failed to extract data: {llm_response.error}")
        return None
    
        logger.debug(f"LLM extracted data response (raw): {llm_response.content}")

        try:
            extracted_data = json.loads(llm_response.content)
            if not isinstance(extracted_data, dict):
                 raise ValueError("LLM did not return a JSON object (dict) for extracted data.")
            
            # Optional: Validate against schema (e.g., check if all keys exist, basic type checks)
            # This can be complex; relying on LLM format instructions for now.
            logger.info(f"Successfully extracted data: {list(extracted_data.keys())}")
            return extracted_data
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode LLM extracted data JSON: {e}. Response: {llm_response.content}")
            return None
        except ValueError as e:
            logger.error(f"Error processing LLM extracted data: {e}. Response: {llm_response.content}")
            return None
        except Exception as e:
             logger.error(f"Unexpected error parsing LLM extracted data: {e}. Response: {llm_response.content}", exc_info=True)
        return None
    
# Example usage placeholder
if __name__ == '__main__':
    # Requires config/settings.yaml and .env with API key
    logging.basicConfig(level=logging.DEBUG)
    from ..core.config_loader import load_config, get_config
    from ..data.models import ColumnDefinition
    try:
        load_config()
        config = get_config()
        llm_core = LLMCore(config=config.get('llm', {}))

        if not llm_core.provider:
             print("LLM Provider failed to initialize. Cannot run tests.")
        else:
            # --- Test Action Selection --- 
            print("\n--- Testing Action Selection ---")
            test_schema = DataTableSchema(columns=[
                ColumnDefinition(name="Name", dtype="Text", description="Person's full name"),
                ColumnDefinition(name="Email", dtype="Email", description="Contact email")
            ])
            test_state = TaskState(
                task_id="test-123",
                task_description="Find contact info for AI researchers",
                target_schema=test_schema,
                current_url="http://example.com/profiles",
                page_content="<html><body><h1>Profiles</h1><p>John Doe - john@example.com</p></body></html>",
                visited_urls={"http://example.com"}
            )
            next_action = llm_core.select_next_action(test_state)
            if next_action:
                print(f"Selected Action: {next_action.type}")
                print(f"Parameters: {next_action.parameters}")
                print(f"Reasoning: {next_action.reasoning}")
            else:
                print("Failed to select next action.")

            # --- Test Data Extraction --- 
            print("\n--- Testing Data Extraction ---")
            test_html = "<html><body><h2>Contact Us</h2><p>Reach out to Jane Doe at <a href='mailto:jane.doe@example.com'>jane.doe@example.com</a>.</p><p>Founded: 2023</p></body></html>"
            extract_schema = DataTableSchema(columns=[
                ColumnDefinition(name="Contact Name", dtype="Text", description="Full name of the contact person"),
                ColumnDefinition(name="Email Address", dtype="Email", description="Email address"),
                 ColumnDefinition(name="Year Founded", dtype="Number", description="The year the company was founded")
            ])
            extracted = llm_core.extract_structured_data(extract_schema, test_html, "Extract contact details")
            if extracted:
                print("Extracted Data:")
                print(json.dumps(extracted, indent=2))
            else:
                print("Failed to extract data.")

    except FileNotFoundError:
        print("Error: config/settings.yaml or .env not found. Cannot run LLMCore example.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() 