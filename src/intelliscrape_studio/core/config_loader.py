import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Determine the base directory of the project (assuming this file is in src/intelliscrape_studio/core)
# Adjust the number of .parent calls if the file structure changes
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "settings.yaml"
ENV_PATH = BASE_DIR / ".env"

_config: Optional[Dict[str, Any]] = None

def load_config(config_path: Optional[Path] = None, env_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads configuration from YAML and .env files.

    Args:
        config_path: Path to the main YAML configuration file.
                     Defaults to `PROJECT_ROOT/config/settings.yaml`.
        env_path: Path to the .env file.
                  Defaults to `PROJECT_ROOT/.env`.

    Returns:
        A dictionary containing the merged configuration.
    """
    global _config
    if _config is not None:
        return _config

    # Use provided paths or defaults
    resolved_config_path = config_path or DEFAULT_CONFIG_PATH
    resolved_env_path = env_path or ENV_PATH

    # Load .env file first (environment variables override .env)
    if resolved_env_path.exists():
        load_dotenv(dotenv_path=resolved_env_path, override=False)
        logger.info(f"Loaded environment variables from: {resolved_env_path}")
    else:
        logger.warning(f".env file not found at: {resolved_env_path}. Relying on environment variables or defaults.")

    # Load base YAML configuration
    if resolved_config_path.exists():
        with open(resolved_config_path, 'r') as f:
            try:
                config_data = yaml.safe_load(f)
                logger.info(f"Loaded configuration from: {resolved_config_path}")
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML file {resolved_config_path}: {e}")
                raise
    else:
        logger.error(f"Configuration file not found at: {resolved_config_path}. Cannot proceed.")
        raise FileNotFoundError(f"Configuration file not found: {resolved_config_path}")

    # --- Environment Variable Overrides ---
    # Allow overriding specific config values via environment variables
    # Example: Set LLM_DEFAULT_PROVIDER=anthropic in .env or environment
    if llm_provider_override := os.getenv('LLM_DEFAULT_PROVIDER'):
        if 'llm' in config_data:
            config_data['llm']['default_provider'] = llm_provider_override
            logger.info(f"Overrode llm.default_provider with env var: {llm_provider_override}")

    if llm_model_override := os.getenv('LLM_DEFAULT_MODEL'):
        if 'llm' in config_data:
            config_data['llm']['default_model'] = llm_model_override
            logger.info(f"Overrode llm.default_model with env var: {llm_model_override}")

    if search_provider_override := os.getenv('SEARCH_DEFAULT_PROVIDER'):
        if 'scraping' in config_data and 'search' in config_data['scraping']:
            config_data['scraping']['search']['default_provider'] = search_provider_override
            logger.info(f"Overrode scraping.search.default_provider with env var: {search_provider_override}")

    # --- API Key Loading (from .env or environment) ---
    # Load API keys into the config dict for easier access, but prioritize environment variables
    config_data.setdefault('api_keys', {})
    config_data['api_keys']['openai'] = os.getenv('OPENAI_API_KEY')
    config_data['api_keys']['anthropic'] = os.getenv('ANTHROPIC_API_KEY')
    config_data['api_keys']['google'] = os.getenv('GOOGLE_API_KEY') # Used for both LLM and potentially Search
    config_data['api_keys']['google_cse_id'] = os.getenv('GOOGLE_CSE_ID')
    config_data['api_keys']['bing'] = os.getenv('BING_SEARCH_V7_SUBSCRIPTION_KEY')

    # Check if essential API keys are missing for the configured default providers
    # (We might refine this check later based on actual usage)

    _config = config_data
    return _config

def get_config() -> Dict[str, Any]:
    """Returns the loaded configuration dictionary.

    Raises:
        RuntimeError: If load_config() has not been called first.
    """
    if _config is None:
        # Try loading with defaults if not loaded explicitly
        logger.warning("Configuration accessed before explicit loading. Attempting load with default paths.")
        return load_config()
        # raise RuntimeError("Configuration has not been loaded. Call load_config() first.")
    return _config

# Example usage (typically called once at application startup)
if __name__ == '__main__':
    # Configure logging minimally for this example
    logging.basicConfig(level=logging.INFO)

    # Example: Load config explicitly (e.g., in your main app entry point)
    try:
        config = load_config()
        print("Configuration loaded successfully:")
        # print(yaml.dump(config, indent=2))

        # Example: Accessing config values
        print(f"\nDefault LLM Provider: {config.get('llm', {}).get('default_provider')}")
        print(f"OpenAI API Key Loaded: {'Yes' if config.get('api_keys', {}).get('openai') else 'No'}")
        print(f"Log file path: {config.get('logging', {}).get('log_file')}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Example: Accessing config later using get_config()
    # try:
    #     retrieved_config = get_config()
    #     print("\nRetrieved config using get_config():")
    #     print(f"Headless browser: {retrieved_config.get('scraping', {}).get('browser', {}).get('headless')}")
    # except RuntimeError as e:
    #     print(e) 