import streamlit as st
import pandas as pd
import os
import json
import yaml # For potential YAML support later
import logging
import threading # For running orchestrator in background
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from io import BytesIO
import queue # For thread-safe communication
import time
import sys # Import sys
import asyncio # Import asyncio
import platform # Need platform to check OS

# --- Path Setup (Add src to sys.path) ---
# Calculate the project root directory (assuming app.py is in src/intelliscrape_studio/)
_project_root = Path(__file__).resolve().parent.parent.parent 
_src_dir = _project_root / "src"

# Add src directory to the Python path if it's not already there
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
    # Add a print statement for debugging purposes when running streamlit
    print(f"[DEBUG] Added {_src_dir} to sys.path", file=sys.stderr)

# --- Page Config (Must be the first Streamlit command) ---
st.set_page_config(page_title="IntelliScrape Studio", layout="wide")

# Configure logging basic setup immediately in case imports fail
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Core Application Imports (Using Absolute Imports) ---
try:
    from intelliscrape_studio.core.config_loader import load_config, get_config, BASE_DIR
    from intelliscrape_studio.core.logging_setup import setup_logging
    from intelliscrape_studio.core.orchestrator import Orchestrator
    from intelliscrape_studio.data.models import DataTableSchema, ColumnDefinition, TaskState
    from intelliscrape_studio.data.handler import DataHandler # Import needed for orchestrator data access
    # Add LLM Provider imports for API Key handling
    from intelliscrape_studio.llm.providers.factory import LLMProviderFactory 
    # Import Searcher or Factory to get available engines
    from intelliscrape_studio.scraping.search import Searcher, _SearchEngineFactory # Use internal factory for now
except ImportError as e:
    # Enhance error message for absolute import failures
    logger.error(f"Error importing core modules using absolute path: {e}. Please ensure the 'src' directory is in Python's path or run from the project root.", exc_info=True)
    st.error(f"错误：无法导入核心模块: {e}。请确保项目结构正确，并尝试从项目根目录运行 `streamlit run src/intelliscrape_studio/app.py`。检查 `src` 目录是否在 PYTHONPATH 中。")
    # Define dummy classes/functions to allow app to load partially
    class Orchestrator: 
        def __init__(self, *args, **kwargs): pass
        def start_task(self, *args, **kwargs): pass
        def stop_task(self, *args, **kwargs): pass
        def get_current_data(self, *args, **kwargs): return []
        data_handler: Any = None
        
    class DataTableSchema: pass
    class ColumnDefinition: pass
    class TaskState: pass
    class DataHandler:
        def __init__(self, *args, **kwargs): pass
        def get_dataframe(self, *args, **kwargs): return pd.DataFrame()
        def export_to_csv(self, *args, **kwargs): return b""
        def export_to_excel(self, *args, **kwargs): return b""
    # Add dummy for LLMProviderFactory
    class LLMProviderFactory:
        @classmethod
        def get_available_providers(cls): return ["openai", "dummy"] # Example
    # Add dummy for Searcher/Factory
    class Searcher:
        @staticmethod
        def get_available_engines(): return ["duckduckgo", "dummy"] # Example
    class _SearchEngineFactory:
        @staticmethod
        def get_available_engines(): return ["duckduckgo", "dummy"]

    def load_config(): return {}
    def setup_logging(): pass
    def get_config(): return {}
    BASE_DIR = Path(".")

# --- Constants & Global Config ---
# Load config early
try:
    APP_CONFIG = load_config()
    # Setup logging based on loaded config
    setup_logging()
    logger.info("Application configuration loaded and logging setup complete.")
except FileNotFoundError as e:
    st.error(f"错误: 配置文件 'config/settings.yaml' 未找到: {e}")
    APP_CONFIG = {} # Use empty config to avoid crashing UI
except Exception as e:
    st.error(f"加载配置时出错: {e}")
    logger.error(f"Error loading configuration: {e}", exc_info=True)
    APP_CONFIG = {}

# Get relevant config sections
UI_CONFIG = APP_CONFIG.get('ui', {})
DATA_CONFIG = APP_CONFIG.get('data', {})
LLM_CONFIG = APP_CONFIG.get('llm', {})
SCRAPING_CONFIG = APP_CONFIG.get('scraping', {})

# Define directories using config
DEFAULT_TEMPLATE_DIR = BASE_DIR / DATA_CONFIG.get('save_template_dir', 'templates')
DEFAULT_EXPORT_DIR = BASE_DIR / DATA_CONFIG.get('default_export_dir', 'results')

# Ensure directories exist
os.makedirs(DEFAULT_TEMPLATE_DIR, exist_ok=True)
os.makedirs(DEFAULT_EXPORT_DIR, exist_ok=True)

# Define default schema using ColumnDefinition fields
DEFAULT_SCHEMA_DATA = [
    {'name': 'Company Name', 'dtype': 'Text', 'description': 'The official name of the company'},
    {'name': 'Website', 'dtype': 'URL', 'description': 'The main company website URL'},
    {'name': 'Description', 'dtype': 'Text', 'description': 'A brief description of the company'},
]
# Check if ColumnDefinition was imported correctly before accessing fields
if hasattr(ColumnDefinition, '__fields__') and 'dtype' in ColumnDefinition.__fields__:
    # Ensure the field actually holds Literal types before accessing __args__
    try:
        COLUMN_TYPES = list(ColumnDefinition.__fields__['dtype'].type_.__args__)
    except AttributeError:
        logger.warning("Could not dynamically determine COLUMN_TYPES from ColumnDefinition. Using fallback.")
        COLUMN_TYPES = ['Text', 'Number', 'Date', 'URL', 'Email', 'Boolean', 'Other']
else:
    COLUMN_TYPES = ['Text', 'Number', 'Date', 'URL', 'Email', 'Boolean', 'Other'] # Fallback

# Queue for thread-safe communication between Orchestrator thread and Streamlit
update_queue = queue.Queue()

# --- Helper Functions ---
def df_to_schema(df: pd.DataFrame) -> Optional[DataTableSchema]:
    """Converts a DataFrame from st.data_editor to a DataTableSchema Pydantic model."""
    columns = []
    required_cols = {'name', 'dtype', 'description'}
    if not required_cols.issubset(df.columns):
         st.error(f"Schema DataFrame 缺少必需列. 需要: {required_cols}. 找到: {list(df.columns)}")
         return None
    try:
        for _, row in df.iterrows():
            if not row['name'] or pd.isna(row['name']): 
                 st.warning("跳过缺少列名的行。")
                 continue
            # Ensure dtype is valid, default to Text if not
            dtype_val = row['dtype'] if row['dtype'] in COLUMN_TYPES else 'Text'
            if row['dtype'] not in COLUMN_TYPES:
                 st.warning(f"列 '{row['name']}' 的数据类型 '{row['dtype']}' 无效. 使用 'Text' 代替.")

            col_def = ColumnDefinition(
                name=str(row['name']).strip(),
                dtype=dtype_val,
                description=str(row.get('description', '')).strip()
            )
            columns.append(col_def)
        if not columns:
            st.error("无法从 DataFrame 创建有效的列定义。请确保至少有一行包含有效的列名。")
            return None
        return DataTableSchema(columns=columns)
    except Exception as e:
        st.error(f"将 DataFrame 转换为 Schema 时出错: {e}")
        logger.error(f"Error converting DataFrame to Schema: {e}", exc_info=True)
        return None

def schema_to_df(schema: DataTableSchema) -> pd.DataFrame:
    """Converts a DataTableSchema Pydantic model to a DataFrame for st.data_editor."""
    data = [{'name': col.name, 'dtype': col.dtype, 'description': col.description}
            for col in schema.columns]
    return pd.DataFrame(data)

def load_template(uploaded_file) -> Optional[DataTableSchema]:
    """Loads a schema template from an uploaded JSON file."""
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.json'):
                schema_dict = json.load(uploaded_file)
            else:
                 st.error("不支持的文件类型。请上传 JSON 文件。")
                 return None
                
            if 'columns' not in schema_dict or not isinstance(schema_dict['columns'], list):
                st.error("模板文件格式无效：缺少 'columns' 列表。")
                return None
            
            # Use Pydantic for validation
            schema = DataTableSchema.parse_obj(schema_dict)
            logger.info(f"Schema template '{uploaded_file.name}' loaded successfully.")
            st.success(f"模板 '{uploaded_file.name}' 已加载！")
            return schema
        except json.JSONDecodeError as e:
            st.error(f"解析 JSON 模板时出错: {e}")
            logger.error(f"Error parsing JSON template '{uploaded_file.name}': {e}", exc_info=True)
            return None
        except Exception as e:
            st.error(f"加载模板时出错: {e}")
            logger.error(f"Error loading template '{uploaded_file.name}': {e}", exc_info=True)
            return None
    return None

def save_template(schema: DataTableSchema, filename: str):
    """Saves the current schema as a JSON template file."""
    if not filename:
        st.error("请输入模板文件名。")
        return
    filename = filename.strip()
    if not filename.endswith('.json'):
        filename += '.json'
    filename = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in filename)
    
    save_path = DEFAULT_TEMPLATE_DIR / filename
    try:
        schema_dict = schema.dict()
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(schema_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Schema template saved to: {save_path}")
        st.success(f"模板已保存为: {filename}")
    except Exception as e:
        st.error(f"保存模板时出错: {e}")
        logger.error(f"Error saving template to {save_path}: {e}", exc_info=True)

# --- Function to get available models ---
# Place this function before main()
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_dynamic_models(provider_name: str, api_key: Optional[str], base_url: Optional[str]) -> List[str]:
    """Attempts to dynamically fetch available models for a provider."""
    if not api_key:
        logger.warning(f"Cannot fetch models for {provider_name}: API key missing.")
        return [] # Return empty if no API key
    
    # Default model lists as fallback
    default_models = {
        "openai": sorted([ 
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-turbo-preview", "gpt-4", "gpt-4-0613",
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0125"
        ]),
        "anthropic": sorted(["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-2.1", "claude-2.0", "claude-instant-1.2"]),
        "gemini": sorted(["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-pro"])
        # Add other providers if needed
    }
    fallback_list = default_models.get(provider_name, [])

    try:
        logger.info(f"Attempting to dynamically load models for provider: {provider_name}")
        # Temporarily create provider instance just to get models
        # Pass necessary kwargs for initialization (api_key, base_url if applicable)
        provider_instance = LLMProviderFactory.create_provider(
            provider_name=provider_name,
            api_key=api_key,
            base_url=base_url # Pass base_url if available
            # Add other necessary kwargs if providers need them
        )
        if hasattr(provider_instance, 'get_available_models') and callable(provider_instance.get_available_models):
            models = provider_instance.get_available_models()
            if models:
                logger.info(f"Successfully loaded {len(models)} models for {provider_name}.")
                return models
            else:
                logger.warning(f"Provider {provider_name} returned empty model list. Using fallback.")
        else:
            logger.warning(f"Provider {provider_name} does not implement get_available_models(). Using fallback.")
            
    except ImportError as e:
         logger.error(f"Failed to load models for {provider_name}: Missing dependency - {e}. Using fallback.")
    except Exception as e:
         logger.error(f"Failed to dynamically load models for {provider_name}: {e}. Using fallback.", exc_info=True)
         # Potentially show error to user? For now, just use fallback.
         # st.sidebar.warning(f"无法动态加载 {provider_name} 模型: {e}")

    return fallback_list

# --- Orchestrator Communication & UI Updates ---

def orchestrator_thread_target(orchestrator: Orchestrator, 
                               task_desc: str, 
                               schema: DataTableSchema, 
                               start_pts: List[str]):
    """Target function for the background thread running the orchestrator."""
    # Now run the orchestrator task (which is now synchronous)
    try:
        orchestrator.start_task(task_desc, schema, start_pts)
    except Exception as e:
        # Log critical error from the thread
        logger.critical(f"Orchestrator thread encountered a critical error: {e}", exc_info=True)
        # Send error state back to main thread via queue
        update_queue.put({"type": "error", "message": f"Orchestrator thread error: {e}"})
    finally:
        # Signal completion (or failure)
        update_queue.put({"type": "finished"})

def process_update_queue():
    """Process updates from the orchestrator thread queue in the main Streamlit thread."""
    needs_rerun = False
    processed_count = 0 # Added counter
    logger.debug("[QueueProc] Starting processing cycle...") # Modified log
    if update_queue.empty():
         logger.debug("[QueueProc] Queue is empty. No items to process.") # Modified log
         return False # No need to rerun if queue was empty
    else:
        logger.debug(f"[QueueProc] Queue size approx: {update_queue.qsize()}") # Modified log
        
    while not update_queue.empty():
        logger.debug(f"[QueueProc] Attempting to get item {processed_count + 1} from queue.") # Added log
        try:
            update = update_queue.get_nowait()
            processed_count += 1
            update_type = update.get('type', 'unknown')
            logger.debug(f"[QueueProc] Got update #{processed_count}: Type={update_type}") # Modified log
            needs_rerun = True # Assume most updates require a rerun, can be overridden below
            
            if update_type == "log":
                level, message = update["payload"]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"{timestamp} [{level.upper()}] {message}"
                logger.debug(f"[QueueProc] Adding log entry: {log_entry[:100]}...") # Added log
                # Limit log length
                # Prepend new logs instead of appending for newest-first view without reversing whole list
                st.session_state.logs.insert(0, log_entry)
                if len(st.session_state.logs) > 200: # Limit size
                    st.session_state.logs = st.session_state.logs[:200]
                
                # Rerun less frequently for logs to avoid excessive updates
                # Use a counter or time-based check
                if time.time() - st.session_state.get('last_log_rerun_time', 0) < 0.5: 
                     logger.debug("[QueueProc] Log received, but suppressing immediate rerun due to frequency limit.")
                     needs_rerun = False # Don't rerun yet if last log was very recent
                else:
                    logger.debug("[QueueProc] Log received, triggering rerun.")
                    st.session_state.last_log_rerun_time = time.time()

            elif update_type == "state":
                new_state: TaskState = update["payload"]
                logger.debug(f"[QueueProc] Updating session state. New status: {new_state.status}, Pages: {new_state.page_count}, URL: {new_state.current_url}") # Added log
                st.session_state.task_state = new_state
                # Update results dataframe if orchestrator and handler exist
                # This might be problematic if orchestrator state is needed and thread ended
                # Consider getting data directly from TaskState if possible or handle missing orchestrator
                orchestrator_instance = st.session_state.get('orchestrator')
                if orchestrator_instance and orchestrator_instance.data_handler:
                    try:
                        df = orchestrator_instance.data_handler.get_dataframe()
                        if not df.equals(st.session_state.get('results_df')):
                             st.session_state.results_df = df
                             logger.debug("[QueueProc] Updated results_df from DataHandler.")
                        else:
                             logger.debug("[QueueProc] results_df from DataHandler hasn't changed.")
                    except Exception as e:
                        logger.error(f"[QueueProc] Failed to get DataFrame from DataHandler: {e}")
                elif new_state.collected_data is not None: # Fallback: Try using data from TaskState if available
                    try:
                        # Convert list of dicts back to DataFrame if needed
                        # Assuming collected_data holds the raw records
                        new_df = pd.DataFrame(new_state.collected_data)
                        if not new_df.equals(st.session_state.get('results_df')):
                            st.session_state.results_df = new_df
                            logger.debug("[QueueProc] Updated results_df from TaskState.collected_data.")
                        else:
                             logger.debug("[QueueProc] results_df from TaskState hasn't changed.")
                    except Exception as e:
                         logger.error(f"[QueueProc] Failed to create DataFrame from TaskState.collected_data: {e}")
                else:
                    logger.warning("[QueueProc] Cannot update results_df: Orchestrator/DataHandler not found and no data in TaskState.")
                logger.debug(f"[QueueProc] State update processed, triggering rerun.") # Added log
            
            elif update_type == "error":
                error_message = update["message"]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"{timestamp} [CRITICAL] Orchestrator Error: {error_message}"
                logger.error(f"[QueueProc] Processing error update: {error_message}") # Added log
                st.session_state.logs.insert(0, log_entry)
                st.session_state.running = False
                if st.session_state.task_state:
                     st.session_state.task_state.status = 'error'
                     st.session_state.task_state.last_error = error_message
                logger.debug(f"[QueueProc] Error update processed, triggering rerun.") # Added log
                
            elif update_type == "finished":
                 logger.info("[QueueProc] Processing finished signal.") # Added log
                 st.session_state.running = False
                 # Final state update might have already been sent via 'state'
                 if st.session_state.task_state and st.session_state.task_state.status not in ['error', 'finished']:
                      st.session_state.task_state.status = 'finished'
                      logger.debug("[QueueProc] Marked task state as finished.")
                 else:
                     logger.debug("[QueueProc] Task state already marked finished or error.")
                 logger.debug(f"[QueueProc] Finished update processed, triggering rerun.") # Added log
                 
            else:
                 logger.warning(f"[QueueProc] Received unknown update type in queue: {update_type}")
                 # Still trigger rerun for unknown types? Maybe not.
                 needs_rerun = False 
                 
        except queue.Empty:
            logger.debug("[QueueProc] Queue became empty during processing loop.")
            break # Queue is empty, stop processing for now
        except Exception as e:
             logger.error(f"[QueueProc] Error processing update item type '{update_type}': {e}", exc_info=True)
             # Avoid crashing the app due to queue processing errors
             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
             st.session_state.logs.insert(0, f"{timestamp} [ERROR] Failed to process internal UI update: {e}")
             needs_rerun = True # Rerun to show the error log
             
    logger.debug(f"[QueueProc] Finished processing cycle. Processed {processed_count} items. Needs rerun: {needs_rerun}") # Added log
    return needs_rerun # Return whether a rerun is needed based on processed items


# --- Streamlit App ---
def main():
    # Note: set_page_config is now at the top level of the script
    st.title("📊 IntelliScrape Studio")
    st.caption("一个由LLM驱动的智能Web数据提取工具")

    # --- Initialize Session State ---
    if 'app_initialized' not in st.session_state:
        logger.info("Initializing Streamlit session state.")
        st.session_state.logs = ["欢迎使用 IntelliScrape Studio! 日志将显示在这里..."]
        st.session_state.results_df = pd.DataFrame()
        st.session_state.running = False
        st.session_state.task_state = None
        st.session_state.schema_df = pd.DataFrame(DEFAULT_SCHEMA_DATA)
        # Initialize search engine and timeout from config
        st.session_state.search_engine_input = SCRAPING_CONFIG.get('search', {}).get('default_provider', 'duckduckgo')
        st.session_state.search_timeout_input = SCRAPING_CONFIG.get('search', {}).get('timeout', 20)
        st.session_state.proxy_url_input = SCRAPING_CONFIG.get('proxy', '') # Already added?
        st.session_state.selected_model = LLM_CONFIG.get('default_model', 'gpt-4o-mini') # Default from config
        st.session_state.app_initialized = True
        # Initialize API key from env var if not set and env var exists
        provider = st.session_state.get('llm_provider_display', LLM_CONFIG.get('default_provider', 'openai'))
        api_key_env = LLM_CONFIG.get('api_keys_env_vars', {}).get(provider)
        session_key = f"{provider}_api_key"
        if session_key not in st.session_state and api_key_env and os.getenv(api_key_env):
             st.session_state[session_key] = os.getenv(api_key_env)
             logger.info(f"Initialized API key for {provider} from environment variable {api_key_env}")
        # Initialize base URL from session state if not already set
        if 'openai_base_url' not in st.session_state:
             st.session_state.openai_base_url = ""

    # --- Process background updates --- 
    # Check the queue on every run and trigger a rerun if needed
    # Add logging here to see if process_update_queue is called
    logger.debug("[App] Checking update queue...") 
    needs_rerun_flag = process_update_queue()
    logger.debug(f"[App] process_update_queue returned: {needs_rerun_flag}") # Added log
    if needs_rerun_flag:
         # Add logging here to see if rerun is triggered
         logger.debug("[App] Rerunning due to queue update.")
         st.rerun()
    else:
         logger.debug("[App] No rerun triggered.") # Added log

    # --- Callbacks for Orchestrator (put data into the queue) ---
    def update_ui_state_callback(state: TaskState):
        # Add logging here
        logger.debug(f"[Callback] Received state update: Status={state.status}, Pages={state.page_count}")
        try:
            update_queue.put({"type": "state", "payload": state})
            logger.debug("[Callback] State update successfully put into queue.") # Modified log
        except Exception as e:
             logger.error(f"[Callback] Failed to put state update into queue: {e}")

    def update_ui_log_callback(level: str, message: str):
        # Add logging here
        logger.debug(f"[Callback] Received log: {level} - {message[:50]}...")
        # Handle special internal message
        if level == "internal" and message == "__THREAD_FINISHED__":
             try:
                 update_queue.put({"type": "finished"})
                 logger.debug("[Callback] __THREAD_FINISHED__ signal put into queue.")
             except Exception as e:
                 logger.error(f"[Callback] Failed to put __THREAD_FINISHED__ signal into queue: {e}")
             return # Don't process as a normal log message
             
        try:
            update_queue.put({"type": "log", "payload": (level, message)})
            logger.debug("[Callback] Log update successfully put into queue.") # Modified log
        except Exception as e:
            logger.error(f"[Callback] Failed to put log update into queue: {e}")

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("⚙️ 配置")

        # --- LLM Configuration ---
        st.subheader("LLM 设置")
        # Select LLM Provider (read-only for now, based on config)
        available_providers = LLMProviderFactory.get_available_providers()
        # Ensure default_provider from config is valid, else fallback
        default_provider = LLM_CONFIG.get('default_provider', 'openai')
        if default_provider not in available_providers:
             st.warning(f"Configured default provider '{default_provider}' is not available. Falling back to '{available_providers[0]}'.")
             default_provider = available_providers[0]
             # Persist this fallback decision? For now, just use it for UI display.
        st.selectbox("LLM Provider", available_providers, index=available_providers.index(default_provider), key='llm_provider_display', disabled=True, help="Provider is currently set in config/settings.yaml")

        # Get current API key and base URL from session state for model fetching
        current_api_key = st.session_state.get(f"{default_provider}_api_key")
        current_base_url = st.session_state.get("openai_base_url") if default_provider == 'openai' else None

        # Get available models (cached)
        available_models = get_dynamic_models(default_provider, current_api_key, current_base_url)

        # Model Selection Dropdown
        default_model_from_config = LLM_CONFIG.get('default_model', 'gpt-4o-mini')
        # Use session state value if available, otherwise use config default
        current_selection = st.session_state.get('selected_model', default_model_from_config)
        
        # Ensure current selection is valid within the available models, else use default or first available
        final_model_selection = default_model_from_config
        if current_selection in available_models:
            final_model_selection = current_selection
        elif default_model_from_config in available_models:
             final_model_selection = default_model_from_config
        elif available_models:
             final_model_selection = available_models[0]
             
        selected_model = st.selectbox(
            "模型",
            options=available_models,
            index=available_models.index(final_model_selection) if final_model_selection in available_models else 0,
            key="selected_model",
            help="选择要使用的 LLM 模型。列表会尝试动态加载，如果失败则显示默认列表。",
            disabled=not available_models # Disable if list is empty
        )

        # API Key Input
        api_key = st.text_input(
            "API Key",
            key=f"{default_provider}_api_key",
            value=st.session_state.get(f"{default_provider}_api_key", ""),
            help=f"输入 {default_provider} 的 API Key。"
        )

        # Base URL Input (only for OpenAI)
        if default_provider == 'openai':
            base_url = st.text_input(
                "Base URL",
                key="openai_base_url",
                value=st.session_state.get("openai_base_url", ""),
                help="输入 OpenAI 的 Base URL（可选）。"
            )

        # --- Scraping Configuration ---
        st.subheader("爬取设置")
        
        # Proxy Input
        proxy_url = st.text_input(
            "代理 URL (可选)",
            key="proxy_url_input",
            value=st.session_state.get('proxy_url_input', SCRAPING_CONFIG.get('proxy', '')), # Initialize from config
            placeholder="例如: http://user:pass@host:port",
            help="用于浏览器和搜索请求的代理服务器 (例如 HTTP, SOCKS5)。"
        )
        
        # Search Engine Selection
        try:
            available_engines = _SearchEngineFactory.get_available_engines()
        except Exception as e:
             logger.warning(f"Could not get available search engines: {e}")
             available_engines = ['duckduckgo'] # Fallback
        
        default_search_engine = st.session_state.get('search_engine_input', SCRAPING_CONFIG.get('search', {}).get('default_provider', 'duckduckgo'))
        # Ensure default is valid
        if default_search_engine not in available_engines:
            default_search_engine = available_engines[0] if available_engines else 'duckduckgo'
        
        search_engine = st.selectbox(
            "搜索引擎",
            available_engines,
            index=available_engines.index(default_search_engine) if default_search_engine in available_engines else 0,
            key="search_engine_input",
            help="选择用于 'Search' 操作的搜索引擎。"
        )
        
        # Search Timeout Input
        search_timeout = st.number_input(
            "搜索超时 (秒)",
            min_value=5,
            max_value=120,
            step=5,
            key="search_timeout_input",
            help="搜索请求等待响应的最长时间。"
        )
        
        # Browser Settings (Headless, Timeout)
        headless_mode = st.checkbox("无头模式", value=st.session_state.get('headless_mode', SCRAPING_CONFIG.get('browser', {}).get('headless', True)), key='headless_mode', help="在没有浏览器窗口的情况下运行浏览器。")
        browser_timeout = st.number_input("浏览器超时 (毫秒)", min_value=5000, max_value=300000, value=st.session_state.get('browser_timeout', SCRAPING_CONFIG.get('browser', {}).get('timeout', 60000)), step=1000, key='browser_timeout', help="浏览器操作（例如页面加载）的最长时间。")
        
        # Task Limits (Max Pages, Max Time)
        max_pages_val = st.session_state.get('max_pages_input', SCRAPING_CONFIG.get('max_pages_per_task', 50))
        max_time_val = st.session_state.get('max_time_input', SCRAPING_CONFIG.get('max_time_per_task', 300))
        start_points_val = st.session_state.get('start_points_input', '')

        max_pages = st.number_input("最大浏览页面数", min_value=1, max_value=500, 
                                     value=max_pages_val, step=1,
                                     key='max_pages_input',
                                     help="单个任务最多访问的页面数量。")
        max_time = st.number_input("最大运行时间 (秒)", min_value=0, max_value=3600, 
                                   value=max_time_val, step=30,
                                   key='max_time_input',
                                   help="单个任务的最大运行时间（0表示不限制）。")
        start_points_input = st.text_area("起始点 (可选)", height=75,
                                         value=start_points_val,
                                         placeholder="输入起始 URL 或搜索关键词 (每行一个)",
                                         key='start_points_input',
                                         help="提供任务开始的 URL 或搜索词。")

        # --- Data Schema Definition ---
        st.subheader("数据模式")
        st.caption("定义你想要提取的数据结构。")
        
        # Schema Template Load/Save
        col_load, col_save, col_fname = st.columns([2, 2, 3])
        with col_load:
            uploaded_template = st.file_uploader("加载模板 (.json)", type=['json'], key='upload_template')
            if uploaded_template:
                # Process upload only once
                if uploaded_template.name != st.session_state.get('_last_uploaded_template_name'):
                     loaded_schema = load_template(uploaded_template)
                     if loaded_schema:
                         st.session_state.schema_df = schema_to_df(loaded_schema)
                         st.session_state._last_uploaded_template_name = uploaded_template.name
                         st.rerun()
                     else:
                          st.session_state._last_uploaded_template_name = None # Reset if load failed
                
        with col_save:
            save_button = st.button("保存模板", key='save_template_button')
        with col_fname:
             template_filename = st.text_input("模板文件名", placeholder="my_template.json", 
                                             key='template_filename_input', label_visibility="collapsed")
             
        if save_button:
            if template_filename:
                 current_schema = df_to_schema(st.session_state.schema_df)
                 if current_schema:
                     save_template(current_schema, template_filename)
                 else:
                      st.warning("无法保存：当前数据模式无效。")
            else:
                st.warning("请输入要保存的模板文件名。")

        # Schema Editor
        edited_schema_df = st.data_editor(
            st.session_state.schema_df,
            num_rows="dynamic",
            column_config={
                "name": st.column_config.TextColumn("列名 (Name)", required=True, help="数据列的名称"),
                "dtype": st.column_config.SelectboxColumn("数据类型 (Type)", required=True, options=COLUMN_TYPES, help="期望的数据格式"),
                "description": st.column_config.TextColumn("描述/提示 (Description)", required=True, help="给LLM的提示，说明要查找什么信息")
            },
            key="schema_editor",
            height=300
        )
        
        # Update session state only if the dataframe actually changed
        if not edited_schema_df.equals(st.session_state.schema_df):
            st.session_state.schema_df = edited_schema_df
            st.rerun()

    # --- Main Area ---
    st.header("🎯 任务定义")
    task_description = st.text_area("输入你的数据收集任务描述:", height=100, 
                                    value=st.session_state.get('task_description_input', ''),
                                    key='task_description_input',
                                    placeholder="例如：查找硅谷地区 A 轮融资后的人工智能初创公司信息，包括公司名称、网站和 CEO。")

    # --- Execution Control ---
    col_start, col_stop, col_progress = st.columns([1, 1, 4])
    with col_start:
        start_button = st.button("🚀 开始执行", type="primary", disabled=st.session_state.running)
    with col_stop:
        stop_button = st.button("⏹️ 停止执行", disabled=not st.session_state.running)

    # --- Progress Bar & Status Display ---
    status_container = st.container()
    progress_bar_container = st.container()

    # --- Execution Logic ---
    if start_button and not st.session_state.get('running', False):
        # Read values from widgets stored in session state
        task_desc_val = st.session_state.task_description_input
        start_pts_val = st.session_state.start_points_input
        
        # Get API key using the correct provider and dynamic session state key
        provider = st.session_state.llm_provider_display # Get current provider from sidebar widget state
        session_key = f"{provider}_api_key"
        # Prioritize session state (UI input), then config file, then env var (handled by config loader)
        api_key_val = st.session_state.get(session_key) or LLM_CONFIG.get('api_keys', {}).get(provider) 
        
        # Validate inputs
        if not api_key_val:
            st.error(f"请输入 {provider.capitalize()} API 密钥或在配置/环境变量中设置它。")
        elif not task_desc_val:
            st.error("请输入任务描述。")
        else:
            current_schema_df = st.session_state.schema_df
            target_schema = df_to_schema(current_schema_df)
            if target_schema and target_schema.columns:
                st.session_state.running = True
                st.session_state.logs = ["任务初始化中..."]
                st.session_state.results_df = pd.DataFrame(columns=[col.name for col in target_schema.columns])
                st.session_state.task_state = None 
                
                # Clear the queue before starting
                while not update_queue.empty(): 
                    try: update_queue.get_nowait() 
                    except queue.Empty: break

                # Prepare start points
                start_points = [p.strip() for p in start_pts_val.split('\n') if p.strip()]
                
                # --- Initialize Orchestrator HERE ---
                try:
                    # --- Inject UI settings into config before initializing ---
                    current_config = get_config() 
                    llm_config = current_config.get('llm', {}).copy()
                    scraping_config = current_config.get('scraping', {}).copy()
                    
                    # Update LLM config with API key from validated value
                    if 'api_keys' not in llm_config: llm_config['api_keys'] = {}
                    llm_config['api_keys'][provider] = api_key_val
                    
                    # Update LLM config with Base URL from UI/session_state if provider is openai
                    if provider == 'openai':
                         ui_base_url = st.session_state.get("openai_base_url")
                         if ui_base_url and ui_base_url.strip():
                              llm_config['base_url'] = ui_base_url.strip()
                              logger.info(f"Using custom OpenAI Base URL from UI: {llm_config['base_url']}")
                         elif 'base_url' in llm_config:
                             # Remove base_url from config if UI input is empty but key exists in config
                             del llm_config['base_url'] 

                    # Update LLM config with model from selected model
                    selected_model_name = st.session_state.get("selected_model")
                    if selected_model_name:
                        llm_config['default_model'] = selected_model_name
                        logger.info(f"Using model selected from UI: {llm_config['default_model']}")

                    # Update Scraping config from UI/session_state
                    # Ensure 'search' sub-dictionary exists
                    if 'search' not in scraping_config: scraping_config['search'] = {}
                    # Update default provider and timeout for search
                    scraping_config['search']['default_provider'] = st.session_state.get('search_engine_input', scraping_config.get('search', {}).get('default_provider', 'duckduckgo'))
                    scraping_config['search']['timeout'] = st.session_state.get('search_timeout_input', scraping_config.get('search', {}).get('timeout', 20))
                    
                    # Ensure 'browser' sub-dictionary exists
                    if 'browser' not in scraping_config: scraping_config['browser'] = {}
                    scraping_config['browser']['headless'] = st.session_state.get('headless_mode', scraping_config.get('browser', {}).get('headless', True))
                    scraping_config['browser']['timeout'] = st.session_state.get('browser_timeout', scraping_config.get('browser', {}).get('timeout', 60000))
                    
                    # Update scraping limits
                    scraping_config['max_pages_per_task'] = st.session_state.get('max_pages_input', scraping_config.get('max_pages_per_task', 50))
                    scraping_config['max_time_per_task'] = st.session_state.get('max_time_input', scraping_config.get('max_time_per_task', 300))
                    
                    # --- Add/Update Proxy Setting --- 
                    ui_proxy_url = st.session_state.get("proxy_url_input")
                    # Ensure 'search' sub-dictionary exists before adding proxy to it
                    if 'search' not in scraping_config: scraping_config['search'] = {}
                    if ui_proxy_url and ui_proxy_url.strip():
                        # Apply proxy to the 'search' sub-config (used by Searcher)
                        scraping_config['search']['proxy'] = ui_proxy_url.strip()
                        logger.info(f"Using proxy from UI for search: {scraping_config['search']['proxy']}")
                        # Optionally, apply to browser config too if browser needs it
                        # scraping_config.setdefault('browser', {})['proxy'] = {
                        #     'server': ui_proxy_url.strip()
                        # }
                    elif 'proxy' in scraping_config['search']:
                        # Remove proxy from 'search' sub-config if UI input is empty
                        del scraping_config['search']['proxy']
                        logger.info("Proxy input is empty, removing from scraping search config.")
                    # --- End Proxy Setting Update ---

                    logger.info("Initializing Orchestrator with current UI settings...")
                    st.session_state.orchestrator = Orchestrator(
                        llm_config=llm_config, # Pass updated LLM config (includes model)
                        scraping_config=scraping_config,
                        state_update_callback=update_ui_state_callback,
                        log_callback=update_ui_log_callback
                    )
                    logger.info("Orchestrator instance created successfully.")
                     
                    # Start orchestrator in a background thread
                    thread = threading.Thread(
                        target=orchestrator_thread_target,
                        args=(
                            st.session_state.orchestrator, # Pass the newly created instance
                            task_desc_val, 
                            target_schema,
                            start_points
                        ),
                        daemon=True
                    )
                    thread.start()
                    logger.info("Orchestrator thread started.")
                    st.rerun() # Update UI immediately to show running state

                except Exception as e:
                    st.error(f"启动任务时出错: {e}")
                    logger.error(f"Failed to create or start Orchestrator: {e}", exc_info=True)
                    st.session_state.running = False
                    st.session_state.orchestrator = None # Ensure it's reset on failure
                    st.rerun()
            else:
                st.error("无法启动任务：数据模式无效或为空。请检查侧边栏中的定义。")

    if stop_button and st.session_state.get('running', False):
        logger.info("Stop button clicked.")
        orchestrator_instance = st.session_state.get('orchestrator')
        if orchestrator_instance:
             try:
                 # We still need to address the async nature of stop_task/cleanup
                 logger.info("Requesting task stop...")
                 orchestrator_instance.stop_task() # This call likely needs modification
                 st.info("已发送停止请求... 正在清理资源，请稍候。")
                 # Note: Cleanup might happen in the background thread upon stop signal
                 # We might not need explicit async handling here if stop_task just signals
                 # But the RuntimeWarning indicates a direct async call needs fixing somewhere
             except Exception as e:
                 st.error(f"发送停止请求时出错: {e}")
                 logger.error(f"Error calling stop_task: {e}", exc_info=True)
                 # Force stop UI state even if stop call failed
                 st.session_state.running = False 
                 st.rerun()
        else:
             st.warning("Orchestrator 实例未找到，无法停止。")
             st.session_state.running = False
             st.rerun()

    # --- Display Status (Reads from session state) ---
    current_task_state = st.session_state.get('task_state')
    running_status = st.session_state.get('running', False)

    with status_container:
         if current_task_state:
             state = current_task_state
             status_line = f"**状态:** `{state.status}` | **当前:** `{state.current_url or 'N/A'}` | **页面:** `{state.page_count}` | **错误:** `{state.error_count}`"
             st.markdown(status_line)
             if state.last_error:
                 st.warning(f"**最后错误:** {state.last_error}")
         elif running_status:
             st.info("任务正在初始化或运行中...")
             
    # Update progress bar based on state (simplified example)
    with progress_bar_container:
        if current_task_state:
            max_p_cfg = st.session_state.get('max_pages_input', SCRAPING_CONFIG.get('max_pages_per_task', 50))
            progress_val = min(float(current_task_state.page_count) / max_p_cfg, 1.0) if max_p_cfg > 0 else 0.0
            progress_text = f"状态: {current_task_state.status} (页面 {current_task_state.page_count})"
            st.progress(progress_val, text=progress_text)
        elif running_status:
             st.progress(0.0, text="任务正在初始化...")
        else:
             st.progress(0.0, text="等待开始...")

    # --- Display Logs ---
    with st.expander("📄 查看日志", expanded=True):
        # Display logs in reverse order (newest first)
        # Use insert(0, ...) in process_update_queue to avoid reversing here
        log_text = "\n".join(st.session_state.logs) # Display logs as they are (newest first)
        st.text_area("", value=log_text, height=300, key="log_display", disabled=True)

    # --- Display Results ---
    st.header("📊 结果")
    results_placeholder = st.empty() # Use placeholder for dynamic updates
    results_df = st.session_state.get('results_df', pd.DataFrame())
    
    if not results_df.empty:
        results_placeholder.dataframe(results_df)

        # --- Data Export ---
        col_csv, col_excel = st.columns(2)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_disabled = running_status # Disable export while running
        
        with col_csv:
            try:
                csv_bytes = results_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
            label="📥 下载结果 (CSV)",
                    data=csv_bytes,
                    file_name=f"intelliscrape_results_{timestamp}.csv",
            mime='text/csv',
                    key='download_csv',
                    disabled=export_disabled
                )
            except Exception as e:
                st.error(f"生成 CSV 时出错: {e}")
        
        with col_excel:
            try:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    results_df.to_excel(writer, index=False, sheet_name='Sheet1')
                excel_bytes = output.getvalue()
                st.download_button(
                    label="📥 下载结果 (Excel)",
                    data=excel_bytes,
                    file_name=f"intelliscrape_results_{timestamp}.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key='download_excel',
                    disabled=export_disabled
                )
            except Exception as e:
                 st.error(f"生成 Excel 时出错: {e}")

    elif running_status:
         results_placeholder.info("任务正在运行中...")
    else:
        # Show empty dataframe structure if schema is defined but no results yet
        if 'schema_df' in st.session_state and not st.session_state.schema_df.empty:
            schema_cols = st.session_state.schema_df['name'].tolist()
            if schema_cols:
                 results_placeholder.dataframe(pd.DataFrame(columns=schema_cols))
            else:
                 results_placeholder.info("请先在侧边栏定义数据模式。")
        else:
             results_placeholder.info("请先在侧边栏定义数据模式。")


if __name__ == "__main__":
    # NOTE: Ensure that Playwright browsers are installed if running for the first time
    # Run `playwright install` in your terminal if you encounter browser issues.
    main() 