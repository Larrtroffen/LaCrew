import logging
import time
import uuid
# import asyncio # Removed asyncio import
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

from ..data.models import DataTableSchema, TaskState, OrchestratorAction, Action, FinishAction
from ..llm.core import LLMCore # Assuming LLMCore class exists
from ..scraping.browser import Browser # Assuming Browser class exists
from ..scraping.search import Searcher # Assuming Searcher class exists
from ..data.handler import DataHandler # Assuming DataHandler class exists
from .config_loader import get_config

logger = logging.getLogger(__name__)

class Orchestrator:
    """Manages the entire data scraping workflow, coordinating LLM, browser, search, and data handling."""

    def __init__(self, 
                 llm_config: Optional[Dict[str, Any]] = None,
                 scraping_config: Optional[Dict[str, Any]] = None,
                 state_update_callback: Optional[Callable[[TaskState], None]] = None,
                 log_callback: Optional[Callable[[str, str], None]] = None):
        """Initializes the Orchestrator.

        Args:
            llm_config: Configuration dictionary for the LLM module. If None, loaded from file.
            scraping_config: Configuration dictionary for the Scraping module. If None, loaded from file.
            state_update_callback: A function to call with the updated TaskState after each step.
            log_callback: A function to call for sending log messages to the UI (level, message).
        """
        base_config = get_config() if llm_config is None or scraping_config is None else {}
        _llm_config = llm_config if llm_config is not None else base_config.get('llm', {})
        _scraping_config = scraping_config if scraping_config is not None else base_config.get('scraping', {})
        _api_keys = _llm_config.get('api_keys', base_config.get('api_keys', {}))

        self.llm_core = LLMCore(config=_llm_config)
        self.browser = Browser(config=_scraping_config.get('browser', {}))
        self.searcher = Searcher(config=_scraping_config.get('search', {}),
                                 api_keys=_api_keys)
        self.data_handler: Optional[DataHandler] = None # Initialized per task
        self.current_task_state: Optional[TaskState] = None
        self.update_callback = state_update_callback
        self.log_callback = log_callback
        self.stop_requested = False
        self.effective_scraping_config = _scraping_config
        self._log("INFO", "Orchestrator initialized.")

    def _log(self, level: str, message: str):
        """Log message to logger and optionally to UI via callback."""
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message)
        if self.log_callback:
            # Add logging to check if callback is called
            logger.debug(f"[_log] Calling log_callback for: {level} - {message[:50]}...")
            try:
                self.log_callback(level, message)
            except Exception as e:
                logger.error(f"Error in log callback: {e}")

    def _update_state(self, updates: Dict[str, Any]):
        """Update the current task state and notify via callback."""
        if self.current_task_state:
            # Add logging for state updates being applied
            logger.debug(f"[_update_state] Updating state with: {updates}")
            for key, value in updates.items():
                setattr(self.current_task_state, key, value)
            self.current_task_state.last_update_time = datetime.now() # Add timestamp
            if self.update_callback:
                # Add logging to check if callback is called
                logger.debug(f"[_update_state] Calling state_update_callback with status: {self.current_task_state.status}")
                try:
                    # Send a copy to prevent modification by the callback receiver
                    self.update_callback(self.current_task_state.copy(deep=True))
                except Exception as e:
                    logger.error(f"Error in state update callback: {e}")
        else:
            logger.warning("Attempted to update state, but no task is active.")

    def start_task(self, 
                   task_description: str, 
                   target_schema: DataTableSchema, 
                   start_points: Optional[List[str]] = None):
        """Starts a new scraping task."""
        if self.current_task_state and self.current_task_state.status not in ['finished', 'error']:
            self._log("ERROR", "Another task is already in progress.")
            raise RuntimeError("Another task is already in progress.")

        task_id = str(uuid.uuid4())
        self._log("INFO", f"Starting new task (ID: {task_id}): {task_description}")
        self.stop_requested = False
        self.data_handler = DataHandler(target_schema)

        self.current_task_state = TaskState(
            task_id=task_id,
            task_description=task_description,
            target_schema=target_schema,
            start_points=start_points or [],
            status='idle',
            start_time=datetime.now()
        )
        self._update_state({'status': 'initializing'}) # Initial state update

        # Start the main loop - run the async run_loop in the current thread's event loop
        # Since this start_task is called from a dedicated background thread in app.py,
        # asyncio.run() will create and manage the event loop for this thread.
        try:
            # Removed asyncio.run()
            self.run_loop()
        except Exception as e:
             # Catch potential errors from the loop itself
             self._log("CRITICAL", f"Run loop execution failed: {e}")
             logger.exception("Run loop crashed")
             if self.current_task_state:
                  self._update_state({'status': 'error', 'last_error': f"Critical error: {e}"})
             # Ensure cleanup is still called if run_loop fails before its finally block
             try:
                  # Removed asyncio.run()
                  self._cleanup()
             except Exception as cleanup_e:
                  logger.error(f"Error during fallback cleanup after run_loop crash: {cleanup_e}")

    def stop_task(self):
        """Requests the current task to stop gracefully."""
        if self.current_task_state and self.current_task_state.status not in ['finished', 'error']:
            self._log("INFO", "Stop requested for the current task.")
            self.stop_requested = True
        else:
            self._log("WARNING", "No active task to stop.")

    # Removed async
    def run_loop(self):
        """The main agentic loop driving the scraping process (now synchronous)."""
        if not self.current_task_state:
            self._log("ERROR", "Cannot run loop: No task initialized.")
            return
        self._log("DEBUG", "[RunLoop] Starting main loop.") # Added log
        try:
            max_pages = self.effective_scraping_config.get('max_pages_per_task', 50)
            max_time_s = self.effective_scraping_config.get('max_time_per_task', 300)
            start_time = self.current_task_state.start_time

            loop_count = 0 # Added counter
            while not self.stop_requested:
                loop_count += 1 # Increment counter
                self._log("DEBUG", f"[RunLoop] Iteration {loop_count} start. Checking stop conditions...") # Added log
                # 1. Check Stop Conditions
                if self.current_task_state.page_count >= max_pages > 0:
                    self._log("INFO", f"Stopping task: Maximum page limit ({max_pages}) reached.")
                    self._update_state({'status': 'finished', 'last_error': 'Max pages reached'})
                    break
                if max_time_s > 0 and (datetime.now() - start_time).total_seconds() > max_time_s:
                     self._log("INFO", f"Stopping task: Maximum time limit ({max_time_s}s) reached.")
                     self._update_state({'status': 'finished', 'last_error': 'Max time reached'})
                     break

                # 2. Plan / Select Next Action (LLM Call - assumed synchronous for now)
                self._log("DEBUG", f"[RunLoop] Iteration {loop_count}: Updating state to 'planning'.") # Added log
                self._update_state({'status': 'planning'})
                self._log("DEBUG", f"[RunLoop] Iteration {loop_count}: Asking LLM for next action...") # Modified log
                next_action: Optional[OrchestratorAction] = self.llm_core.select_next_action(self.current_task_state)
                self._log("DEBUG", f"[RunLoop] Iteration {loop_count}: LLM call returned. Action: {next_action.type if next_action else 'None'}") # Added log

                if not next_action or isinstance(next_action, FinishAction):
                    reason = next_action.parameters.get('reason', 'LLM decided to finish') if next_action else 'LLM failed to provide action'
                    self._log("INFO", f"Stopping task: {reason}")
                    self._update_state({'status': 'finished', 'last_error': reason})
                    break
                
                self._log("INFO", f"LLM selected action: {next_action.type} with params: {next_action.parameters}")
                if next_action.reasoning:
                    self._log("DEBUG", f"LLM Reasoning: {next_action.reasoning}")
                self._log("DEBUG", f"[RunLoop] Iteration {loop_count}: Preparing to execute action {next_action.type}.") # Added log
                # 3. Execute Action
                observation = None
                action_start_time = time.time() # Added timer
                try:
                    self._log("DEBUG", f"[RunLoop] Iteration {loop_count}: Updating state to 'executing_{next_action.type.lower()}'.") # Added log
                    self._update_state({'status': f'executing_{next_action.type.lower()}'})
                    if next_action.type == 'Search':
                        # Search is synchronous
                        query = next_action.parameters.get('query')
                        if query:
                            observation = self.searcher.search(query)
                        else:
                            raise ValueError("Search action requires a 'query' parameter.")
                    
                    elif next_action.type == 'VisitURL':
                        url = next_action.parameters.get('url')
                        if url and url not in self.current_task_state.visited_urls:
                            # Removed await
                            final_url, content = self.browser.goto_url(url)
                            observation = {"url": final_url, "content": content} # Maybe summarize content?
                            self._update_state({'current_url': final_url, 'page_count': self.current_task_state.page_count + 1})
                            self.current_task_state.visited_urls.add(final_url)
                        elif url in self.current_task_state.visited_urls:
                            observation = {"status": "skipped", "reason": "URL already visited"}
                            self._log("INFO", f"Skipping already visited URL: {url}")
                        else:
                            raise ValueError("VisitURL action requires a 'url' parameter.")

                    elif next_action.type == 'ClickElement':
                        selector = next_action.parameters.get('selector')
                        if selector:
                            # Removed await
                            final_url, content = self.browser.click_element(selector)
                            observation = {"url": final_url, "content": content} # Summarize?
                            if final_url != self.current_task_state.current_url and final_url not in self.current_task_state.visited_urls:
                                self._update_state({'current_url': final_url, 'page_count': self.current_task_state.page_count + 1})
                                self.current_task_state.visited_urls.add(final_url)
                            else:
                                self._update_state({'current_url': final_url})
                        else:
                            raise ValueError("ClickElement action requires a 'selector' parameter.")
                    
                    elif next_action.type == 'FillInput':
                        selector = next_action.parameters.get('selector')
                        text = next_action.parameters.get('text')
                        if selector and text is not None:
                           # Removed await
                           success = self.browser.fill_input(selector, text)
                           observation = {"status": "success" if success else "failure"}
                        else:
                            raise ValueError("FillInput action requires 'selector' and 'text' parameters.")

                    elif next_action.type == 'ExtractData':
                        # Removed await
                        page_content = self.browser.get_content() if self.current_task_state.current_url else "No page loaded"
                        # LLM call is synchronous
                        extracted_data = self.llm_core.extract_structured_data(
                            self.current_task_state.target_schema,
                            page_content,
                            self.current_task_state.task_description
                        )
                        observation = {"extracted_data": extracted_data}
                        # Note: The LLM might decide to fill immediately or wait

                    elif next_action.type == 'FillTable':
                        # This action is synchronous
                        record = next_action.parameters.get('record')
                        if record and self.data_handler:
                            added_row_index = self.data_handler.add_record(record)
                            observation = {"status": "success", "row_index": added_row_index}
                            self._update_state({'collected_data': self.data_handler.get_records()})
                        else:
                            raise ValueError("FillTable action requires a 'record' parameter.")
                    
                    else:
                         # Synchronous
                         self._log("WARNING", f"Unknown or unhandled action type: {next_action.type}")
                         observation = {"status": "error", "message": "Unhandled action type"}
                    
                    action_duration = time.time() - action_start_time # Calculate duration
                    self._log("DEBUG", f"[RunLoop] Iteration {loop_count}: Action {next_action.type} executed in {action_duration:.2f}s. Observation: {str(observation)[:100]}...") # Added log
                
                except Exception as e:
                    # Synchronous error handling
                    action_duration = time.time() - action_start_time # Calculate duration
                    self._log("ERROR", f"[RunLoop] Iteration {loop_count}: Error executing action {next_action.type} after {action_duration:.2f}s: {e}") # Modified log
                    logger.exception("Action execution failed") # Log traceback
                    error_message = f"Error during {next_action.type}: {e}"
                    self._update_state({'error_count': self.current_task_state.error_count + 1, 'last_error': error_message})
                    # Add error handling logic (e.g., retry, skip, stop after N errors)
                    if self.current_task_state.error_count > 5: # Example threshold
                       self._log("CRITICAL", "Stopping task due to excessive errors.")
                       self._update_state({'status': 'error', 'last_error': 'Too many errors'})
                       break

                # 4. Update History
                self._log("DEBUG", f"[RunLoop] Iteration {loop_count}: Updating history with action and observation.") # Added log
                if self.current_task_state:
                    # Corrected: Append directly to the history list
                    self.current_task_state.history.append((next_action, observation))
                    # Re-added: Trigger state update callback to reflect the new history/observation
                    self._update_state({}) 
                else:
                     self._log("WARNING", "[RunLoop] Iteration {loop_count}: Cannot update history, task state is None.")

                # Optional: Short delay between iterations?
                # time.sleep(0.1)
                self._log("DEBUG", f"[RunLoop] Iteration {loop_count} end.") # Added log

        except Exception as loop_error:
            self._log("CRITICAL", f"[RunLoop] Unhandled exception in main loop: {loop_error}")
            logger.exception("Orchestrator run_loop crashed")
            if self.current_task_state:
                self._update_state({'status': 'error', 'last_error': f"Critical loop error: {loop_error}"})
        finally:
             self._log("INFO", "[RunLoop] Exiting run loop.")
             # Ensure cleanup is called when loop exits normally or via break/error
             try:
                 # Removed asyncio.run()
                 self._cleanup()
                 # Signal finished state to the main thread
                 if self.log_callback: # Use log_callback to signal thread finish
                      self._log("DEBUG", "[RunLoop] Signaling finished state via queue.")
                      try:
                          # Use a special log or state type if queue supports it, otherwise use log
                          self.log_callback("internal", "__THREAD_FINISHED__") 
                      except Exception as cb_e:
                          logger.error(f"Error signaling finish via callback: {cb_e}")
             except Exception as cleanup_e:
                  logger.error(f"Error during cleanup after run_loop exit: {cleanup_e}")

    # Removed async
    def _cleanup(self):
        """Clean up resources after a task finishes or is stopped."""
        self._log("INFO", "Cleaning up task resources...")
        # Close browser resources (now synchronous)
        try:
            # Removed await
            self.browser.close()
            self._log("INFO", "Browser resources closed.")
        except Exception as e:
            self._log("ERROR", f"Error closing browser during cleanup: {e}")

        if self.current_task_state and self.current_task_state.status not in ['finished', 'error']:
            # If cleanup is called before a final state was set (e.g., due to stop request)
            self._update_state({'status': 'finished', 'last_error': 'Task stopped or interrupted before completion.'})
        
        # Don't reset current_task_state or data_handler here, allow retrieval after finish
        # self.current_task_state = None
        # self.data_handler = None 
        self.stop_requested = False # Reset stop flag for next potential task
        self._log("INFO", "Cleanup complete.")

    def get_current_data(self) -> Optional[List[Dict[str, Any]]]:
        """Returns the data collected by the current or last finished task."""
        if self.data_handler:
            return self.data_handler.get_records()
        return None

# Example placeholder classes needed for Orchestrator
# These would be implemented in their respective modules

# class LLMCore:
#     def __init__(self, config):
#         pass
#     def select_next_action(self, state: TaskState) -> Optional[OrchestratorAction]:
#         # Placeholder: Return a dummy action
#         print("LLM selecting action...")
#         if not state.current_url:
#              return SearchAction(parameters={"query": state.task_description[:20]}) 
#         else:
#              return FinishAction(parameters={"reason": "Example finish"})
#     def extract_structured_data(self, schema, content, task_desc):
#         print("LLM extracting data...")
#         return {col.name: "extracted_value" for col in schema.columns}

# class Browser:
#     def __init__(self, config):
#         pass
#     def goto_url(self, url):
#         print(f"Browser visiting: {url}")
#         return f"<html><body>Content of {url}</body></html>", url
#     def click_element(self, selector):
#          print(f"Browser clicking: {selector}")
#          return "<html><body>New content after click</body></html>", "http://example.com/afterclick"
#     def fill_input(self, selector, text):
#          print(f"Browser filling '{text}' into: {selector}")
#          return True
#     def get_content(self):
#         return "<html><body>Current page content</body></html>"
#     def close(self):
#         print("Browser closing.")

# class Searcher:
#      def __init__(self, config, api_keys):
#          pass
#      def search(self, query):
#          print(f"Searching for: {query}")
#          return [{"url": f"http://result{i}.com", "title": f"Result {i}"} for i in range(3)]

# class DataHandler:
#     def __init__(self, schema):
#         self.schema = schema
#         self.records = []
#     def add_record(self, record):
#         print(f"Adding record: {record}")
#         self.records.append(record)
#         return len(self.records) - 1
#     def get_records(self):
#         return self.records

if __name__ == '__main__':
    # Basic setup for standalone testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Mock setup (replace with actual config/logging setup)
    # load_config()
    # setup_logging()

    print("Testing Orchestrator...")

    # Define a simple schema
    schema = DataTableSchema(
        columns=[
            ColumnDefinition(name="URL", dtype="URL", description="Source URL"),
            ColumnDefinition(name="Title", dtype="Text", description="Page Title")
        ]
    )

    # Create orchestrator
    orchestrator = Orchestrator()

    # Start a task
    try:
        orchestrator.start_task(
            task_description="Find example websites",
            target_schema=schema,
            start_points=["initial search query"]
        )
    except RuntimeError as e:
        print(f"Error starting task: {e}")
    
    # In a real app, the run_loop would likely run in a background thread.
    # Here, it runs synchronously.
    print("Orchestrator loop finished.")
    final_data = orchestrator.get_current_data()
    print("\nFinal Collected Data:")
    print(final_data) 