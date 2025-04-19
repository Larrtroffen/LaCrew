from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any, Union, Tuple
from datetime import datetime

# --- Data Table Structure ---

class ColumnDefinition(BaseModel):
    """Defines a single column in the user-defined data table."""
    name: str = Field(..., description="The name of the column.")
    dtype: Literal['Text', 'Number', 'Date', 'URL', 'Email', 'Boolean', 'Other'] = Field(
        ..., description="The expected data type for the column."
    )
    description: str = Field(
        ...,
        description="A detailed description or prompt hint for the LLM explaining what data to extract for this column."
    )

class DataTableSchema(BaseModel):
    """Defines the entire structure of the user-defined data table."""
    columns: List[ColumnDefinition] = Field(
        default_factory=list, description="The list of column definitions."
    )
    description: Optional[str] = Field(None, description="Optional high-level description of the table's purpose.")

# --- Orchestrator Actions ---
# Defines the actions the orchestrator can instruct components to perform.

class Action(BaseModel):
    """Base model for all actions."""
    type: str = Field(..., description="The type of action to perform.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters required for the action.")
    reasoning: Optional[str] = Field(None, description="LLM's reasoning for choosing this action.")


class SearchAction(Action):
    """Action to perform a web search."""
    type: Literal['Search'] = 'Search'
    parameters: Dict[str, Any] = Field(..., description="Must contain 'query': str")
    # Example: parameters={'query': 'AI startups funding'}

class VisitURLAction(Action):
    """Action to navigate the browser to a specific URL."""
    type: Literal['VisitURL'] = 'VisitURL'
    parameters: Dict[str, Any] = Field(..., description="Must contain 'url': str")
    # Example: parameters={'url': 'https://example.com'}

class ClickElementAction(Action):
    """Action to click an element on the current webpage."""
    type: Literal['ClickElement'] = 'ClickElement'
    parameters: Dict[str, Any] = Field(..., description="Must contain 'selector': str (CSS selector or XPath)")
    # Example: parameters={'selector': 'a.next_page'}

class FillInputAction(Action):
    """Action to fill text into an input field."""
    type: Literal['FillInput'] = 'FillInput'
    parameters: Dict[str, Any] = Field(..., description="Must contain 'selector': str and 'text': str")
    # Example: parameters={'selector': '#search-box', 'text': 'Query'}

class ExtractDataAction(Action):
    """Action to extract specific data points based on the table schema."""
    type: Literal['ExtractData'] = 'ExtractData'
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional. Can contain hints like 'target_elements': List[str] or specific extraction instructions."
    )
    # Example: parameters={'target_elements': ['div.profile > h2', 'div.profile > p.email']}

class FillTableAction(Action):
    """Action to add a completed record to the data table."""
    type: Literal['FillTable'] = 'FillTable'
    parameters: Dict[str, Any] = Field(
        ..., description="Must contain 'record': Dict[str, Any] matching the DataTableSchema"
    )
    # Example: parameters={'record': {'Company Name': 'AI Corp', 'CEO': 'John Doe', 'Founded': 2023}}

class FinishAction(Action):
    """Action indicating the task is considered complete or cannot proceed."""
    type: Literal['Finish'] = 'Finish'
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional. Can contain 'reason': str (e.g., 'Target achieved', 'Max pages reached', 'Error threshold exceeded')"
    )
    # Example: parameters={'reason': 'Target data collected.'}


# Union type for easy handling in the Orchestrator
OrchestratorAction = Union[
    SearchAction,
    VisitURLAction,
    ClickElementAction,
    FillInputAction,
    ExtractDataAction,
    FillTableAction,
    FinishAction,
    Action # Fallback for potentially custom actions
]

# --- LLM Interaction Models ---

class LLMResponse(BaseModel):
    """Standard structure for responses from the LLM core module."""
    content: Optional[str] = None # Raw text response
    structured_output: Optional[Any] = None # Parsed structured data (e.g., an Action object, a dict)
    error: Optional[str] = None # Error message if any

# --- Task State ---

class TaskState(BaseModel):
    """Represents the current state of a scraping task."""
    task_id: str
    task_description: str
    target_schema: DataTableSchema
    start_points: List[str] = Field(default_factory=list)
    current_url: Optional[str] = None
    page_content: Optional[str] = None # Added field to store current page HTML
    visited_urls: set[str] = Field(default_factory=set)
    collected_data: List[Dict[str, Any]] = Field(default_factory=list) # Consider using the handler's DataFrame directly later
    history: List[Tuple[Action, Any]] = Field(default_factory=list) # Action taken, Observation received
    status: Literal['idle', 'initializing', 'planning', 'searching', 'browsing', 'clicking', 'filling_input', 'extracting', 'filling_table', 'finished', 'error'] = 'idle' # Added more statuses
    error_count: int = 0
    page_count: int = 0
    start_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None # Added field for UI refresh logic
    last_error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True # Allow set and potentially DataFrame 