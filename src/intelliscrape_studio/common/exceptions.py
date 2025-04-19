# src/intelliscrape_studio/common/exceptions.py

class IntelliScrapeError(Exception):
    """Base exception class for IntelliScrape Studio errors."""
    pass

class ConfigurationError(IntelliScrapeError):
    """Error related to configuration loading or validation."""
    pass

class LLMError(IntelliScrapeError):
    """Error related to LLM interactions."""
    pass

class ProviderError(LLMError):
     """Error specific to an LLM provider API call."""
     pass

class ScrapingError(IntelliScrapeError):
    """Error related to web scraping or browser interaction."""
    pass

class ParsingError(ScrapingError):
     """Error during page content parsing."""
     pass
     
class DataHandlingError(IntelliScrapeError):
     """Error related to data validation, merging, or export."""
     pass

# Add more specific exceptions as needed 