import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import os

# Import the configuration loader
from .config_loader import get_config, BASE_DIR

# Default values (used only if config loading fails or keys are missing)
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_FILE = BASE_DIR / "logs" / "intelliscrape_studio.log"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024 # 10MB
DEFAULT_BACKUP_COUNT = 5

def setup_logging():
    """Configures global logging based on settings loaded from config_loader."""
    try:
        config = get_config()
        log_config = config.get('logging', {})
    except Exception as e:
        print(f"Error loading configuration for logging: {e}. Using defaults.", file=sys.stderr)
        log_config = {}

    log_level_str = log_config.get('level', DEFAULT_LOG_LEVEL)
    log_format = log_config.get('format', DEFAULT_LOG_FORMAT)
    log_file_str = log_config.get('log_file', str(DEFAULT_LOG_FILE)) # Get path from config or default
    max_bytes = log_config.get('max_bytes', DEFAULT_MAX_BYTES)
    backup_count = log_config.get('backup_count', DEFAULT_BACKUP_COUNT)

    level = getattr(logging, log_level_str.upper(), logging.INFO)
    formatter = logging.Formatter(log_format)

    # Ensure log file path is absolute or relative to BASE_DIR
    log_file_path = Path(log_file_str)
    if not log_file_path.is_absolute():
        log_file_path = BASE_DIR / log_file_path

    # Get the root logger
    # Using getLogger() without arguments gets the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level) # Set root logger level

    # --- Clear existing handlers to avoid duplication during reloads/tests --- 
    # Be cautious if external libraries also configure logging.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # --- Configure Console Handler --- 
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    # Optionally set a different level for console vs file
    # console_handler.setLevel(max(level, logging.INFO)) # Example: Never show DEBUG on console
    root_logger.addHandler(console_handler)

    # --- Configure Rotating File Handler --- 
    try:
        # Ensure the directory exists
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Use RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        print(f"Logging configured. Level: {log_level_str}, File: {log_file_path}")

    except Exception as e:
        print(f"Error setting up file logging to {log_file_path}: {e}", file=sys.stderr)
        # Continue without file logging if it fails

# Example usage (typically called once at application startup):
if __name__ == '__main__':
    # This example assumes config/settings.yaml exists and is valid
    # You might need to run config_loader.py first if testing standalone
    from .config_loader import load_config
    try:
        load_config() # Load config before setting up logging
        setup_logging()

        # Test logging
        logger = logging.getLogger('my_test_module')
        logger.debug("This is a debug message (should not appear if level is INFO).")
        logger.info("This is an info message.")
        logger.warning("This is a warning message.")
        logger.error("This is an error message.")
        try:
            1 / 0
        except ZeroDivisionError:
            logger.exception("Caught an exception.")

        print(f"\nLog file should be at: {DEFAULT_LOG_FILE}")
        print("Check the log file for output.")

    except FileNotFoundError:
         print("Error: config/settings.yaml not found. Cannot run logging example.")
    except Exception as e:
         print(f"An unexpected error occurred: {e}") 