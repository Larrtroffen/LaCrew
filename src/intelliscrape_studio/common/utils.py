# src/intelliscrape_studio/common/utils.py

import re
from typing import Optional


def sanitize_filename(filename: str, replacement: str = '_') -> str:
    """Sanitizes a string to be used as a filename by removing invalid characters.

    Args:
        filename: The original filename string.
        replacement: The character to replace invalid characters with.

    Returns:
        A sanitized filename string.
    """
    # Remove characters invalid in most filesystems (Windows/Unix)
    # Allow letters, numbers, underscores, hyphens, periods, spaces
    # Adjust the regex as needed for more/less strictness
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', replacement, filename)
    # Replace multiple consecutive replacements with a single one
    sanitized = re.sub(rf'{re.escape(replacement)}{{2,}}', replacement, sanitized)
    # Remove leading/trailing whitespace and replacements
    sanitized = sanitized.strip(f' .{replacement}')
    # Limit length (optional)
    # max_len = 200
    # sanitized = sanitized[:max_len]
    return sanitized if sanitized else "downloaded_file"

# Add other common utility functions here, for example:
# - Function to load data from various file types
# - Function to handle rate limiting or retries consistently
# - Function to normalize URLs (could move from scraping.search)


# Example of a function potentially moved from scraping.search
# Need to import urlparse
from urllib.parse import urlparse

def is_valid_url(url: str) -> bool:
    """Checks if a URL has a valid scheme (http/https) and network location."""
    if not isinstance(url, str) or len(url) > 2048: # Basic checks
        return False
    try:
        result = urlparse(url)
        return result.scheme in ['http', 'https'] and bool(result.netloc)
    except ValueError:
        return False 