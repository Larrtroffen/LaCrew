import requests
import os
from typing import List, Dict, Any, Optional, Callable, Tuple
import time
from datetime import datetime
import logging

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions # Import Options
from selenium.webdriver.edge.service import Service as EdgeService # Import Edge Service
from selenium.webdriver.edge.options import Options as EdgeOptions # Import Edge Options
# from selenium.webdriver.firefox.service import Service as FirefoxService # Example for Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

# Import selenium-stealth
from selenium_stealth import stealth

# Potentially need WebDriver Manager if auto-download desired, but start without it
# from webdriver_manager.chrome import ChromeDriverManager 

# --- Adjusted Import Path (Keep this if ParserFactory is still used elsewhere) ---
# from intelliscrape_studio.scraping.parsers.factory import ParserFactory # Keep if needed

logger = logging.getLogger(__name__)

# Rename class to Browser according to target structure
class Browser:
    def __init__(self, config: Dict[str, Any]):
        """Initializes the Browser using Selenium.

        Args:
            config: Configuration dictionary for the browser (from scraping.browser section).
        """
        self.config = config
        self.headless = config.get('headless', True)
        # Selenium uses implicit wait or explicit waits, not a single default timeout like Playwright page methods.
        # We'll use explicit waits. Store a reasonable default wait time.
        self.explicit_wait_timeout = config.get('timeout', 60000) / 1000 # Convert ms to seconds for Selenium waits
        # Add a shorter timeout specifically for element interactions
        self.element_interaction_timeout = config.get('element_timeout', 15000) / 1000 # Default 15 seconds
        self.user_agent = config.get('user_agent')
        self.webdriver_path = config.get('webdriver_path') # Optional: Path to webdriver executable
        self.browser_type = config.get('browser_type', 'chrome').lower() # Read browser type, default to chrome

        # Selenium WebDriver instance
        self.driver = None 
        # Removed Playwright specific attributes
        # self.playwright = None
        # self.browser = None
        # self.context = None
        # self.page = None

    def _ensure_browser_initialized(self):
        """Initializes Selenium WebDriver instance if not already done."""
        if self.driver:
            return

        logger.info(f"Initializing Selenium WebDriver ({self.browser_type})...")
        try:
            # --- Configure WebDriver Options based on browser_type ---
            if self.browser_type == 'edge':
                options = EdgeOptions()
                options.use_chromium = True # Important for modern Edge
                ServiceClass = EdgeService
                WebDriverClass = webdriver.Edge
                driver_name = "msedgedriver"
            elif self.browser_type == 'chrome':
                options = ChromeOptions()
                ServiceClass = ChromeService
                WebDriverClass = webdriver.Chrome
                driver_name = "chromedriver"
            # Add elif for 'firefox' here if needed
            else:
                raise ValueError(f"Unsupported browser_type: {self.browser_type}. Use 'chrome' or 'edge'.")

            # Apply common options
            if self.headless:
                options.add_argument('--headless')
                options.add_argument('--disable-gpu') # Often needed for headless mode
            if self.user_agent:
                options.add_argument(f'user-agent={self.user_agent}')
            options.add_argument("--window-size=1920,1080") # Set a reasonable window size
            options.add_argument("--log-level=3") # Reduce console noise 

            # --- Specify WebDriver Path (if provided) ---
            if self.webdriver_path and os.path.exists(self.webdriver_path):
                 service = ServiceClass(executable_path=self.webdriver_path)
                 logger.info(f"Using {driver_name} from path: {self.webdriver_path}")
                 self.driver = WebDriverClass(service=service, options=options)
                 # Apply stealth patches only if using Chrome
                 if self.browser_type == 'chrome':
                     try:
                         stealth(self.driver,
                                 languages=["en-US", "en"],
                                 vendor="Google Inc.",
                                 platform="Win32",
                                 webgl_vendor="Intel Inc.",
                                 renderer="Intel Iris OpenGL Engine",
                                 fix_hairline=True,
                                 )
                         logger.info("Applied selenium-stealth patches for Chrome.")
                     except Exception as stealth_e:
                         logger.warning(f"Failed to apply selenium-stealth patches: {stealth_e}")
                 else:
                     logger.info("Skipping selenium-stealth patches (not using Chrome).")
            else:
                 # Attempt to use WebDriver from PATH
                 logger.info(f"Attempting to use {driver_name} from system PATH...")
                 try:
                     # This assumes the correct driver (chromedriver/msedgedriver) is in PATH
                     self.driver = WebDriverClass(options=options)
                     # Apply stealth patches only if using Chrome
                     if self.browser_type == 'chrome':
                         try:
                             stealth(self.driver,
                                     languages=["en-US", "en"],
                                     vendor="Google Inc.",
                                     platform="Win32",
                                     webgl_vendor="Intel Inc.",
                                     renderer="Intel Iris OpenGL Engine",
                                     fix_hairline=True,
                                     )
                             logger.info("Applied selenium-stealth patches for Chrome.")
                         except Exception as stealth_e:
                             logger.warning(f"Failed to apply selenium-stealth patches: {stealth_e}")
                     else:
                         logger.info("Skipping selenium-stealth patches (not using Chrome).")
                 except WebDriverException as e:
                     logger.error(f"Failed to find/start {driver_name} in PATH. Ensure it's installed and in PATH, or specify 'webdriver_path' in config. Error: {e}")
                     raise

            # Implicit wait - less preferred than explicit waits, but can set a base
            # self.driver.implicitly_wait(5) # Wait up to 5 seconds for elements if not immediately found
            
            logger.info("Selenium WebDriver initialization complete.")

        except WebDriverException as e:
             logger.error(f"Failed to initialize Selenium WebDriver: {e}", exc_info=True)
             raise
        except Exception as e:
            logger.error(f"Unexpected error initializing Selenium: {e}", exc_info=True)
            self.close() # Attempt cleanup
            raise
        # Removed Playwright specific initialization

    def goto_url(self, url: str) -> Tuple[str, Optional[str]]:
        """Navigates to the specified URL using Selenium and returns the final URL and page source.

        Args:
            url: The URL to navigate to.

        Returns:
            A tuple containing:
                - final_url (str): The URL after any redirects.
                - content (Optional[str]): The HTML content of the page, or None on error.
        """
        logger.info(f"Navigating to URL (Selenium): {url}")
        try:
            self._ensure_browser_initialized()
            if not self.driver:
                raise RuntimeError("Selenium WebDriver not initialized")

            self.driver.get(url)
            
            # Wait for page to load - using a simple check for <body> presence as a basic indicator
            # More robust checks might be needed depending on the site (e.g., wait for specific element)
            try:
                WebDriverWait(self.driver, self.explicit_wait_timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                logger.warning(f"Timeout waiting for body tag on {url}. Page might not have loaded correctly.")
                # Decide whether to return content or None here based on requirements
                # For now, let's try to get content anyway

            final_url = self.driver.current_url
            page_source = self.driver.page_source # Get page source instead of content
            
            # Note: Selenium doesn't easily provide HTTP status code like Playwright's response object.
            # We assume navigation worked if no exception occurred.
            logger.info(f"Navigation successful. Final URL: {final_url}")
            return final_url, page_source

        except TimeoutException as e:
             logger.error(f"Timeout during navigation or waiting for page load at {url}: {e}", exc_info=True)
             # Try to get current URL even on timeout
             try:
                 final_url = self.driver.current_url
             except Exception:
                 final_url = url # Fallback
             return final_url, None
        except WebDriverException as e:
            logger.error(f"Selenium WebDriver error during navigation to {url}: {e}", exc_info=True)
            return url, None # Return original URL and None content on WebDriver error
        except Exception as e:
            logger.error(f"Unexpected error during navigation to {url}: {e}", exc_info=True)
            return url, None

    def get_content(self) -> Optional[str]:
        """Returns the HTML source of the current page using Selenium."""
        if not self.driver:
            logger.warning("Cannot get content, Selenium WebDriver not initialized.")
            return None
        try:
            content = self.driver.page_source
            return content
        except WebDriverException as e:
            logger.error(f"Failed to get page source: {e}")
            return None
        except Exception as e: # Catch broader exceptions during page source access
             logger.error(f"Unexpected error getting page source: {e}", exc_info=True)
             return None

    def click_element(self, selector: str, wait_for_navigation: bool = True) -> Tuple[str, Optional[str]]:
        """Clicks an element matching the CSS selector and returns the new page state.

        Args:
            selector: CSS selector or XPath for the element to click.
            wait_for_navigation: Whether to wait for potential navigation after the click.

        Returns:
            A tuple containing:
                - final_url (str): The URL after the click (might be the same).
                - content (Optional[str]): HTML content of the page after the click, or None on error.
        """
        logger.info(f"Attempting to click element (Selenium): {selector}")
        if not self.driver:
            logger.error("Cannot click element, Selenium WebDriver not initialized.")
            return "unknown", None

        current_url = "unknown"
        try:
            current_url = self.driver.current_url # Get URL before click
            logger.debug(f"Current URL before clicking '{selector}': {current_url}")

            # Use the shorter interaction timeout here
            wait = WebDriverWait(self.driver, self.element_interaction_timeout)
            element = None

            # Step 1: Wait for element visibility
            try:
                logger.debug(f"Waiting for visibility of element: {selector}")
                visible_element = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, selector)))
                logger.debug(f"Element {selector} found and is visible.")
                # Step 2: Wait for element to be clickable
                try:
                    logger.debug(f"Waiting for element to be clickable: {selector}")
                    element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                    logger.debug(f"Element {selector} is clickable.")
                except TimeoutException:
                    logger.error(f"Timeout waiting for element '{selector}' to become clickable (it was visible).", exc_info=True)
                    # Optionally: Log element details like is_enabled()
                    try:
                        if visible_element:
                             logger.warning(f"Element '{selector}' state: enabled={visible_element.is_enabled()}, displayed={visible_element.is_displayed()}")
                    except Exception as detail_e:
                        logger.warning(f"Could not get details for non-clickable element: {detail_e}")
                    return current_url, None # Return if not clickable

            except TimeoutException:
                 logger.error(f"Timeout waiting for element '{selector}' to be visible.", exc_info=True)
                 # If not visible, we can't click it.
                 # Optionally, save screenshot/source for debugging
                 try:
                    # Construct filename with timestamp and selector info
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_selector = "".join(c if c.isalnum() else '_' for c in selector)[:50] # Sanitize selector for filename
                    screenshot_filename = f"error_screenshot_{ts}_{safe_selector}.png"
                    pagesource_filename = f"error_pagesource_{ts}_{safe_selector}.html"
                    
                    if self.driver:
                        logger.info(f"Saving screenshot to {screenshot_filename}")
                        self.driver.save_screenshot(screenshot_filename)
                        logger.info(f"Saving page source to {pagesource_filename}")
                        with open(pagesource_filename, "w", encoding="utf-8") as f:
                            f.write(self.driver.page_source)
                 except Exception as save_e:
                    logger.error(f"Failed to save screenshot or page source: {save_e}")
                 return current_url, None
            except NoSuchElementException: # Should be caught by WebDriverWait, but good practice
                 logger.error(f"Element '{selector}' not found in DOM after explicit wait.", exc_info=True)
                 return current_url, None

            # If we got the clickable element
            if element:
                # Scroll into view if needed (Selenium often handles this, but can be explicit)
                # self.driver.execute_script("arguments[0].scrollIntoView(true);", element)
                # time.sleep(0.5) # Small delay after scroll before click

                element.click()
                logger.info(f"Clicked element: {selector}")

                # If navigation is expected, add a small delay or a more specific wait
                if wait_for_navigation:
                     time.sleep(2) # Simple delay - VERY unreliable, replace with better wait if possible
                     logger.debug("Waited briefly after click for potential navigation.")

                final_url = self.driver.current_url
                content = self.driver.page_source
                logger.info(f"Click action complete. Final URL: {final_url}")
                return final_url, content
            else:
                # Should not happen if checks above are correct, but as a fallback
                logger.error(f"Element was not assigned after waits, cannot click '{selector}'.")
                return current_url, None

        except WebDriverException as e:
            logger.error(f"Selenium WebDriver error clicking element '{selector}': {e}", exc_info=True)
            return current_url, None
        except Exception as e:
            logger.error(f"Unexpected error clicking element '{selector}': {e}", exc_info=True)
            return current_url, None

    def fill_input(self, selector: str, text: str) -> bool:
        """Fills text into an input field identified by the CSS selector using Selenium."""
        logger.info(f"Attempting to fill '{text[:50]}...' into element (Selenium): {selector}")
        if not self.driver:
            logger.error("Cannot fill input, Selenium WebDriver not initialized.")
            return False
        try:
            # Wait for element to be present using the shorter timeout
            wait = WebDriverWait(self.driver, self.element_interaction_timeout)
            element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
            
            # Clear the input field first (good practice)
            element.clear()
            element.send_keys(text)
            
            logger.info(f"Successfully filled input for {selector}")
            return True
        except TimeoutException:
             logger.error(f"Timeout waiting for input element '{selector}'.", exc_info=True)
             return False
        except NoSuchElementException:
             logger.error(f"Input element '{selector}' not found.", exc_info=True)
             return False
        except WebDriverException as e:
            logger.error(f"Selenium WebDriver error filling input '{selector}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error filling input '{selector}': {e}", exc_info=True)
            return False

    def close(self):
        """Closes the Selenium WebDriver session."""
        logger.info("Closing Selenium WebDriver...")
        if self.driver:
            try:
                self.driver.quit() # Closes all browser windows and ends the session
                logger.info("Selenium WebDriver closed.")
            except WebDriverException as e:
                logger.warning(f"Error closing Selenium WebDriver: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error during WebDriver close: {e}")
            finally:
                 self.driver = None # Ensure driver attribute is cleared
        # Removed Playwright specific cleanup

    # --- Simple HTTP GET using Requests (remains synchronous) ---
    @staticmethod
    def http_get(url: str, timeout: int = 30) -> Tuple[str, Optional[str], int]:
        """Performs a simple HTTP GET request using the requests library.
        
        Args:
            url: The URL to fetch.
            timeout: Request timeout in seconds.
            
        Returns:
            A tuple containing:
               - final_url (str): URL after redirects.
               - content (Optional[str]): HTML content or None on error.
               - status_code (int): HTTP status code.
        """
        logger.info(f"Performing HTTP GET request for: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            final_url = response.url
            status_code = response.status_code
            logger.info(f"HTTP GET completed. Final URL: {final_url}, Status: {status_code}")
            if response.ok:
                response.encoding = response.apparent_encoding # Guess encoding
                return final_url, response.text, status_code
            else:
                logger.warning(f"HTTP GET request failed with status {status_code}")
                return final_url, None, status_code
        except requests.exceptions.Timeout:
            logger.error(f"HTTP GET request timed out for {url}")
            return url, None, 408 # Request Timeout
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP GET request failed for {url}: {e}")
            return url, None, 500 # Internal Server Error
        except Exception as e:
            logger.error(f"Unexpected error during HTTP GET for {url}: {e}", exc_info=True)
            return url, None, 500

    # Ensure cleanup on deletion - less critical with Selenium as driver.quit() is the main cleanup.
    def __del__(self):
        # Explicit close() is strongly preferred. 
        # Trying to call self.close() here is unreliable.
        if self.driver:
            logger.warning("Browser object deleted without explicit close(). WebDriver session might remain active.")
            # Attempting self.close() here is generally discouraged in __del__

# Example Usage (for testing)
def main():
    print("Testing Browser class...")
    config = {'headless': True, 'timeout': 30000}
    browser = Browser(config)

    # Test HTTP GET
    print("\n--- Testing HTTP GET ---")
    f_url, content, status = browser.http_get("http://httpbin.org/html")
    print(f"GET Status: {status}, Final URL: {f_url}")
    # print(f"GET Content Snippet: {content[:200]}... if content else "No Content")

    # Test Selenium Navigation
    print("\n--- Testing Selenium Navigation ---")
    final_url, html_content = browser.goto_url("http://httpbin.org/html")
    print(f"Selenium Nav Final URL: {final_url}")
    if html_content:
        print(f"Selenium Nav Content Length: {len(html_content)}")
        # Test Get Content
        print("\n--- Testing Get Content ---")
        current_content = browser.get_content()
        print(f"Get Content Length: {len(current_content) if current_content else 0}")
    else:
        print("Selenium Navigation failed to get content.")

    # Test Input Fill (Example on a known site)
    print("\n--- Testing Fill Input ---")
    nav_url, _ = browser.goto_url("https://duckduckgo.com")
    if nav_url:
        success = browser.fill_input("#search_form_input_homepage", "Selenium test")
        print(f"Fill Input Success: {success}")
        # Optional: Add click test for search button
        # click_url, click_content = await browser.click_element("#search_button_homepage")
        # print(f"Click URL: {click_url}")

    browser.close()
    print("\nBrowser closed.")

if __name__ == "__main__":
    main() 