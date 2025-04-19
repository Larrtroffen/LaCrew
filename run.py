import sys
import os
from pathlib import Path

# --- Add project src to path ---
# Assume run.py is in the project root directory (e.g., Lacrew/)
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
    print(f"Added {SRC_DIR} to sys.path")

# --- Now import and run the Streamlit app's main function ---
try:
    # Note: The app itself handles streamlit execution indirectly
    # We just need to ensure the modules can be imported correctly.
    # Running this script might not launch Streamlit directly unless
    # app.main() is modified or streamlit methods are called here.
    
    # A better way to run is usually: streamlit run src/intelliscrape_studio/app.py
    
    # If app.py's main function is intended to be runnable directly (less common for Streamlit):
    # from intelliscrape_studio import app 
    # if __name__ == "__main__":
    #     app.main() 
    
    # --- Alternative: Use subprocess to run Streamlit command --- 
    # This is often more reliable for Streamlit apps within packages.
    import subprocess
    
    app_path = SRC_DIR / "intelliscrape_studio" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    
    print(f"Running Streamlit command: {' '.join(command)}")
    
    # Run streamlit command from the project root directory
    process = subprocess.Popen(command, cwd=PROJECT_ROOT)
    process.wait() # Wait for Streamlit process to finish
    
except ImportError as e:
    print(f"Error importing app: {e}")
    print("Please ensure you are running run.py from the project root directory")
    print("and the src/intelliscrape_studio structure is correct.")
except FileNotFoundError:
    print(f"Error: Could not find Streamlit executable or app file at {app_path}")
    print("Make sure Streamlit is installed in your environment.")
except Exception as e:
    print(f"An unexpected error occurred: {e}") 