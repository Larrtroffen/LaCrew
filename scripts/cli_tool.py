import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import argparse # Use argparse for better CLI handling

# Add src directory to sys.path to allow importing the package
# This assumes the script is run from the project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# --- Adjusted Import Paths ---
try:
    from intelliscrape_studio.llm.core import LLMCore
    from intelliscrape_studio.scraping.browser import Browser as WebInteraction # Alias Browser as WebInteraction if core logic expects that name
    from intelliscrape_studio.data.handler import DataHandler
    # LLMProviderFactory might be implicitly used by LLMCore, direct import might not be needed here
    # from intelliscrape_studio.llm.providers.factory import LLMProviderFactory 
except ImportError as e:
    print(f"Error importing library components: {e}")
    print("Please ensure the script is run from the project root directory and the src structure is correct.")
    sys.exit(1)

# Basic logging callback for CLI
def cli_log(message: str, level: str = "info"):
    print(f"[{level.upper()}] {message}")

# Basic progress callback for CLI
def cli_progress(current: int, total: int, message: str):
    print(f"Progress: {current}/{total} - {message}")

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="IntelliScrape Studio CLI Tool")
    parser.add_argument("-t", "--task", required=True, help="Task description for data collection.")
    parser.add_argument("-s", "--schema", required=True, help="Path to the data schema CSV file (columns: Field, Type, Required[optional]).")
    parser.add_argument("-o", "--output", default="collected_data.csv", help="Output file path (e.g., data.csv, data.xlsx, data.json). Format determined by extension.")
    parser.add_argument("-p", "--pages", type=int, default=5, help="Maximum number of pages to browse.")
    parser.add_argument("-n", "--count", type=int, default=10, help="Maximum number of data items to collect.")
    parser.add_argument("-d", "--domain", default=None, help="Restrict browsing to a specific domain (e.g., 'example.com').")
    parser.add_argument("--provider", default="openai", help="LLM provider name (e.g., openai).")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM model name.")
    # Add other relevant arguments like proxy, search engine if needed
    
    args = parser.parse_args()

    # --- Load Environment & Config ---
    load_dotenv()
    api_key = os.getenv(f"{args.provider.upper()}_API_KEY")
    proxy_url = os.getenv('PROXY_SERVER') # Use proxy if set in .env

    if not api_key:
        print(f"Error: API key for {args.provider.upper()} not found in .env file.")
        sys.exit(1)

    # --- Load Data Schema ---
    try:
        schema_df = pd.read_csv(args.schema)
        # Basic schema validation
        if 'Field' not in schema_df.columns or 'Type' not in schema_df.columns:
             raise ValueError("Schema CSV must contain 'Field' and 'Type' columns.")
        # Ensure 'Required' column exists, default to False if not
        if 'Required' not in schema_df.columns:
             schema_df['Required'] = False
        else:
             # Convert potentially non-boolean values to boolean
             schema_df['Required'] = schema_df['Required'].apply(lambda x: str(x).lower() in ['true', '1', 'yes', 'y'])
        cli_log(f"Loaded schema from: {args.schema}")
    except FileNotFoundError:
        print(f"Error: Schema file not found at {args.schema}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading schema: {e}")
        sys.exit(1)
    
    # --- Initialize Components ---
    cli_log("Initializing components...")
    try:
        llm_core = LLMCore(
            provider_name=args.provider,
            api_key=api_key,
            model_name=args.model,
            log_callback=cli_log
        )
        web_interaction = WebInteraction( # Using Browser class aliased as WebInteraction
            log_callback=cli_log,
            proxy=proxy_url
            # search_engine=args.search_engine # Add if search engine arg is added
        )
        data_handler = DataHandler()
    except Exception as e:
         print(f"Error initializing components: {e}")
         sys.exit(1)
    
    # --- Execute Task ---
    cli_log(f"Starting data collection task: {args.task}")
    results_df = pd.DataFrame() # Initialize empty dataframe
    try:
        results_df = llm_core.execute_task(
            task_description=args.task,
            data_schema=schema_df,
            web_interaction=web_interaction,
            max_pages=args.pages,
            domain_restriction=args.domain,
            max_data_count=args.count,
            progress_callback=cli_progress
        )
        
        if not results_df.empty:
            print("\n--- Collected Data ---")
            print(results_df.to_string()) # Use to_string for better CLI display
            
            # --- Export Data ---
            output_format = Path(args.output).suffix[1:].lower() # Get format from extension
            if not output_format:
                output_format = 'csv' # Default to csv if no extension
                output_file = f"{args.output}.{output_format}"
            else:
                output_file = args.output
                
            try:
                export_data_bytes = data_handler.export_data(results_df, output_format)
                with open(output_file, "wb") as f:
                    f.write(export_data_bytes)
                cli_log(f"Data successfully exported to {output_file} ({output_format} format)", "success")
            except ValueError as e:
                 print(f"Error during export: {e}") # Handle format errors from DataHandler
            except Exception as e:
                 print(f"Error writing output file {output_file}: {e}")
                 
        else:
            cli_log("Task finished, but no data was collected.", "warning")
            
    except ValueError as e:
         # Catch validation errors from LLMCore.execute_task (e.g., bad max_pages)
         print(f"Execution Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during task execution: {e}")
    finally:
         # Ensure browser is closed even if errors occur
         cli_log("Cleaning up resources...")
         web_interaction.close() 
         cli_log("Cleanup complete.")

if __name__ == "__main__":
    main() 