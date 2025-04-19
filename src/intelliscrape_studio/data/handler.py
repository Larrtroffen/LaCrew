import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from io import BytesIO

# Import Pydantic models from data.models
from .models import DataTableSchema, ColumnDefinition

logger = logging.getLogger(__name__)

class DataHandler:
    """Manages the collected structured data using a Pandas DataFrame."""

    def __init__(self, schema: DataTableSchema):
        """Initializes the DataHandler with the target data schema.

        Args:
            schema: The DataTableSchema defining the structure of the data to be collected.
        """
        self.schema = schema
        self.columns = [col.name for col in schema.columns]
        # Initialize an empty DataFrame with columns based on the schema
        self.dataframe = pd.DataFrame(columns=self.columns)
        logger.info(f"DataHandler initialized with columns: {self.columns}")
        # Optional: Try setting initial dtypes (might be tricky with mixed types/None)
        # self._set_initial_dtypes()

    def _set_initial_dtypes(self):
        """Attempt to set initial data types based on schema (best effort)."""
        dtype_map = {
            'Text': 'object', # Use object for strings
            'Number': 'float64', # Use float to accommodate potential NaNs
            'Date': 'datetime64[ns]', # Pandas datetime type
            'URL': 'object',
            'Email': 'object',
            'Boolean': 'boolean' # Pandas nullable boolean type
        }
        schema_dtypes = {col.name: dtype_map.get(col.dtype, 'object') for col in self.schema.columns}
        try:
             self.dataframe = self.dataframe.astype(schema_dtypes)
             logger.debug(f"Initial DataFrame dtypes set: {self.dataframe.dtypes}")
        except Exception as e:
             logger.warning(f"Could not set initial DataFrame dtypes: {e}. Columns will default to object.")

    def add_record(self, record: Dict[str, Any]) -> int:
        """Adds a single record (dictionary) to the DataFrame.

        Performs basic validation: ensures all schema columns exist in the record 
        (filling missing ones with None) and removes extra fields.

        Args:
            record: A dictionary representing a row of data.

        Returns:
            The index of the added row.
        """
        if not isinstance(record, dict):
            logger.warning(f"Attempted to add non-dict record: {type(record)}. Skipping.")
            return -1

        # Prepare the record according to the schema
        validated_record = {}
        missing_cols = []
        extra_cols = list(record.keys())

        for col_name in self.columns:
            validated_record[col_name] = record.get(col_name) # Get value or None if missing
            if col_name not in record:
                 missing_cols.append(col_name)
            else:
                 if col_name in extra_cols:
                      extra_cols.remove(col_name)
        
        if missing_cols:
            logger.debug(f"Record missing columns (filled with None): {missing_cols}. Original keys: {list(record.keys())}")
        if extra_cols:
            logger.debug(f"Record contained extra columns (ignored): {extra_cols}. Original keys: {list(record.keys())}")

        # --- Basic Type Coercion (Optional but helpful) ---
        # Attempt to convert values based on schema dtype before adding.
        # More robust validation/conversion could happen before calling add_record.
        for col_def in self.schema.columns:
            col_name = col_def.name
            target_dtype = col_def.dtype
            current_value = validated_record[col_name]

            if current_value is None:
                 continue
                 
            try:
                if target_dtype == 'Number' and not isinstance(current_value, (int, float)):
                    validated_record[col_name] = float(current_value)
                elif target_dtype == 'Boolean' and not isinstance(current_value, bool):
                    if str(current_value).lower() in ('true', '1', 'yes', 'y'):
                         validated_record[col_name] = True
                    elif str(current_value).lower() in ('false', '0', 'no', 'n'):
                         validated_record[col_name] = False
                    else:
                         validated_record[col_name] = None # Cannot convert to bool
                elif target_dtype == 'Date':
                     # Basic date parsing - can be expanded or handled upstream
                     if not isinstance(current_value, pd.Timestamp):
                         try:
                             validated_record[col_name] = pd.to_datetime(current_value)
                         except (ValueError, TypeError):
                             logger.warning(f"Could not convert value '{current_value}' to Date for column '{col_name}'. Keeping original.")
                             # Keep original or set to None? Let's keep original for now.
                             # validated_record[col_name] = None 
                # Ensure Text, URL, Email are strings
                elif target_dtype in ('Text', 'URL', 'Email') and not isinstance(current_value, str):
                    validated_record[col_name] = str(current_value)
            except (ValueError, TypeError) as e:
                logger.warning(f"Type conversion failed for column '{col_name}' (Value: '{current_value}', Target: {target_dtype}): {e}. Keeping original value.")
        # --- End Basic Type Coercion ---

        try:
            # Convert the single record to a DataFrame to append
            # Using loc is generally preferred over append for adding rows
            new_index = len(self.dataframe)
            self.dataframe.loc[new_index] = validated_record
            logger.info(f"Record added at index {new_index}. Total records: {len(self.dataframe)}")
            return new_index
        except Exception as e:
            logger.error(f"Failed to add record to DataFrame: {e}", exc_info=True)
            logger.error(f"Failed record data: {validated_record}")
            return -1

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the current Pandas DataFrame containing the collected data."""
        return self.dataframe.copy() # Return a copy to prevent external modification

    def get_records(self) -> List[Dict[str, Any]]:
        """Returns all collected records as a list of dictionaries."""
        return self.dataframe.to_dict(orient='records')

    def export_to_csv(self, file_path: Optional[str] = None) -> Optional[bytes]:
        """Exports the DataFrame to a CSV file or returns bytes.

        Args:
            file_path: Optional path to save the CSV file. If None, returns bytes.

        Returns:
            Bytes of the CSV data if file_path is None, otherwise None.
        """
        if self.dataframe.empty:
            logger.warning("No data to export to CSV.")
            return b"" if file_path is None else None
        try:
            csv_data = self.dataframe.to_csv(index=False, encoding='utf-8-sig')
            if file_path:
                with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
                    f.write(csv_data)
                logger.info(f"Data successfully exported to CSV: {file_path}")
                return None
            else:
                return csv_data.encode('utf-8-sig') # Already encoded by to_csv?
        except Exception as e:
            logger.error(f"Error exporting data to CSV ('{file_path}' if file_path else 'bytes'): {e}", exc_info=True)
            return None
            
    def export_to_excel(self, file_path: Optional[str] = None) -> Optional[bytes]:
        """Exports the DataFrame to an Excel file or returns bytes.

        Args:
            file_path: Optional path to save the Excel file. If None, returns bytes.

        Returns:
            Bytes of the Excel data if file_path is None, otherwise None.
        """
        if self.dataframe.empty:
            logger.warning("No data to export to Excel.")
            return b"" if file_path is None else None
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Avoid writing pandas index to Excel file
                self.dataframe.to_excel(writer, index=False, sheet_name='Sheet1')
            excel_data = output.getvalue()

            if file_path:
                with open(file_path, 'wb') as f:
                    f.write(excel_data)
                logger.info(f"Data successfully exported to Excel: {file_path}")
                return None
            else:
                return excel_data
        except Exception as e:
            logger.error(f"Error exporting data to Excel ('{file_path}' if file_path else 'bytes'): {e}", exc_info=True)
            return None
                
    def clear_data(self):
         """Clears all data from the handler, resetting the DataFrame."""
         self.dataframe = pd.DataFrame(columns=self.columns)
         # Optional: Re-apply dtypes if needed
         # self._set_initial_dtypes()
         logger.info("Data handler cleared.")

# Example Usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Define schema
    schema = DataTableSchema(
        columns=[
            ColumnDefinition(name="Company Name", dtype="Text", description="Official company name"),
            ColumnDefinition(name="CEO", dtype="Text", description="Current CEO"),
            ColumnDefinition(name="Founded", dtype="Number", description="Year founded"),
            ColumnDefinition(name="Website", dtype="URL", description="Official website")
        ]
    )

    # Initialize handler
    handler = DataHandler(schema)
    print("Initial DataFrame:")
    print(handler.get_dataframe())

    # Add records
    handler.add_record({"Company Name": "TestCorp", "CEO": "Alice", "Founded": "2020", "Website": "https://test.com", "Extra": "ignored"})
    handler.add_record({"Company Name": "Innovate Inc", "CEO": "Bob", "Founded": 2021, "Website": "http://innovate.co"})
    handler.add_record({"Company Name": "Data LLC", "CEO": "Charlie", "Founded": None, "Website": "invalid-url"}) # Website will be kept as string

    print("\nDataFrame after adding records:")
    df = handler.get_dataframe()
    print(df)
    print("\nDataFrame dtypes:")
    print(df.dtypes)

    # Get records as list
    records_list = handler.get_records()
    print("\nRecords as list:")
    import json
    print(json.dumps(records_list, indent=2))

    # Export (to bytes)
    print("\nExporting to CSV (bytes)...")
    csv_bytes = handler.export_to_csv()
    if csv_bytes:
        print(f"CSV Bytes length: {len(csv_bytes)}")
        # print(csv_bytes.decode('utf-8-sig'))
    
    print("\nExporting to Excel (bytes)...")
    excel_bytes = handler.export_to_excel()
    if excel_bytes:
        print(f"Excel Bytes length: {len(excel_bytes)}")

    # Clear data
    handler.clear_data()
    print("\nDataFrame after clearing:")
    print(handler.get_dataframe()) 