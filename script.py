import pandas as pd

def convert_txt_to_excel(txt_file_path, excel_file_path):
    """
    Converts a custom-delimited TXT file to an Excel file.

    Args:
        txt_file_path (str): The path to the input TXT file.
        excel_file_path (str): The path to the output Excel file.
    """
    try:
        # Read the TXT file, assuming '¬' as the delimiter
        # The 'engine' parameter is set to 'python' for single-character delimiters
        df = pd.read_csv(txt_file_path, delimiter='¬', encoding='utf-8', engine='python')

        # Write the DataFrame to an Excel file
        df.to_excel(excel_file_path, index=False)
        print(f"Successfully converted '{txt_file_path}' to '{excel_file_path}'")
    except FileNotFoundError:
        print(f"Error: The file '{txt_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Define the input and output file paths
txt_file3 = 'SaberPro_2021.txt'
excel_file3 = 'SaberPro_2021.xlsx'

# Run the conversion
convert_txt_to_excel(txt_file3, excel_file3)