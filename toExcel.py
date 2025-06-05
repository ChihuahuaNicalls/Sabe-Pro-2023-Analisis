import pandas as pd

#Funcion para convertir un archivo de texto delimitado por '¬' a un archivo Excel
def toExcel(txt_file_path, excel_file_path):
    try:
        df = pd.read_csv(txt_file_path, delimiter='¬', encoding='utf-8', engine='python')
        df.to_excel(excel_file_path, index=False)
        print(f"Successfully converted '{txt_file_path}' to '{excel_file_path}'")
    except FileNotFoundError:
        print(f"Error: The file '{txt_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


#Definir los nombres de los archivos de texto y Excel
txtFile3 = 'SaberPro_2021.txt'
excelFile = 'SaberPro_2021.xlsx'
toExcel(txtFile3, excelFile)