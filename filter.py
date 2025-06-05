import pandas as pd

file_path = 'SaberPro.xlsx'
data = pd.read_excel(file_path)

grupo = "INGENIERÍA"

dataFiltered = data[
    data['gruporeferencia'] == grupo
]

dataFiltered.to_excel('SaberProFiltered.xlsx', index=False)
print(f"Filtered data for group '{grupo}' has been saved to 'SaberProFiltered.xlsx'.")