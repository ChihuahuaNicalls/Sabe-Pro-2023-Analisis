import pandas as pd

file_path = 'icfes_data.xlsx'
data = pd.read_excel(file_path)

grupo = "INGENIERÍA"

dataFiltered = data[
    data['gruporeferencia'] == grupo
]

dataFiltered.to_excel('icfesFiltered.xlsx', index=False)

print(dataFiltered)
