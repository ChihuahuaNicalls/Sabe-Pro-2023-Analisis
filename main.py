import pandas as pd

file_path = 'icfes_data.xlsx'
data = pd.read_excel(file_path)

cods = [1301, 1101, 1102, 1103, 1104, 1206, 1701, 1702, 1831, 1731]
grupo = "INGENIERÍA"

dataFiltered = data[
    data['inst_cod_institucion'].isin(cods) &
    (data['gruporeferencia'] == grupo)
]

dataFiltered.to_excel('icfesFiltered.xlsx', index=False)

print(dataFiltered)