import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo Excel (reemplaza 'archivo.xlsx' con el nombre de tu archivo)
df = pd.read_excel("C:/Users/Sebas/Desktop/Probabilidad_datos/SaberProFiltered.xlsx")

# Crear el histograma de la columna 'punt_global'
plt.hist(df['punt_global'], bins=10, edgecolor='black')  # Puedes ajustar bins

# Etiquetas y título
plt.xlabel('Puntaje Global')
plt.ylabel('Frecuencia')
plt.title('Histograma de puntaje global')

# Mostrar gráfico
plt.show()
