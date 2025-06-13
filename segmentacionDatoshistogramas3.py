import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo Excel
df = pd.read_excel("C:/Users/Sebas/Desktop/Probabilidad_datos/SaberProFiltered.xlsx")

# Definir los rangos y nombres de los niveles
bins = [0, 60, 120, 180, 240, 300]
labels = ['Nivel Bajo', 'Medio Bajo', 'Nivel Medio', 'Medio Alto', 'Nivel Alto']

# Crear nueva columna con los niveles
df['nivel'] = pd.cut(df['punt_global'], bins=bins, labels=labels, right=False)

# Contar frecuencia de cada nivel
nivel_counts = df['nivel'].value_counts().sort_index()

# Graficar
plt.bar(nivel_counts.index, nivel_counts.values, color='skyblue', edgecolor='black')

# Etiquetas y título
plt.xlabel('Nivel')
plt.ylabel('Cantidad de personas')
plt.title('Distribución por Nivel de Puntaje Global')
plt.xticks(rotation=45)
plt.tight_layout()

# Mostrar gráfico
plt.show()
