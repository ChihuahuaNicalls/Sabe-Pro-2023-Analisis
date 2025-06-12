import pandas as pd
import matplotlib.pyplot as plt
import re

# Cargar datos
df = pd.read_excel("C:/Users/Sebas/Desktop/Probabilidad_datos/SaberProFiltered.xlsx")

# Normalizar nombres de departamentos
df['estu_depto_reside'] = df['estu_depto_reside'].str.strip().str.lower()

# Columnas a graficar
columnas_categoricas = ['estu_genero', 'estu_depto_reside', 'mod_ingles_desem']
columna_numerica_textual = 'fami_estratovivienda'

# Diccionario con títulos personalizados
titulos = {
    'estu_genero': 'Distribución por Género',
    'estu_depto_reside': 'Departamentos de Residencia',
    'mod_ingles_desem': 'Nivel de Inglés',
    'fami_estratovivienda': 'Estrato Familiar de la Vivienda'
}

# Función para extraer número del estrato
def extraer_estrato(valor):
    if pd.isna(valor):
        return None
    match = re.search(r'\d+', str(valor))
    return int(match.group()) if match else None

# Lista normalizada para excluir departamentos
excluir_departamentos = ['extranjero', 'denver', 'quebec', 'zaragoza', 'paris', 'sao paulo']

# Crear figura 2x2 más amplia
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Gráficos de variables categoricas', fontsize=16)

# Graficar columnas categóricas
for i, columna in enumerate(columnas_categoricas):
    ax = axs[i // 2, i % 2]
    
    if columna == 'estu_depto_reside':
        conteo = df.loc[~df[columna].isin(excluir_departamentos), columna].value_counts(dropna=True)
        conteo = conteo.sort_values(ascending=True)
        conteo.plot(kind='barh', color='lightcoral', edgecolor='black', ax=ax)
        ax.set_xlabel('Frecuencia')
        ax.set_ylabel('Departamento')
    else:
        conteo = df[columna].value_counts(dropna=True)
        conteo.plot(kind='bar', color='lightgreen', edgecolor='black', ax=ax)
        ax.set_xlabel('Categoría')
        ax.set_ylabel('Frecuencia')
        ax.tick_params(axis='x', rotation=45)
    
    ax.set_title(titulos.get(columna, columna))

# Graficar histograma de estrato
estratos = df[columna_numerica_textual].apply(extraer_estrato).dropna()
ax = axs[1, 1]
ax.hist(estratos, bins=range(1, 8), align='left', color='skyblue', edgecolor='black', rwidth=0.8)
ax.set_title(titulos.get(columna_numerica_textual, columna_numerica_textual))
ax.set_xlabel('Estrato')
ax.set_ylabel('Frecuencia')
ax.set_xticks(range(1, 7))

# Ajustar espacios
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
