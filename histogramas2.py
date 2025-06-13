import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("C:/Users/Sebas/Desktop/Probabilidad_datos/SaberProFiltered.xlsx")


cols = ['estu_pagomatriculabeca', 'estu_pagomatriculacredito', 'estu_pagomatriculapadres', 'estu_pagomatriculapropio']

titulos = {
    'estu_pagomatriculabeca': 'Pago de matrícula con beca',
    'estu_pagomatriculacredito': 'Pago de matrícula con crédito',
    'estu_pagomatriculapadres': 'Pago de matrícula por padres',
    'estu_pagomatriculapropio': 'Pago de matrícula propio'
}

# Limpiar y normalizar respuestas para cada columna
for col in cols:
    # Pasar a str para evitar errores y luego a minúsculas y strip
    df[col] = df[col].astype(str).str.strip().str.lower()
    # Opcional: Si hay valores raros, los convierto en 'no'
    df[col] = df[col].apply(lambda x: x if x in ['sí', 'si', 'no'] else 'no')
    # Unificar 'si' a 'sí'
    df[col] = df[col].replace('si', 'sí')

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, col in enumerate(cols):
    conteos = df[col].value_counts()
    conteos = conteos.reindex(['sí', 'no'], fill_value=0)
    conteos.plot(kind='bar', ax=axs[i], color=['green', 'red'])
    axs[i].set_title(titulos[col], fontsize=14)
    axs[i].set_ylabel('Cantidad', fontsize=12)
    axs[i].set_ylim(0, max(conteos) + 1)
    axs[i].set_xticklabels(['Sí', 'No'], rotation=0, fontsize=12)

plt.tight_layout()
plt.show()
