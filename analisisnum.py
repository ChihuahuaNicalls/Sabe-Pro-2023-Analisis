import pandas as pd

# Cargar el archivo Excel (ajusta el nombre si es necesario)
df = pd.read_excel("C:/Users/Sebas/Desktop/Probabilidad_datos/SaberProFiltered.xlsx", engine="openpyxl")

# Columnas a analizar
columnas = [
    'estu_cod_reside_depto',
    'estu_cod_reside_mcpio',
    'inst_cod_institucion',
    'estu_snies_prgmacademico',
    'mod_razona_cuantitat_punt',
    'mod_razona_cuantitat_desem',
    'mod_razona_cuantitativo_pnal',
    'mod_razona_cuantitativo_pnbc',
    'mod_lectura_critica_punt',
    'mod_lectura_critica_desem',
    'mod_lectura_critica_pnal',
    'mod_lectura_critica_pnbc',
    'mod_competen_ciudada_punt',
    'mod_competen_ciudada_desem',
    'mod_competen_ciudada_pnal',
    'mod_competen_ciudada_pnbc',
    'mod_ingles_punt',
    'mod_ingles_desem',
    'mod_ingles_pnal',
    'mod_ingles_pnbc',
    'mod_comuni_escrita_punt',
    'mod_comuni_escrita_desem',
    'mod_comuni_escrita_pnal',
    'mod_comuni_escrita_pnbc',
    'punt_global',
    'percentil_global',
    'percentil_nbc'
]


# Seleccionar solo esas columnas
df_sel = df[columnas]

# Convertir a numérico (coerce fuerza a NaN los valores no numéricos)
df_sel = df_sel.apply(pd.to_numeric, errors='coerce')

# Crear estadísticas básicas (describe ya incluye: count, mean, std, min, 25%, 50%, 75%, max)
resumen = df_sel.describe(percentiles=[0.25, 0.5, 0.75]).T

# Agregar la moda
moda = df_sel.mode().iloc[0]  # Solo la primera moda si hay varias
resumen['Moda'] = moda

# Agregar conteo de valores nulos por columna
resumen['Valores nulos'] = df_sel.isnull().sum()

# Guardar el resumen en un archivo Excel
resumen.to_excel("resumen_estadistico_completo.xlsx")

print("✅ Análisis completado. Archivo generado: resumen_estadistico_completo.xlsx")
