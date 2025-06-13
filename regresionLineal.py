import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Ruta del archivo
ruta = "SaberProFiltered.xlsx"

# Columnas deseadas, incluyendo las variables socioeconómicas y percentiles, excluyendo puntajes por asignatura
columnas_iniciales_deseadas = [
    "percentil_global",
    "punt_global",
    "periodo",
    "estu_valormatriculauniversidad", # Cuánto paga (categórica)
    "estu_pagomatriculabeca",         # De dónde se paga: Beca (binaria)
    "estu_pagomatriculacredito",      # De dónde se paga: Crédito (binaria)
    "estu_pagomatriculapadres",       # De dónde se paga: Padres (binaria)
    "estu_pagomatriculapropio",       # De dónde se paga: Propio (binaria)
    "fami_educacionpadre",            # Educación de los padres (categórica)
    "fami_educacionmadre",            # Educación de la madre (categórica)
    "estu_horassemanatrabaja",       # Horas que trabaja (numérica, pero puede ser object)
    "fami_estratovivienda",           # Estrato (categórica)
    "fami_tieneinternet",             # Conexión a internet (binaria)
    "fami_tienecomputador",           # Computador (binaria)
    "fami_tienelavadora",             # Lavadora (binaria)
    "fami_tienehornomicroogas",       # Horno microondas (binaria)
    "fami_tieneserviciotv",           # Servicio de TV (binaria)
    "fami_tieneautomovil",            # Vehículo (auto) (binaria)
    "fami_tienemotocicleta",          # Vehículo (moto) (binaria)
    "fami_tieneconsolavideojuegos",   # Consola de videojuegos (binaria)
]

try:
    df = pd.read_excel(ruta, usecols=columnas_iniciales_deseadas)
    print("DataFrame cargado exitosamente, primeros datos: ")
    print(df.head())
    print("\nInformación inicial del DataFrame:")
    df.info()
except FileNotFoundError:
    print(f"Error: El archivo '{ruta}' no fue encontrado.")
    exit()
except ValueError as e:
    print(f"Error al cargar las columnas: {e}")
    print(
        "Asegúrate de que todos los nombres en 'columnas_iniciales_deseadas' existan en tu archivo Excel y estén escritos exactamente igual."
    )
    exit()

### Preprocesamiento de Datos

print("\n--- Preprocesamiento de Datos ---")

# Identificar tipos de columnas para un manejo adecuado
# Estas columnas categóricas se combinarán antes de One-Hot Encoding
categorical_cols_to_combine = [
    "estu_valormatriculauniversidad",
    "fami_educacionpadre",
    "fami_educacionmadre",
    "fami_estratovivienda",
]

# Las columnas binarias 'Si'/'No' se mapearán a 1/0
binary_cols = [
    "estu_pagomatriculabeca",
    "estu_pagomatriculacredito",
    "estu_pagomatriculapadres",
    "estu_pagomatriculapropio",
    "fami_tieneinternet",
    "fami_tienecomputador",
    "fami_tienelavadora",
    "fami_tienehornomicroogas",
    "fami_tieneserviciotv",
    "fami_tieneautomovil",
    "fami_tienemotocicleta",
    "fami_tieneconsolavideojuegos",
]

# Columnas que deberían ser numéricas pero podrían cargarse como 'object'
numeric_might_be_object_cols = [
    "estu_horassemanatrabaja",
]

# Convertir columnas que deben ser numéricas (y no son categóricas ni binarias)
for col in ["punt_global", "percentil_global", "percentil_nbc"] + numeric_might_be_object_cols:
    if col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"Columna '{col}' ya es numérica ({df[col].dtype}).")
        else:
            print(f"La columna '{col}' no es numérica ({df[col].dtype}). Intentando convertir a numérica...")
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isnull().any():
                print(f"  Advertencia: Después de la conversión, la columna '{col}' ahora tiene valores nulos.")
            else:
                print(f"  Columna '{col}' convertida a numérica exitosamente.")

# Manejo de la columna 'periodo'
if "periodo" in df.columns:
    if not pd.api.types.is_integer_dtype(df["periodo"]):
        print(f"La columna 'periodo' no es de tipo entero ({df['periodo'].dtype}). Intentando convertir a entero...")
        df["periodo"] = pd.to_numeric(df["periodo"], errors="coerce").astype(pd.Int64Dtype())
        if df["periodo"].isnull().any():
            print(f"  Advertencia: Después de la conversión, la columna 'periodo' ahora tiene valores nulos.")
        else:
            print(f"  Columna 'periodo' convertida a entero exitosamente.")
    else:
        print(f"Columna 'periodo' ya es de tipo entero ({df['periodo'].dtype}).")

# Mapear las columnas binarias 'Si'/'No' a 1/0 y manejar nulos
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].map({'si': 1, 'no': 0, 'otro': np.nan, 'no aplica': np.nan})
        if df[col].isnull().any():
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
                print(f"Columna '{col}' mapeada a 1/0 y nulos rellenados con la moda.")
            else:
                df[col] = df[col].fillna(0)
                print(f"Columna '{col}' mapeada a 1/0 y nulos rellenados con 0 (moda no disponible).")
    else:
        print(f"Advertencia: La columna binaria '{col}' no se encontró en el DataFrame.")


### Combinación de Categorías para variables específicas

print("\n--- Combinando Categorías en Variables Socioeconómicas ---")

# estu_valormatriculauniversidad
if "estu_valormatriculauniversidad" in df.columns:
    print("Combinando categorías para 'estu_valormatriculauniversidad'...")
    df['estu_valormatriculauniversidad'] = df['estu_valormatriculauniversidad'].astype(str).str.strip().str.lower()
    mapping_matricula = {
        'no pagó matrícula': 'No pagó',
        'no pagó matricula': 'No pagó', # Considerar posibles inconsistencias
        'menos de 500 mil': 'Menos de 1 millón',
        'entre 500 mil y menos de 1 millón': 'Menos de 1 millón',
        'entre 1 millón y menos de 2.5 millones': 'Entre 1 y 4 millones',
        'entre 2.5 millones y menos de 4 millones': 'Entre 1 y 4 millones',
        'entre 4 millones y menos de 5.5 millones': 'Más de 4 millones',
        'entre 5.5 millones y menos de 7 millones': 'Más de 4 millones',
        'más de 7 millones': 'Más de 4 millones',
        'nan': np.nan # Asegurar que los nulos sigan siendo nulos para la imputación
    }
    df['estu_valormatriculauniversidad'] = df['estu_valormatriculauniversidad'].replace(mapping_matricula)
    print(f"Nuevas categorías para 'estu_valormatriculauniversidad': {df['estu_valormatriculauniversidad'].unique()}")
else:
    print("Columna 'estu_valormatriculauniversidad' no encontrada para combinación.")

# fami_educacionpadre y fami_educacionmadre
education_mapping = {
    'no aplica': 'No aplica', # 'No aplica' para quienes no tienen padre/madre
    'no sabe': 'No sabe',
    'ninguno': 'Ninguno/Primaria',
    'primaria incompleta': 'Ninguno/Primaria',
    'primaria completa': 'Ninguno/Primaria',
    'secundaria incompleta': 'Secundaria/Media',
    'secundaria completa': 'Secundaria/Media',
    'técnica o tecnológica incompleta': 'Técnica/Tecnológica',
    'técnica o tecnológica completa': 'Técnica/Tecnológica',
    'universitaria incompleta': 'Universitaria/Posgrado',
    'universitaria completa': 'Universitaria/Posgrado',
    'posgrado': 'Universitaria/Posgrado',
    'nan': np.nan # Asegurar que los nulos sigan siendo nulos para la imputación
}

for col in ["fami_educacionpadre", "fami_educacionmadre"]:
    if col in df.columns:
        print(f"Combinando categorías para '{col}'...")
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace(education_mapping)
        print(f"Nuevas categorías para '{col}': {df[col].unique()}")
    else:
        print(f"Columna '{col}' no encontrada para combinación.")

# fami_estratovivienda
if "fami_estratovivienda" in df.columns:
    print("Combinando categorías para 'fami_estratovivienda'...")
    df['fami_estratovivienda'] = df['fami_estratovivienda'].astype(str).str.strip().str.lower()
    estrato_mapping = {
        'estrato 1': 'Estratos 1-2',
        'estrato 2': 'Estratos 1-2',
        'estrato 3': 'Estratos 3-4',
        'estrato 4': 'Estratos 3-4',
        'estrato 5': 'Estratos 5-6',
        'estrato 6': 'Estratos 5-6',
        'nan': np.nan # Asegurar que los nulos sigan siendo nulos para la imputación
    }
    df['fami_estratovivienda'] = df['fami_estratovivienda'].replace(estrato_mapping)
    # También maneja el caso de 'Sin Estrato' o similares si aparecen
    df['fami_estratovivienda'] = df['fami_estratovivienda'].replace({'sin estrato': 'Sin Estrato', 'zona rural': 'Zona Rural'})
    print(f"Nuevas categorías para 'fami_estratovivienda': {df['fami_estratovivienda'].unique()}")
else:
    print("Columna 'fami_estratovivienda' no encontrada para combinación.")

# Las columnas categóricas para One-Hot Encoding ahora usan las combinadas
categorical_cols_for_ohe = [
    "estu_valormatriculauniversidad",
    "fami_educacionpadre",
    "fami_educacionmadre",
    "fami_estratovivienda",
]

# --- IMPUTACIÓN DE NULOS EN COLUMNAS NUMÉRICAS Y CATEGÓRICAS ANTES DE OHE ---
print("\n--- Imputando Nulos después de la combinación de categorías ---")

# Imputar nulos en 'estu_horassemanatrabaja' con la mediana
if 'estu_horassemanatrabaja' in df.columns and df['estu_horassemanatrabaja'].isnull().any():
    median_hours = df['estu_horassemanatrabaja'].median()
    df['estu_horassemanatrabaja'] = df['estu_horassemanatrabaja'].fillna(median_hours)
    print(f"Columna 'estu_horassemanatrabaja': Nulos imputados con la mediana ({median_hours}).")

# Imputar nulos en columnas categóricas (ahora con las categorías combinadas) con la moda
for col in categorical_cols_for_ohe:
    if col in df.columns and df[col].isnull().any():
        if not df[col].mode().empty:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"Columna '{col}': Nulos imputados con la moda ('{mode_val}').")
        else:
            print(f"Advertencia: Columna '{col}' tiene todos sus valores nulos después de la combinación y no se puede imputar con la moda. Se mantendrán los nulos para el One-Hot Encoder (se manejará con 'handle_unknown').")
            # Para estas columnas, 'handle_unknown='ignore'' en OneHotEncoder significa que los nulos resultantes se tratarán como una categoría desconocida.
            # Sin embargo, es mejor imputarlos para evitar problemas. Si un column tiene todos los valores nulos, OneHotEncoder puede fallar o crear una columna de ceros.
            # Una opción sería imputar con 'Desconocido' o eliminar la columna si es el caso.
            # Por ahora, simplemente dejamos los nulos si mode() está vacío, y OneHotEncoder los manejará si la columna no es completamente nula.

# Imputar nulos en 'percentil_global' y 'percentil_nbc' si existen
for col in ["percentil_global", "percentil_nbc"]:
    if col in df.columns and df[col].isnull().any():
        median_percentil = df[col].median()
        df[col] = df[col].fillna(median_percentil)
        print(f"Columna '{col}': Nulos imputados con la mediana ({median_percentil:.2f}).")


# Seleccionar solo las columnas que realmente existen en el DataFrame para One-Hot Encoding
categorical_cols_existing = [col for col in categorical_cols_for_ohe if col in df.columns]

# Manejo de One-Hot Encoding para las columnas categóricas
if categorical_cols_existing:
    print(f"\nAplicando One-Hot Encoding a: {categorical_cols_existing}")
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols_existing)
        ],
        remainder='passthrough'
    )

    temp_df = df.drop(columns=["punt_global", "periodo"], errors='ignore')
    X_processed = preprocessor.fit_transform(temp_df)
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols_existing)
    passthrough_cols = [col for col in temp_df.columns if col not in categorical_cols_existing and col in df.columns]

    new_column_names = list(ohe_feature_names) + passthrough_cols
    df_encoded = pd.DataFrame(X_processed, columns=new_column_names, index=df.index)

    for col in ["punt_global", "periodo"]:
        if col in df.columns:
            df_encoded[col] = df[col]
    df = df_encoded
    print(f"DataFrame después de One-Hot Encoding. Nuevas dimensiones: {df.shape}")
    print(df.head())
else:
    print("\nNo se encontraron columnas categóricas para One-Hot Encoding en el DataFrame.")

# Comprobar nulos después de la codificación y mapeo
original_rows = df.shape[0]
df.dropna(inplace=True)
rows_after_dropna = df.shape[0]
print(
    f"\nFilas originales: {original_rows}, Filas después de eliminar nulos: {rows_after_dropna} ---"
)
if df.empty:
    print("Error: DataFrame vacío después de eliminar nulos. ¡Esto no debería pasar si la imputación funcionó correctamente!")
    exit()

# Última verificación de nulos
if df.isnull().sum().sum() > 0:
    print("Error: Nulos presentes después de la limpieza final. Detalles:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    exit()
else:
    print("No nulos después del preprocesamiento.")

### Análisis de Correlación

print("\n--- Análisis de Correlación ---")
df_for_corr = df.drop(columns=["periodo"], errors="ignore")

print("Matriz de Correlación de Variables Numéricas (excluyendo 'periodo')")
matriz_correlacion = df_for_corr.corr(numeric_only=True)

plt.figure(figsize=(16, 14))
sns.heatmap(
    matriz_correlacion,
    annot=False,
    cmap="Blues",
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"label": "Coeficiente de Correlación"},
)
plt.title("Matriz de Correlación de Variables Numéricas", fontsize=16)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.show()

variable_dependiente = "punt_global"
if variable_dependiente not in matriz_correlacion.columns:
    print(f"Error: La variable dependiente '{variable_dependiente}' no se encuentra en la matriz de correlación.")
    exit()

correlaciones_con_dependiente = (
    matriz_correlacion[variable_dependiente].abs().sort_values(ascending=False)
)
correlaciones_con_dependiente = correlaciones_con_dependiente[
    correlaciones_con_dependiente.index != variable_dependiente
]

if len(correlaciones_con_dependiente) >= 5:
    top_n_variables = correlaciones_con_dependiente.head(10).index.tolist()
    print(
        f"Las {len(top_n_variables)} variables independientes más correlacionadas con '{variable_dependiente}' son: {top_n_variables}"
    )
    print("Sus correlaciones son:")
    for var in top_n_variables:
        print(f"  {var}: {matriz_correlacion.loc[variable_dependiente, var]:.4f}")
elif len(correlaciones_con_dependiente) > 0:
    top_n_variables = correlaciones_con_dependiente.index.tolist()
    print(
        f"Menos de 5 variables más correlacionadas, se seleccionarán todas las variables independientes correlacionadas: {top_n_variables}"
    )
    print("Sus correlaciones son:")
    for var in top_n_variables:
        print(f"  {var}: {matriz_correlacion.loc[variable_dependiente, var]:.4f}")
else:
    print("Error: No se encontraron variables con correlación para el modelo.")
    exit()

variables_independientes_seleccionadas = top_n_variables

X = df[variables_independientes_seleccionadas]
y = df[variable_dependiente]

print(f"\nVariable dependiente '{variable_dependiente}' tipo de dato: {y.dtype}")
print(f"Variables independientes (X) seleccionadas: {X.columns.tolist()}")
print(f"Dimensiones de X: {X.shape}, Dimensiones de y: {y.shape}")


### División de Datos

print("\n--- División de Datos ---")
periodos_entrenamiento = [20222, 20223, 20225, 20226, 20231, 20232]
periodos_prueba = [20233, 20234]

df_train = df[df["periodo"].isin(periodos_entrenamiento)]
df_test = df[df["periodo"].isin(periodos_prueba)]

if df_train.empty:
    print(
        f"Error: No se encontraron datos para los periodos de entrenamiento: {periodos_entrenamiento}."
    )
    exit()
if df_test.empty:
    print(
        f"Error: No se encontraron datos para los periodos de prueba: {periodos_prueba}."
    )
    exit()

X_train = df_train[variables_independientes_seleccionadas]
y_train = df_train[variable_dependiente]

X_test = df_test[variables_independientes_seleccionadas]
y_test = df_test[variable_dependiente]

print(
    f"Tamaño del conjunto de entrenamiento X_train: {X_train.shape} (Periodos: {periodos_entrenamiento})"
)
print(
    f"Tamaño del conjunto de prueba X_test: {X_test.shape} (Periodos: {periodos_prueba})"
)
print(f"Tamaño del conjunto de entrenamiento y_train: {y_train.shape}")
print(f"Tamaño del conjunto de prueba y_test: {y_test.shape}")

total_rows = df.shape[0]
train_percentage = (X_train.shape[0] / total_rows) * 100
test_percentage = (X_test.shape[0] / total_rows) * 100
print(f"Porcentaje de datos en entrenamiento: {train_percentage:.2f}%")
print(f"Porcentaje de datos en prueba: {test_percentage:.2f}%")

if X_train.isnull().sum().sum() > 0:
    print("\nError: Nulos presentes en X_train después de la división")
    print(X_train.isnull().sum()[X_train.isnull().sum() > 0])
    exit()
if y_train.isnull().sum() > 0:
    print("\nError: Nulos presentes en y_train después de la división")
    print(y_train.isnull().sum())
    exit()
if X_test.isnull().sum().sum() > 0:
    print("\nError: Nulos presentes en X_test después de la división")
    print(X_test.isnull().sum()[X_test.isnull().sum() > 0])
    exit()
if y_test.isnull().sum() > 0:
    print("\nError: Nulos presentes en y_test después de la división")
    print(y_test.isnull().sum())
    exit()

### Modelo de Regresión Lineal Múltiple

print("\n--- Modelo de Regresión Lineal Múltiple ---")
modelo_rl = LinearRegression()
modelo_rl.fit(X_train, y_train)

y_pred = modelo_rl.predict(X_test)

print(f"\nCoeficientes (modelo_rl.coef_): {modelo_rl.coef_}")
print(f"Intercepto (modelo_rl.intercept_): {modelo_rl.intercept_}")

ecuacion = f"ŷ = {modelo_rl.intercept_:.2f}"
for i, col in enumerate(X_train.columns):
    ecuacion += f" + {modelo_rl.coef_[i]:.4f} * {col}"
print(f"\nEcuación de regresión: {ecuacion}")


### Evaluación del Modelo

print("\n--- Evaluación del Modelo ---")
r_squared_test = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nPrecisión (R-cuadrado en el conjunto de prueba): {r_squared_test:.4f}")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Raíz del Error Cuadratico Medio (RMSE): {rmse:.2f}")

### Gráficos de Diagnóstico

print("\n--- Gráficos de Diagnóstico ---")
residuals = y_test - y_pred
standardized_residuals = residuals / np.std(residuals)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_pred, y=standardized_residuals, alpha=0.6)
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Valores Ajustados (Predicciones)")
plt.ylabel("Residuos Estandarizados")
plt.title("Residuos Estandarizados vs. Valores Ajustados")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

plt.figure(figsize=(8, 6))
sm.qqplot(residuals, line="s", fit=True, ax=plt.gca())
plt.title("Gráfica de Cuantiles Normales de los Residuos")
plt.xlabel("Cuantiles Teóricos Normales")
plt.ylabel("Cuantiles de los Residuos Estandarizados")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()