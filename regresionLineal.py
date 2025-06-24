import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import statsmodels.api as sm

#Ruta del archivo
ruta = "SaberProFiltered.xlsx"

#Categorias de las variables
columnas_iniciales_deseadas = [
    "punt_global",
    "periodo",
    "estu_valormatriculauniversidad", #Categorica
    "estu_pagomatriculabeca",          #Binaria
    "estu_pagomatriculacredito",       #Binaria
    "estu_pagomatriculapadres",        #Binaria
    "estu_pagomatriculapropio",        #Binaria
    "fami_educacionpadre",             #Categorica
    "fami_educacionmadre",             #Categorica
    "estu_horassemanatrabaja",         #Categorica
    "fami_estratovivienda",            #Categorica
    "fami_tieneinternet",              #Binaria
    "fami_tienecomputador",            #Binaria
    "fami_tienelavadora",              #Binaria
    "fami_tienehornomicroogas",        #Binaria
    "fami_tieneserviciotv",            #Binaria
    "fami_tieneautomovil",             #Binaria
    "fami_tienemotocicleta",           #Binaria
    "fami_tieneconsolavideojuegos",    #Binaria
]

try:
    df = pd.read_excel(ruta, usecols=columnas_iniciales_deseadas)
    print("DataFrame cargado exitosamente, primeros datos: ")
    print(df.head())
except FileNotFoundError:
    print("Error")
    exit()
except ValueError as e:
    print(f"Error al cargar las columnas: {e}")
    exit()


#Preprocesamiento de datos

#Puntaje global y periodo
if "punt_global" in df.columns:
    df["punt_global"] = pd.to_numeric(df["punt_global"], errors="coerce")
else:
    print(f"Advertencia: La columna 'punt_global' no se encontró en el DataFrame.")

if "periodo" in df.columns:
    df["periodo"] = pd.to_numeric(df["periodo"], errors="coerce")
    if df["periodo"].isnull().any():
        df["periodo"] = df["periodo"].fillna(df["periodo"].mode()[0])
    df["periodo"] = df["periodo"].astype(int)
else:
    print(f"Advertencia: La columna 'periodo' no se encontró en el DataFrame.")

#Columnas binarias
binary_cols = [
    "estu_pagomatriculabeca", "estu_pagomatriculacredito",
    "estu_pagomatriculapadres", "estu_pagomatriculapropio",
    "fami_tieneinternet", "fami_tienecomputador", "fami_tienelavadora",
    "fami_tienehornomicroogas", "fami_tieneserviciotv",
    "fami_tieneautomovil", "fami_tienemotocicleta",
    "fami_tieneconsolavideojuegos",
]

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].map({'si': 1, 'no': 0, 'otro': np.nan, 'no aplica': np.nan})
        if df[col].isnull().any():
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(0)
        df[col] = df[col].astype(int)
    else:
        print(f"Advertencia: La columna binaria '{col}' no se encontró en el DataFrame.")

#Columnas categoricas
education_order = {
    'no aplica': 0, 'no sabe': 0,
    'ninguno': 1,
    'primaria incompleta': 2,
    'primaria completa': 3,
    'secundaria incompleta': 4,
    'secundaria completa': 5,
    'técnica o tecnológica incompleta': 6,
    'técnica o tecnológica completa': 7,
    'universitaria incompleta': 8,
    'universitaria completa': 9,
    'postgrado': 10,
    np.nan: np.nan
}

for col in ["fami_educacionpadre", "fami_educacionmadre"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace('primaria incompleta', 'primaria incompleta')
        df[col] = df[col].map(education_order)
        df[col] = df[col].astype(float) 
    else:
        print(f"Columna '{col}' no encontrada para mapeo.")

hours_mapping = {
    '0': 0, 'cero': 0,
    'menos de 10 horas': 1,
    'entre 11 y 20 horas': 2,
    'entre 21 y 30 horas': 3,
    'más de 30 horas': 4,
    'nan': np.nan
}
if "estu_horassemanatrabaja" in df.columns:
    df['estu_horassemanatrabaja'] = df['estu_horassemanatrabaja'].astype(str).str.strip().str.lower()
    df['estu_horassemanatrabaja'] = df['estu_horassemanatrabaja'].map(hours_mapping)
    df['estu_horassemanatrabaja'] = df['estu_horassemanatrabaja'].astype(float)
else:
    print("Columna 'estu_horassemanatrabaja' no encontrada para mapeo.")

estrato_mapping = {
    'estrato 1': 1, 'estrato 2': 2, 'estrato 3': 3,
    'estrato 4': 4, 'estrato 5': 5, 'estrato 6': 6,
    'sin estrato': 0, 'zona rural': 0,
    'nan': np.nan
}

if "fami_estratovivienda" in df.columns:
    df['fami_estratovivienda'] = df['fami_estratovivienda'].astype(str).str.strip().str.lower()
    df['fami_estratovivienda'] = df['fami_estratovivienda'].map(estrato_mapping)
    df['fami_estratovivienda'] = df['fami_estratovivienda'].astype(float)
else:
    print("Columna 'fami_estratovivienda' no encontrada para mapeo.")

matricula_mapping = {
    'no pagó matrícula': 0, 'no pagó matricula': 0,
    'menos de 500 mil': 1,
    'entre 500 mil y menos de 1 millón': 2,
    'entre 1 millón y menos de 2.5 millones': 3,
    'entre 2.5 millones y menos de 4 millones': 4,
    'entre 4 millones y menos de 5.5 millones': 5,
    'entre 5.5 millones y menos de 7 millones': 6,
    'más de 7 millones': 7,
    'nan': np.nan
}
if "estu_valormatriculauniversidad" in df.columns:
    df['estu_valormatriculauniversidad'] = df['estu_valormatriculauniversidad'].astype(str).str.strip().str.lower()
    df['estu_valormatriculauniversidad'] = df['estu_valormatriculauniversidad'].map(matricula_mapping)
    df['estu_valormatriculauniversidad'] = df['estu_valormatriculauniversidad'].astype(float)
else:
    print("Columna 'estu_valormatriculauniversidad' no encontrada para mapeo.")

imputable_numeric_cols = [
    "punt_global",
    "estu_horassemanatrabaja",
    "fami_educacionpadre",
    "fami_educacionmadre",
    "fami_estratovivienda",
    "estu_valormatriculauniversidad"
]

for col in imputable_numeric_cols:
    if col in df.columns and df[col].isnull().any():
        median_val = df[col].median()
        if pd.isna(median_val):
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = df[col].fillna(median_val)
            print(f"Columna '{col}': Nulos imputados con la mediana ({median_val}).")
        if col != "punt_global":
            df[col] = df[col].astype(int)

#Eliminacion de nulos restantes
original_rows = df.shape[0]
if df.isnull().sum().sum() > 0:
    df.dropna(inplace=True)
rows_after_dropna = df.shape[0]

if df.empty:
    print("Error")
    exit()


#Analisis de correlacion
df_for_corr = df.drop(columns=["periodo"], errors="ignore")

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
elif len(correlaciones_con_dependiente) > 0:
    top_n_variables = correlaciones_con_dependiente.index.tolist()
else:
    print("Error: No se encontraron variables con correlación para el modelo.")
    exit()

#Division de datos
variables_independientes_seleccionadas = top_n_variables

X = df[variables_independientes_seleccionadas]
y = df[variable_dependiente]

print(f"\nVariable dependiente '{variable_dependiente}' tipo de dato: {y.dtype}")
print(f"Variables independientes (X) seleccionadas: {X.columns.tolist()}")

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

print(f"Tamaño del conjunto de entrenamiento: {X_train.shape} (Periodos: {periodos_entrenamiento})")
print(f"Tamaño del conjunto de prueba: {X_test.shape} (Periodos: {periodos_prueba})")
print(f"Tamaño del conjunto de entrenamiento: {y_train.shape}")
print(f"Tamaño del conjunto de prueba: {y_test.shape}")

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

#Regresion lineal multiple
modelo_rl = LinearRegression()
modelo_rl.fit(X_train, y_train)

y_pred = modelo_rl.predict(X_test)

print(f"\nCoeficientes: {modelo_rl.coef_}")
print(f"Intercepto: {modelo_rl.intercept_}")

ecuacion = f"ŷ = {modelo_rl.intercept_:.2f}"
for i, col in enumerate(X_train.columns):
    ecuacion += f" + {modelo_rl.coef_[i]:.4f} * {col}"
print(f"\nEcuación de regresión: {ecuacion}")

def categorize_score(score):
    if 0 <= score <= 60:
        return "Nivel Bajo"
    elif 60 < score <= 120:
        return "Medio Bajo"
    elif 120 < score <= 180:
        return "Nivel Medio"
    elif 180 < score <= 240:
        return "Medio Alto"
    elif 240 < score <= 300:
        return "Nivel Alto"
    else:
        return "Fuera de Rango"

y_pred_category = np.vectorize(categorize_score)(y_pred)
y_test_category = np.vectorize(categorize_score)(y_test)

#Evaluacion de precision por categoria
min_len = min(len(y_test_category), len(y_pred_category))
accuracy_category = accuracy_score(y_test_category[:min_len], y_pred_category[:min_len])

print(f"Porcentaje de exito basado en categoria: {accuracy_category:.4f}")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Raíz del Error Cuadratico Medio (RMSE): {rmse:.2f}")


#Graficos de diagnostico
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