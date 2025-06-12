import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import scipy.stats as stats
# from sklearn.model_selection import train_test_split # Ya no se necesita esta importación

#Ruta del archivo
ruta = 'SaberProFiltered.xlsx'

#Columnas a analizar
columnas_iniciales_deseadas = [
    'percentil_global',
    'punt_global',
    'percentil_nbc',
    'mod_comuni_escrita_punt',
    'mod_ingles_punt',
    'mod_competen_ciudada_punt',
    'mod_lectura_critica_punt',
    'mod_razona_cuantitat_punt',
    'periodo'
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
    print("Asegúrate de que todos los nombres en 'columnas_iniciales_deseadas' existan en tu archivo Excel y estén escritos exactamente igual.")
    exit()

#Convertir a tipo entero si es necesario
for col in df.columns:
    if col != 'periodo' and not pd.api.types.is_numeric_dtype(df[col]):
        print(f"La columna '{col}' no es numérica ({df[col].dtype}). Intentando convertir a numérica...")
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().any():
            print(f"  Advertencia: Después de la conversión, la columna '{col}' ahora tiene valores nulos.")
    elif col == 'periodo' and not pd.api.types.is_integer_dtype(df[col]):
        print(f"La columna '{col}' no es de tipo entero ({df[col].dtype}). Intentando convertir a entero...")
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(pd.Int64Dtype())
        if df[col].isnull().any():
            print(f"  Advertencia: Después de la conversión, la columna '{col}' (periodo) ahora tiene valores nulos.")
    else:
        print(f"Columna '{col}' ya es de un tipo de dato apropiado ({df[col].dtype}).")

#Comprobar nulos
original_rows = df.shape[0]
df.dropna(inplace=True)
rows_after_dropna = df.shape[0]
print(f"\nFilas originales: {original_rows}, Filas después de eliminar nulos: {rows_after_dropna} ---")
if df.empty:
    print("Error")
    exit()

#Ultima verificación de nulos
if df.isnull().sum().sum() > 0:
    print("Error")
    print(df.isnull().sum())
    exit()
else:
    print("No nulos")

#Matriz de correlación
df_for_corr = df.drop(columns=['periodo'], errors='ignore')

print("Matriz de Correlación de Variables Numéricas (excluyendo 'periodo')")
matriz_correlacion = df_for_corr.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    matriz_correlacion,
    annot=True,
    cmap='Blues',
    fmt=".2f",
    linewidths=.5,
    cbar_kws={'label': 'Coeficiente de Correlación'}
)
plt.title('Matriz de Correlación de Variables Numéricas', fontsize=16)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

variable_dependiente = 'punt_global'
correlaciones_con_dependiente = matriz_correlacion[variable_dependiente].abs().sort_values(ascending=False)
correlaciones_con_dependiente = correlaciones_con_dependiente[correlaciones_con_dependiente.index != variable_dependiente]

if len(correlaciones_con_dependiente) >= 3:
    top_3_variables = correlaciones_con_dependiente.head(3).index.tolist()
    print(f"Las 3 variables independientes más correlacionadas con '{variable_dependiente}' son: {top_3_variables}")
    print("Sus correlaciones son:")
    for var in top_3_variables:
        print(f"  {var}: {matriz_correlacion.loc[variable_dependiente, var]:.4f}")
elif len(correlaciones_con_dependiente) > 0:
    top_3_variables = correlaciones_con_dependiente.index.tolist()
    print(f"Menos de 3 variables más correlacionadas, se seleccionarán todas las variables independientes correlacionadas: {top_3_variables}")
    print("Sus correlaciones son:")
    for var in top_3_variables:
        print(f"  {var}: {matriz_correlacion.loc[variable_dependiente, var]:.4f}")
else:
    print("Error")
    exit()

#Variables dependiente e independientes
variables_independientes_seleccionadas = top_3_variables

X = df[variables_independientes_seleccionadas]
y = df[variable_dependiente]

print(f"Variable dependiente '{variable_dependiente}' tipo de dato: {y.dtype}")
print(f"Variables independientes (X) seleccionadas: {X.columns.tolist()}")
print(f"Dimensiones de X: {X.shape}, Dimensiones de y: {y.shape}")


#Split datos de entrenamiento y prueba
# Nuevos periodos para entrenamiento y prueba
periodos_entrenamiento = [20222, 20223, 20225, 20226, 20231, 20232]
periodos_prueba = [20233, 20234]

df_train = df[df['periodo'].isin(periodos_entrenamiento)]
df_test = df[df['periodo'].isin(periodos_prueba)] # Usar isin para múltiples periodos de prueba

#Comprobar si los DataFrames de entrenamiento y prueba están vacíos
if df_train.empty:
    print(f"Error: No se encontraron datos para los periodos de entrenamiento: {periodos_entrenamiento}.")
    exit()
if df_test.empty:
    print(f"Error: No se encontraron datos para los periodos de prueba: {periodos_prueba}.")
    exit()

X_train = df_train[variables_independientes_seleccionadas]
y_train = df_train[variable_dependiente]

X_test = df_test[variables_independientes_seleccionadas]
y_test = df_test[variable_dependiente]

#Tamaños de splits
print(f"Tamaño del conjunto de entrenamiento X_train: {X_train.shape} (Periodos: {periodos_entrenamiento})")
print(f"Tamaño del conjunto de prueba X_test: {X_test.shape} (Periodos: {periodos_prueba})")
print(f"Tamaño del conjunto de entrenamiento y_train: {y_train.shape}")
print(f"Tamaño del conjunto de prueba y_test: {y_test.shape}")

#Calcular el porcentaje de datos en cada conjunto para verificar la proporción
total_rows = df.shape[0]
train_percentage = (X_train.shape[0] / total_rows) * 100
test_percentage = (X_test.shape[0] / total_rows) * 100
print(f"Porcentaje de datos en entrenamiento: {train_percentage:.2f}%")
print(f"Porcentaje de datos en prueba: {test_percentage:.2f}%")


# Verificación de nulos en los conjuntos de entrenamiento y prueba antes del modelado
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

#Modelo de Regresión Lineal Múltiple
modelo_rl = LinearRegression()
modelo_rl.fit(X_train, y_train)

y_pred = modelo_rl.predict(X_test)

print(f"\nCoeficientes (modelo_rl.coef_): {modelo_rl.coef_}")
print(f"Intercepto (modelo_rl.intercept_): {modelo_rl.intercept_}")

ecuacion = f"ŷ = {modelo_rl.intercept_:.2f}"
for i, col in enumerate(X_train.columns):
    ecuacion += f" + {modelo_rl.coef_[i]:.4f} * {col}"
print(f"\nEcuación de regresión: {ecuacion}")

#Datos resultantes
r_squared_test = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nPrecisión (R-cuadrado en el conjunto de prueba): {r_squared_test:.4f}")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Raíz del Error Cuadratico Medio (RMSE): {rmse:.2f}")

#Gráfica de Residuos Estandarizados Vs Valores ajustados del modelo de regresión lineal
residuals = y_test - y_pred
standardized_residuals = residuals / np.std(residuals)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_pred, y=standardized_residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valores Ajustados (Predicciones)')
plt.ylabel('Residuos Estandarizados')
plt.title('Residuos Estandarizados vs. Valores Ajustados')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

#Gráfica de Cuantiles Normales (Q-Q Plot) del modelo de Regresión Lineal (para residuos)
plt.figure(figsize=(8, 6))
sm.qqplot(residuals, line='s', fit=True, ax=plt.gca())
plt.title('Gráfica de Cuantiles Normales de los Residuos')
plt.xlabel('Cuantiles Teóricos Normales')
plt.ylabel('Cuantiles de los Residuos Estandarizados')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()