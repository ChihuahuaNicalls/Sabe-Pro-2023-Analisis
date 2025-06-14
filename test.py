import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV # Añadimos RandomizedSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor

def entrenar_y_evaluar_modelo_regresion_completo(ruta_excel):
    try:
        #Cargar los datos desde la direccion del archivo Excel
        df = pd.read_excel(ruta_excel)
        print("Datos cargados correctamente.")
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        print("Asegúrate de que el enlace sea correcto y el archivo XLSX sea accesible.")
        return None

    # Columnas a utilizar en el análisis
    columnas_seleccionadas = [
        "punt_global",
        "periodo",
        "estu_valormatriculauniversidad",
        "estu_pagomatriculabeca",
        "estu_pagomatriculacredito",
        "estu_pagomatriculapadres",
        "estu_pagomatriculapropio",
        "fami_educacionpadre",
        "fami_educacionmadre",
        "estu_horassemanatrabaja",
        "fami_estratovivienda",
        "fami_tieneinternet",
        "fami_tienecomputador",
        "fami_tienelavadora",
        "fami_tienehornomicroogas",
        "fami_tieneserviciotv",
        "fami_tieneautomovil",
        "fami_tienemotocicleta",
        "fami_tieneconsolavideojuegos"
    ]

    # Verificar que las columnas seleccionadas existen en el DataFrame
    for col in columnas_seleccionadas:
        if col not in df.columns:
            print(f"Advertencia: La columna '{col}' no se encontró en el archivo Excel. Por favor, revisa los nombres de las columnas.")
            return None

    df_seleccionado = df[columnas_seleccionadas].copy()

    #Conversión de tipos de datos y limpieza
    df_seleccionado['periodo'] = pd.to_numeric(df_seleccionado['periodo'], errors='coerce')
    df_seleccionado.dropna(subset=['periodo', 'punt_global'], inplace=True)
    if df_seleccionado.empty:
        print("Error: No quedan datos después de eliminar filas con valores nulos en 'periodo' o 'punt_global'.")
        return None

    #Mapeo de variables ordinales a numéricas
    mapeos_ordinales = {
        "estu_valormatriculauniversidad": {
            "No pagó matricula": 0, "Menos de 500 mil": 1, "Entre 500 mil y menos de 1 millón": 2,
            "Entre 1 millón y menos de 2.5 millones": 3, "Entre 2.5 millones y menos de 4 millones": 4,
            "Entre 4 millones y menos de 5.5 millones": 5, "Entre 5.5 millones y menos de 7 millones": 6,
            "Más de 7 millones": 7
        },
        "fami_educacionpadre": {
            "No Aplica": 0, "No sabe": 0, "Ninguno": 0, "Primaria incompleta": 1, "Primaria completa": 2,
            "Secundaria (Bachillerato) incompleta": 3, "Secundaria (Bachillerato) completa": 4,
            "Técnica o tecnológica incompleta": 5, "Técnica o tecnológica completa": 6,
            "Educación profesional incompleta": 7, "Educación profesional completa": 8, "Postgrado": 9
        },
        "fami_educacionmadre": {
            "No Aplica": 0, "No sabe": 0, "Ninguno": 0, "Primaria incompleta": 1, "Primaria completa": 2,
            "Secundaria (Bachillerato) incompleta": 3, "Secundaria (Bachillerato) completa": 4,
            "Técnica o tecnológica incompleta": 5, "Técnica o tecnológica completa": 6,
            "Educación profesional incompleta": 7, "Educación profesional completa": 8, "Postgrado": 9
        },
        "estu_horassemanatrabaja": {
            "0": 0, "Menos de 10 horas": 1, "Entre 11 y 20 horas": 2,
            "Entre 21 y 30 horas": 3, "Más de 30 horas": 4
        },
        "fami_estratovivienda": {
            "Estrato 1": 1, "Estrato 2": 2, "Estrato 3": 3,
            "Estrato 4": 4, "Estrato 5": 5, "Estrato 6": 6
        }
    }

    for col, mapping in mapeos_ordinales.items():
        if col in df_seleccionado.columns:
            df_seleccionado[col] = df_seleccionado[col].astype(str).str.strip().map(mapping)
            if df_seleccionado[col].isnull().any():
                df_seleccionado[col] = df_seleccionado[col].fillna(df_seleccionado[col].mean())
            df_seleccionado[col] = df_seleccionado[col].astype(float)
            print(f"Columna '{col}' mapeada a numérica.")

    #Mejoras utilizando ingenieria de características avanzada

    #Tratamiento de Outliers con Winsorización
    columnas_para_winsorizar = []
    for col in columnas_para_winsorizar:
        if col in df_seleccionado.columns:
            df_seleccionado[col] = pd.to_numeric(df_seleccionado[col], errors='coerce')
            df_seleccionado[col] = df_seleccionado[col].fillna(df_seleccionado[col].mean())
            
            df_seleccionado[col] = winsorize(df_seleccionado[col], limits=[0.01, 0.01])
            print(f"Columna '{col}' winsorizada (límites 1%-99%).")

    #Transformaciones No Lineales (Logarítmica)
    columnas_para_log_transformar = []
    for col in columnas_para_log_transformar:
        if col in df_seleccionado.columns:
            df_seleccionado[col] = np.log(df_seleccionado[col] + 1e-6)
            print(f"Columna '{col}' transformada logarítmicamente.")

    #Creacion de características polinómicas
    poly = PolynomialFeatures(degree=2, include_bias=False)
    numeric_cols_for_poly = [
        'estu_valormatriculauniversidad', 'fami_educacionpadre', 
        'fami_educacionmadre', 'estu_horassemanatrabaja', 'fami_estratovivienda'
    ]
    numeric_cols_for_poly = [col for col in numeric_cols_for_poly if col in df_seleccionado.columns]

    if numeric_cols_for_poly:
        for col in numeric_cols_for_poly:
            if df_seleccionado[col].isnull().any():
                df_seleccionado[col] = df_seleccionado[col].fillna(df_seleccionado[col].mean())

        df_poly_features = pd.DataFrame(poly.fit_transform(df_seleccionado[numeric_cols_for_poly]),
                                         columns=poly.get_feature_names_out(numeric_cols_for_poly),
                                         index=df_seleccionado.index)
        df_seleccionado = df_seleccionado.drop(columns=numeric_cols_for_poly, errors='ignore')
        df_seleccionado = pd.concat([df_seleccionado, df_poly_features], axis=1)
        print(f"Se crearon características polinómicas para: {numeric_cols_for_poly}.")
        print(f"Nuevas columnas polinómicas (muestra): {df_poly_features.columns.tolist()[:5]}")
    else:
        print("No se encontraron columnas adecuadas para características polinómicas.")


    #Filtrar por período para 80% entrenamiento y 20% prueba
    periodos_entrenamiento = [20233, 20234, 20225, 20226, 20231, 20232]
    periodos_prueba = [20222, 20223]

    df_entrenamiento = df_seleccionado[df_seleccionado['periodo'].isin(periodos_entrenamiento)].copy()
    df_prueba = df_seleccionado[df_seleccionado['periodo'].isin(periodos_prueba)].copy()

    if df_entrenamiento.empty or df_prueba.empty:
        print("Error")
        return None

    print(f"Filas para entrenamiento: {len(df_entrenamiento)}")
    print(f"Filas para prueba: {len(df_prueba)}")

    #Manejo de variables categóricas Binarias con One-Hot Encoding
    columnas_binarias_ohe = [
        "estu_pagomatriculabeca", "estu_pagomatriculacredito",
        "estu_pagomatriculapadres", "estu_pagomatriculapropio",
        "fami_tieneinternet", "fami_tienecomputador", "fami_tienelavadora",
        "fami_tienehornomicroogas", "fami_tieneserviciotv",
        "fami_tieneautomovil", "fami_tienemotocicleta", "fami_tieneconsolavideojuegos"
    ]

    df_entrenamiento_encoded = pd.get_dummies(df_entrenamiento, columns=columnas_binarias_ohe, drop_first=True, dtype=float)
    df_prueba_encoded = pd.get_dummies(df_prueba, columns=columnas_binarias_ohe, drop_first=True, dtype=float)

    #Alinear las columnas después del One-Hot Encoding y la creación de interacciones
    train_cols = set(df_entrenamiento_encoded.columns)
    test_cols = set(df_prueba_encoded.columns)

    missing_in_test = list(train_cols - test_cols)
    for col in missing_in_test:
        df_prueba_encoded[col] = 0.0

    missing_in_train = list(test_cols - train_cols)
    for col in missing_in_train:
        df_entrenamiento_encoded[col] = 0.0

    df_prueba_encoded = df_prueba_encoded[df_entrenamiento_encoded.columns]

    for df_proc in [df_entrenamiento_encoded, df_prueba_encoded]:
        for col in df_proc.columns:
            if df_proc[col].dtype == 'object':
                df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')

    X_train = df_entrenamiento_encoded.drop(columns=["punt_global", "periodo"], errors='ignore')
    y_train = df_entrenamiento_encoded["punt_global"]

    X_test = df_prueba_encoded.drop(columns=["punt_global", "periodo"], errors='ignore')
    y_test = df_prueba_encoded["punt_global"]

    #Imputación de valores nulos
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    y_train = y_train.fillna(y_train.mean())
    y_test = y_test.fillna(y_test.mean())

    X_train = X_train_imputed
    X_test = X_test_imputed

    if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
        print("Error")
        return None

    #Analisis de multicolinealidad con Variance Inflation Factor
    X_train_vif = sm.add_constant(X_train, has_constant='add')
    
    X_train_vif_numeric = X_train_vif.select_dtypes(include=np.number)
    
    X_train_vif_numeric = X_train_vif_numeric.dropna()

    vif_data = pd.DataFrame()
    vif_data["feature"] = X_train_vif_numeric.columns
    
    vif_data["VIF"] = [variance_inflation_factor(X_train_vif_numeric.values, i) for i in range(X_train_vif_numeric.shape[1])]

    print(vif_data.sort_values(by="VIF", ascending=False))

    resultados_modelos = {}

    #Modelo de Regresión Lineal Múltiple con características mejoradas
    modelo_sklearn_ols = LinearRegression()
    modelo_sklearn_ols.fit(X_train, y_train)
    y_pred_ols = modelo_sklearn_ols.predict(X_test)

    rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
    r2_ols = r2_score(y_test, y_pred_ols)
    resultados_modelos['OLS'] = {'R^2': r2_ols, 'RMSE': rmse_ols}

    print(f"Métricas del Modelo OLS en el conjunto de prueba:")
    print(f"Error Cuadrático Medio (RMSE): {rmse_ols:.2f}")
    print(f"Coeficiente de Determinación (R^2): {r2_ols:.2f}")

    #Obtener y mostrar la fórmula de regresión lineal
    intercept = modelo_sklearn_ols.intercept_
    coefficients = modelo_sklearn_ols.coef_
    feature_names = X_train.columns

    formula = f"punt_global = {intercept:.4f}"
    for i, coef in enumerate(coefficients):
        formula += f" + ({coef:.4f} * {feature_names[i]})"

    print(formula)

    #Modelo de Regresión Lasso
    lasso_model = Lasso(alpha=0.1, random_state=42)
    lasso_model.fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)

    rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
    r2_lasso = r2_score(y_test, y_pred_lasso)
    resultados_modelos['Lasso'] = {'R^2': r2_lasso, 'RMSE': rmse_lasso}

    print(f"Métricas del Modelo Lasso en el conjunto de prueba:")
    print(f"Error Cuadrático Medio (RMSE): {rmse_lasso:.2f}")
    print(f"Coeficiente de Determinación (R^2): {r2_lasso:.2f}")
    print(f"Número de características seleccionadas por Lasso: {np.sum(lasso_model.coef_!= 0)}")


    #Modelo XGBoost Regressor
    xgboost_model_base = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)
    xgboost_model_base.fit(X_train, y_train)
    y_pred_xgboost_base = xgboost_model_base.predict(X_test)

    rmse_xgboost_base = np.sqrt(mean_squared_error(y_test, y_pred_xgboost_base))
    r2_xgboost_base = r2_score(y_test, y_pred_xgboost_base)
    resultados_modelos['XGBoost_Base'] = {'R^2': r2_xgboost_base, 'RMSE': rmse_xgboost_base}

    print(f"Métricas del Modelo XGBoost (Base) en el conjunto de prueba:")
    print(f"Error Cuadrático Medio (RMSE): {rmse_xgboost_base:.2f}")
    print(f"Coeficiente de Determinación (R^2): {r2_xgboost_base:.2f}")

    #Ajuste de Hiperparámetros para XGBoost con RandomizedSearchCV
    param_grid_xgboost = {
        'n_estimators': [300, 500, 700, 1000, 1500],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.005, 0.01, 0.05],
        'reg_lambda': [0, 0.005, 0.01, 0.05]
    }
    try:
        random_search_xgboost = RandomizedSearchCV(estimator=XGBRegressor(random_state=42, n_jobs=-1),
                                                 param_distributions=param_grid_xgboost,
                                                 n_iter=100,
                                                 scoring='r2',
                                                 cv=3,
                                                 verbose=1,
                                                 random_state=42,
                                                 n_jobs=-1)
        random_search_xgboost.fit(X_train, y_train)
        
        print(f"\nMejores parámetros para XGBoost (RandomizedSearch): {random_search_xgboost.best_params_}")
        best_xgboost_model = random_search_xgboost.best_estimator_
        y_pred_best_xgboost = best_xgboost_model.predict(X_test)
        r2_best_xgboost = r2_score(y_test, y_pred_best_xgboost)
        rmse_best_xgboost = np.sqrt(mean_squared_error(y_test, y_pred_best_xgboost))
        resultados_modelos['XGBoost_Optimized'] = {'R^2': r2_best_xgboost, 'RMSE': rmse_best_xgboost}
        print(f"R^2 del Modelo XGBoost: {r2_best_xgboost:.2f}")
        print(f"RMSE del Modelo XGBoost: {rmse_best_xgboost:.2f}")
    except Exception as e:
        print(f"\nError durante RandomizedSearchCV para XGBoost: {e}")

    #Graficas del modelo
    X_train_sm = sm.add_constant(X_train, has_constant='add')
    
    X_train_sm = X_train_sm.astype(float)
    y_train_sm = y_train.astype(float)

    try:
        modelo_sm = sm.OLS(y_train_sm, X_train_sm).fit()
        print(modelo_sm.summary())

        #Gráfica de Residuos Estandarizados vs. Valores Ajustados
        residuals_standardized = modelo_sm.get_influence().resid_studentized_internal
        fitted_values = modelo_sm.fittedvalues

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=fitted_values, y=residuals_standardized, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Valores Ajustados')
        plt.ylabel('Residuos Estandarizados')
        plt.title('Gráfica de Residuos Estandarizados vs. Valores Ajustados')
        plt.grid(True)
        plt.show()

        #Gráfica de Cuantiles Normales (Q-Q Plot)
        plt.figure(figsize=(8, 6))
        stats.probplot(residuals_standardized, dist="norm", plot=plt)
        plt.title('Gráfica de Cuantiles Normales (Q-Q Plot) de Residuos')
        plt.show()

    except Exception as e:
        print(f"\nError al generar las gráficas de diagnóstico: {e}")

    for model_name, metrics_dict in resultados_modelos.items():
        print(f"Modelo {model_name}: R^2 = {metrics_dict['R^2']:.2f}, RMSE = {metrics_dict['RMSE']:.2f}")

    return resultados_modelos

ruta_excel = 'SaberProFiltered.xlsx'
resultados = entrenar_y_evaluar_modelo_regresion_completo(ruta_excel)

if resultados is None:
    print("\nEl análisis no pudo completarse")