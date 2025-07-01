import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def entrenar_y_evaluar_modelos_arboles(ruta_excel):
    # Crear directorio para resultados
    os.makedirs('resultados', exist_ok=True)
    
    try:
        # Cargar datos
        df = pd.read_excel(ruta_excel)
        print("Datos cargados correctamente.")
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None

    # Columnas seleccionadas
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

    # Verificar columnas
    for col in columnas_seleccionadas:
        if col not in df.columns:
            print(f"Error: Columna '{col}' no encontrada.")
            return None

    df_seleccionado = df[columnas_seleccionadas].copy()

    # Limpieza y transformaciones
    df_seleccionado['periodo'] = pd.to_numeric(df_seleccionado['periodo'], errors='coerce')
    df_seleccionado.dropna(subset=['periodo', 'punt_global'], inplace=True)

    # Mapeo de variables categóricas
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
            df_seleccionado[col] = df_seleccionado[col].fillna(df_seleccionado[col].mean())
            print(f"Columna '{col}' mapeada a numérica.")

    # One-Hot Encoding
    columnas_binarias = [
        "estu_pagomatriculabeca", "estu_pagomatriculacredito",
        "estu_pagomatriculapadres", "estu_pagomatriculapropio",
        "fami_tieneinternet", "fami_tienecomputador", "fami_tienelavadora",
        "fami_tienehornomicroogas", "fami_tieneserviciotv",
        "fami_tieneautomovil", "fami_tienemotocicleta", "fami_tieneconsolavideojuegos"
    ]
    
    for col in columnas_binarias:
        df_seleccionado[col] = df_seleccionado[col].apply(lambda x: 1 if str(x).strip().lower() == 'si' else 0)
    
    # Winsorización (opcional)
    columnas_para_winsorizar = ['punt_global']
    for col in columnas_para_winsorizar:
        df_seleccionado[col] = winsorize(df_seleccionado[col], limits=[0.01, 0.01])
    
    # Crear categorías para punt_global (60 puntos por categoría)
    bins = [0, 60, 120, 180, 240, 300]
    labels = ['bajo', 'medio bajo', 'medio', 'medio alto', 'alto']
    df_seleccionado['categoria_punt_global'] = pd.cut(
        df_seleccionado['punt_global'], 
        bins=bins, 
        labels=labels,
        include_lowest=True
    )
    print("\nDistribución de categorías:")
    print(df_seleccionado['categoria_punt_global'].value_counts(normalize=True))
    
    # División de datos
    periodos_entrenamiento = [20233, 20234, 20225, 20226, 20231, 20232]
    periodos_prueba = [20222, 20223]

    train = df_seleccionado[df_seleccionado['periodo'].isin(periodos_entrenamiento)]
    test = df_seleccionado[df_seleccionado['periodo'].isin(periodos_prueba)]

    X_train = train.drop(columns=['punt_global', 'periodo', 'categoria_punt_global'])
    y_train = train['categoria_punt_global']
    X_test = test.drop(columns=['punt_global', 'periodo', 'categoria_punt_global'])
    y_test = test['categoria_punt_global']

    # Imputación de valores faltantes
    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Entrenamiento y evaluación de modelos (clasificación)
    resultados = {}
    
    # 1. Árbol de Decisión (Clasificación)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    resultados['Decision Tree'] = {'Accuracy': accuracy, 'Classification Report': report}
    print(f"\nDecision Tree - Accuracy: {accuracy:.4f}")
    print(report)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusión (Decision Tree)')
    plt.ylabel('Verdaderos')
    plt.xlabel('Predichos')
    plt.savefig('resultados/decision_tree_confusion_matrix.png')
    plt.close()
    
    # Importancia de características (Árbol)
    feat_importances = pd.Series(dt.feature_importances_, index=X_train.columns)
    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Características Importantes (Decision Tree)')
    plt.savefig('resultados/decision_tree_importances.png')
    plt.close()
    
    # 2. Random Forest (Clasificación)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    resultados['Random Forest'] = {'Accuracy': accuracy, 'Classification Report': report}
    print(f"\nRandom Forest - Accuracy: {accuracy:.4f}")
    print(report)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusión (Random Forest)')
    plt.ylabel('Verdaderos')
    plt.xlabel('Predichos')
    plt.savefig('resultados/random_forest_confusion_matrix.png')
    plt.close()

    # Importancia de características (Random Forest)
    feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Características Importantes (Random Forest)')
    plt.savefig('resultados/random_forest_importances.png')
    plt.close()

    # Optimización de Random Forest
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf_opt = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_grid_rf,
        n_iter=5,
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    rf_opt.fit(X_train, y_train)
    best_rf = rf_opt.best_estimator_
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    resultados['Optimized RF'] = {'Accuracy': accuracy, 'Classification Report': report}
    print(f"\nOptimized Random Forest - Accuracy: {accuracy:.4f}")
    print(f"Mejores parámetros: {rf_opt.best_params_}")
    print(report)

    # 3. XGBoost Optimizado (Clasificación)
    param_grid_xgb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Convertir etiquetas a números para XGBoost
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    xgb_opt = RandomizedSearchCV(
        estimator=XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        param_distributions=param_grid_xgb,
        n_iter=5,
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    xgb_opt.fit(X_train, y_train_encoded)
    best_xgb = xgb_opt.best_estimator_
    y_pred_encoded = best_xgb.predict(X_test)
    y_pred = le.inverse_transform(y_pred_encoded)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    resultados['Optimized XGBoost'] = {'Accuracy': accuracy, 'Classification Report': report}
    print(f"\nOptimized XGBoost - Accuracy: {accuracy:.4f}")
    print(f"Mejores parámetros: {xgb_opt.best_params_}")
    print(report)

    # Matriz de confusión (XGBoost)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusión (XGBoost)')
    plt.ylabel('Verdaderos')
    plt.xlabel('Predichos')
    plt.savefig('resultados/xgboost_confusion_matrix.png')
    plt.close()

    # Importancia de características (XGBoost)
    feat_importances = pd.Series(best_xgb.feature_importances_, index=X_train.columns)
    plt.figure(figsize=(10, 6))
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Características Importantes (XGBoost)')
    plt.savefig('resultados/xgboost_importances.png')
    plt.close()

    # Resumen de resultados
    print("\nResumen de Resultados:")
    for modelo, metrics in resultados.items():
        print(f"{modelo}: Accuracy = {metrics['Accuracy']:.4f}")

    # Guardar resultados en archivo
    with open('resultados/resumen_modelos.txt', 'w') as f:
        f.write("RESUMEN DE RESULTADOS DE CLASIFICACIÓN\n")
        f.write("====================================\n\n")
        f.write("Categorías de puntaje global:\n")
        f.write(" - Bajo: 0-60\n")
        f.write(" - Medio bajo: 61-120\n")
        f.write(" - Medio: 121-180\n")
        f.write(" - Medio alto: 181-240\n")
        f.write(" - Alto: 241-300\n\n")
        
        f.write("Distribución de categorías en los datos:\n")
        f.write(str(df_seleccionado['categoria_punt_global'].value_counts(normalize=True)) + "\n\n")
        
        for modelo, metrics in resultados.items():
            f.write(f"-------------------- {modelo} --------------------\n")
            f.write(f"Accuracy: {metrics['Accuracy']:.4f}\n")
            f.write("Classification Report:\n")
            f.write(metrics['Classification Report'] + "\n\n")
        
        f.write("\nMEJORES PARÁMETROS:\n")
        f.write(f"Random Forest: {rf_opt.best_params_}\n")
        f.write(f"XGBoost: {xgb_opt.best_params_}\n")
    
    # Guardar dataset procesado
    df_seleccionado.to_csv('resultados/dataset_procesado.csv', index=False)
    
    return resultados

# Ejecutar función
if __name__ == "__main__":
    ruta_excel = 'SaberProFiltered.xlsx'
    resultados = entrenar_y_evaluar_modelos_arboles(ruta_excel)