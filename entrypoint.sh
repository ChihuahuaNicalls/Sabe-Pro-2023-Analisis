#!/bin/bash

# Verificar si el archivo de datos existe
if [ ! -f "SaberProFiltered.xlsx" ]; then
    echo "ERROR: El archivo SaberProFiltered.xlsx no se encuentra en el directorio /app"
    echo "Por favor, monte el archivo usando:"
    echo "docker run -v /ruta/local/SaberProFiltered.xlsx:/app/SaberProFiltered.xlsx ..."
    exit 1
fi

# Ejecutar el script de Python
echo "=============================================="
echo "  INICIANDO ENTRENAMIENTO DE MODELOS"
echo "=============================================="
echo " - Dataset: SaberProFiltered.xlsx"
echo " - Modelos: Árboles de Decisión, Random Forest, XGBoost"
echo " - Tiempo estimado: 5-30 min (depende del hardware)"
echo "=============================================="

python arboles_decision.py

echo "=============================================="
echo "  PROCESO COMPLETADO"
echo "=============================================="
echo " Resultados guardados en: /app (dentro del contenedor)"
echo " Para obtener los gráficos:"
echo "  docker cp <nombre-contenedor>:/app/graficos/ ./"
echo "=============================================="