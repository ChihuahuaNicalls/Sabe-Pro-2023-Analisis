#!/bin/bash

# Verificar archivo de datos
if [ ! -f "SaberProFiltered.xlsx" ]; then
    echo "ERROR: Archivo de datos no encontrado!"
    echo "Monta el archivo con:"
    echo "docker run -v $(pwd)/datos:/app -v $(pwd)/resultados:/app/resultados ..."
    exit 1
fi

# Ejecutar entrenamiento
echo "====================================="
echo "  INICIANDO ENTRENAMIENTO DE MODELOS"
echo "====================================="
python arboles_decision.py

# Mover resultados a volumen persistente
if [ -d "resultados" ]; then
    echo "Resultados guardados en: /app/resultados"
else
    echo "Error: No se generaron resultados"
    exit 1
fi