# Usar imagen base de Python
FROM python:3.9-slim-bullseye

# Establecer variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos necesarios
COPY requirements.txt .
COPY arboles_decision.py .
COPY entrypoint.sh .

# Instalar dependencias del sistema y Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libgomp1 && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y gcc && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Dar permisos de ejecución al entrypoint
RUN chmod +x entrypoint.sh

# Punto de entrada
ENTRYPOINT ["./entrypoint.sh"]
