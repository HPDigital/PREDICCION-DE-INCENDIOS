# Imagen base de Python
FROM python:3.9-slim

# Información del mantenedor
LABEL maintainer="herwig@example.com"
LABEL description="API para predicción de incendios forestales"

# Establecer directorio de trabajo
WORKDIR /app

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar código de la aplicación
COPY src/ ./src/
COPY deployment/ ./deployment/
COPY models/ ./models/

# Crear directorios necesarios
RUN mkdir -p /app/results /app/logs

# Exponer puerto de la API
EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Comando por defecto
CMD ["python", "deployment/api.py"]
