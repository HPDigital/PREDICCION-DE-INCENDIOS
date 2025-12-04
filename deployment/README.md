# 🚀 Deployment - Sistema de Predicción de Incendios Forestales

Este directorio contiene todos los componentes necesarios para desplegar el sistema en diferentes entornos.

---

## 📁 Contenido

### Scripts Principales

#### 1. `api.py` - API REST
API REST completa con Flask para realizar predicciones.

**Características:**
- 5 endpoints RESTful
- CORS habilitado
- Logging completo
- Manejo robusto de errores
- Health checks automáticos

**Uso:**
```bash
python deployment/api.py
```

**Endpoints:**
```
GET  /               # Información del servicio
GET  /health         # Estado del sistema
POST /predict        # Predicción individual
POST /predict/batch  # Predicción por lotes
GET  /stats          # Estadísticas
```

**Ejemplo de solicitud:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperatura": 35.0,
    "humedad": 30.0,
    "velocidad_viento": 25.0,
    "precipitacion": 0.0,
    "indice_sequedad": 75.0,
    "vegetacion_seca": 0.8,
    "mes": 7,
    "dia_semana": 3
  }'
```

---

#### 2. `dashboard.py` - Dashboard Interactivo
Dashboard web completo con Streamlit.

**Páginas:**
- 🔮 Predicción Individual
- 📦 Predicción por Lotes
- 📊 Estadísticas del Sistema

**Uso:**
```bash
streamlit run deployment/dashboard.py
```

**URL:** http://localhost:8501

**Características:**
- Sliders interactivos
- Visualizaciones en tiempo real
- Upload de archivos CSV
- Exportación de resultados
- Gráficos con Plotly

---

#### 3. `monitor.py` - Monitor en Tiempo Real
Sistema de monitoreo continuo con simulación de sensores.

**Características:**
- Simulación de datos meteorológicos
- Predicciones automáticas
- Sistema de alertas
- Dashboard temporal
- Estadísticas de uptime

**Uso:**
```bash
python deployment/monitor.py
```

**Parámetros configurables:**
```python
monitor = RealTimeMonitor(
    predictor=predictor,
    max_history=100,        # Historial de observaciones
    alert_threshold=0.7     # Umbral de alerta
)

# Ejecutar por 60 segundos, lecturas cada 5 segundos
monitor.run_continuous(interval=5, duration=60)
```

---

#### 4. `setup.py` - Script de Configuración
Gestor automatizado de despliegue y configuración.

**Comandos:**

```bash
# Configuración completa inicial
python deployment/setup.py setup

# Ejecutar tests
python deployment/setup.py test

# Construir imágenes Docker
python deployment/setup.py build

# Desplegar servicios
python deployment/setup.py deploy

# Detener servicios
python deployment/setup.py stop

# Ver logs en tiempo real
python deployment/setup.py logs

# Ver logs de un servicio específico
python deployment/setup.py logs --service api

# Despliegue completo automatizado
python deployment/setup.py full
```

**Con dependencias de desarrollo:**
```bash
python deployment/setup.py setup --dev
```

---

## 🐳 Docker

### Archivos de Configuración

#### `Dockerfile` - Imagen de la API
Imagen base para la API REST.

**Construcción:**
```bash
docker build -t fire-prediction-api -f Dockerfile .
```

**Ejecución:**
```bash
docker run -p 5000:5000 \
  -v ./models:/app/models:ro \
  -v ./results:/app/results \
  fire-prediction-api
```

---

#### `Dockerfile.streamlit` - Imagen del Dashboard
Imagen optimizada para Streamlit.

**Construcción:**
```bash
docker build -t fire-prediction-dashboard -f Dockerfile.streamlit .
```

**Ejecución:**
```bash
docker run -p 8501:8501 \
  -v ./models:/app/models:ro \
  fire-prediction-dashboard
```

---

#### `docker-compose.yml` - Orquestación
Orquesta todos los servicios del sistema.

**Servicios incluidos:**
1. **api** - API REST (puerto 5000)
2. **dashboard** - Dashboard Streamlit (puerto 8501)
3. **monitor** - Monitor en tiempo real

**Uso:**
```bash
# Iniciar todos los servicios
docker-compose up -d

# Ver estado
docker-compose ps

# Ver logs
docker-compose logs -f

# Ver logs de un servicio
docker-compose logs -f api

# Detener servicios
docker-compose down

# Reconstruir imágenes
docker-compose build

# Reiniciar un servicio
docker-compose restart api
```

---

## ⚙️ Configuración

### Variables de Entorno

Crear archivo `.env` basado en `.env.example`:

```bash
# Configuración de Flask
FLASK_ENV=production
API_HOST=0.0.0.0
API_PORT=5000

# Rutas de modelos
MODEL_PATH=./models/random_forest_model.pkl
SCALER_PATH=./models/scaler.pkl

# Monitor
MONITOR_INTERVAL=60

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

---

## 📊 Monitoreo y Logs

### Estructura de Logs

```
logs/
├── api.log           # Logs de la API
├── dashboard.log     # Logs del dashboard
└── monitor.log       # Logs del monitor
```

### Ver Logs en Tiempo Real

**Logs de la API:**
```bash
# Local
tail -f logs/api.log

# Docker
docker-compose logs -f api
```

**Logs del Dashboard:**
```bash
# Local
tail -f logs/dashboard.log

# Docker
docker-compose logs -f dashboard
```

**Logs del Monitor:**
```bash
# Local
tail -f logs/monitor.log

# Docker
docker-compose logs -f monitor
```

---

## 🔍 Health Checks

### API Health Check

**Endpoint:**
```bash
curl http://localhost:5000/health
```

**Respuesta esperada:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-04T10:30:00.000000",
  "model_loaded": true,
  "total_predictions": 42
}
```

**Códigos de estado:**
- `200` - Sistema saludable
- `503` - Sistema no disponible

---

## 🚀 Guías de Despliegue

### Despliegue Local (Desarrollo)

```bash
# 1. Entrenar modelo si no existe
python src/models/train_random_forest.py

# 2. Iniciar API
python deployment/api.py

# 3. En otra terminal, iniciar dashboard
streamlit run deployment/dashboard.py

# 4. Opcional: iniciar monitor
python deployment/monitor.py
```

---

### Despliegue con Docker (Recomendado)

```bash
# 1. Asegurar que el modelo existe
ls -la models/random_forest_model.pkl

# 2. Desplegar con setup script (automático)
python deployment/setup.py full

# O manualmente:
# 2a. Construir imágenes
docker-compose build

# 2b. Iniciar servicios
docker-compose up -d

# 3. Verificar estado
docker-compose ps

# 4. Ver logs
docker-compose logs -f
```

---

### Despliegue en Producción

**Requisitos previos:**
- Servidor Linux (Ubuntu 20.04+)
- Docker y Docker Compose instalados
- Puerto 5000 y 8501 disponibles
- Modelo entrenado disponible

**Pasos:**

1. **Clonar repositorio:**
```bash
git clone [repository-url]
cd INCENDIOS_FINAL
```

2. **Configurar variables de entorno:**
```bash
cp .env.example .env
# Editar .env según necesidades
```

3. **Desplegar:**
```bash
python deployment/setup.py full
```

4. **Configurar firewall:**
```bash
sudo ufw allow 5000/tcp   # API
sudo ufw allow 8501/tcp   # Dashboard
```

5. **Configurar NGINX (opcional):**
```nginx
server {
    listen 80;
    server_name api.ejemplo.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

server {
    listen 80;
    server_name dashboard.ejemplo.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

6. **Configurar SSL con Let's Encrypt:**
```bash
sudo certbot --nginx -d api.ejemplo.com
sudo certbot --nginx -d dashboard.ejemplo.com
```

---

## 🔧 Troubleshooting

### Problema: Modelo no encontrado

**Error:**
```
⚠️ Error: No se encontró el modelo en ./models/random_forest_model.pkl
```

**Solución:**
```bash
python src/models/train_random_forest.py
```

---

### Problema: Puerto ya en uso

**Error:**
```
Address already in use
```

**Solución:**
```bash
# Encontrar proceso usando el puerto
lsof -i :5000

# Terminar proceso
kill -9 [PID]

# O cambiar puerto en .env
```

---

### Problema: Docker no puede acceder a modelos

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: './models/random_forest_model.pkl'
```

**Solución:**
Verificar que el volumen está correctamente montado:
```bash
docker-compose down
docker-compose up -d
docker-compose logs api
```

---

### Problema: API retorna 503

**Causa:** Sistema no inicializado correctamente

**Solución:**
```bash
# Ver logs
docker-compose logs api

# Reiniciar servicio
docker-compose restart api

# Verificar health check
curl http://localhost:5000/health
```

---

## 📈 Escalabilidad

### Múltiples Instancias de API

Editar `docker-compose.yml`:

```yaml
services:
  api:
    # ...
    deploy:
      replicas: 3
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
```

---

### Load Balancer con NGINX

`nginx.conf`:
```nginx
upstream api_backend {
    least_conn;
    server api:5000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 🔐 Seguridad

### Mejores Prácticas

1. **Variables de entorno sensibles:**
```bash
# Nunca commitear .env
# Usar secrets en producción
```

2. **Limitar acceso a la API:**
```python
# Implementar autenticación
# Rate limiting
# Validación de entrada
```

3. **HTTPS en producción:**
```bash
# Usar Let's Encrypt
# Configurar SSL/TLS
```

4. **Actualizar dependencias:**
```bash
pip list --outdated
pip install -U [paquete]
```

---

## 📊 Métricas y Monitoreo

### Prometheus + Grafana (Futuro)

```yaml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

---

## 📞 Soporte

Para problemas o preguntas:
- Revisar documentación completa en `/README.md`
- Ver ejemplos en `/notebooks/`
- Consultar `/RESUMEN_EJECUTIVO.md`

---

**Última actualización:** Diciembre 2024  
**Versión:** 1.0.0
