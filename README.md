# Prediccion de riesgo de incendios forestales (Cochabamba, Bolivia)

Proyecto listo para que cualquier usuario cargue sus propios datos, entrene un modelo binario (Incendio/No incendio) y genere un artefacto desplegable en Azure como API REST.

## Flujo de extremo a extremo
- **Datos**: meteorologia diaria (24 variables) + focos MODIS filtrados; guarda CSV en `data/processed/` con columna objetivo `Incendio`.
- **Entrenamiento**: `src/models/train_random_forest.py` entrena, evalua y guarda `models/fire_model.pkl` (modelo + scaler + lista de features).
- **Prediccion local**: `src/models/predict.py` carga el artefacto y expone metodos de prediccion y riesgo.
- **API**: `deployment/api.py` usa el mismo artefacto y valida contra `EXPECTED_FEATURES`.
- **Azure**: `deployment/deploy_to_azure.py` registra `fire_model.pkl` y despliega `deployment/score.py` en Azure Container Instances.

## Estructura minima
```
src/                # codigo principal (datos, modelos, features)
deployment/         # API Flask y scripts de Azure (score.py, deploy_to_azure.py)
tests/              # tests unitarios
models/             # artefactos entrenados (fire_model.pkl)
data/               # data/raw y data/processed (no versionado)
```

## Pasos rapidos
1) Instalar dependencias
```
python -m venv .venv
.venv\Scripts\activate   # o source .venv/bin/activate
pip install -r requirements.txt
```

2) Recolectar y preparar datos (usa tus claves en `.env`)
```
python src/data/data_collection.py            # descarga meteo + focos MODIS
python src/data/preprocesamiento_de_datos.py  # limpia y agrega features temporales
```

3) Entrenar y generar artefacto
```
python src/models/train_random_forest.py      # crea models/fire_model.pkl
```

4) Probar prediccion local
```
python src/models/predict.py                  # usa EXPECTED_FEATURES definido en el archivo
```

5) Levantar API local
```
python deployment/api.py
# POST http://localhost:5000/predict con JSON que tenga todas las columnas de EXPECTED_FEATURES
```

6) Desplegar en Azure (requiere config.json del workspace)
```
az login
python deployment/deploy_to_azure.py          # registra fire_model.pkl y despliega score.py
```

## Tests
```
python tests/run_tests.py
```
Verifica preprocesamiento y prediccion binaria.

## Personaliza tus datos
- Reemplaza `data/raw/*.csv` con tus fuentes y ajusta `src/data/data_collection.py` si usas APIs distintas.
- Asegura que el CSV final tenga la columna `Incendio` y que las columnas de entrada coincidan con `EXPECTED_FEATURES` en `src/models/predict.py` (24 variables).
- Vuelve a entrenar (`train_random_forest.py`) para generar tu propio `fire_model.pkl` antes de desplegar.
