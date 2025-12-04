"""
Script de scoring para Azure ML.
Carga un artefacto (modelo + scaler + feature_names) registrado como 'fire-model'.
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

# Hacer accesible la lista de features esperadas
ROOT = Path(__file__).parent.parent / "src"
if ROOT.exists():
    sys.path.append(str(ROOT))
    try:
        from models.predict import EXPECTED_FEATURES  # type: ignore
    except Exception:
        EXPECTED_FEATURES = None
else:
    EXPECTED_FEATURES = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
scaler = None
feature_names = EXPECTED_FEATURES


def init():
    """Carga el artefacto registrado en Azure ML."""
    global model, scaler, feature_names
    try:
        logger.info("Inicializando servicio de prediccion...")
        model_path = Model.get_model_path("fire-model")
        payload = joblib.load(model_path)

        if isinstance(payload, dict) and "model" in payload:
            model = payload["model"]
            scaler = payload.get("scaler")
            feature_names = payload.get("feature_names") or feature_names
        else:
            model = payload

        if feature_names is None:
            raise ValueError("No se encontraron nombres de features para validar el payload.")

        logger.info("Modelo cargado: %s", type(model).__name__)
        logger.info("Features esperados: %s", len(feature_names))
    except Exception as exc:
        logger.error("Error en inicializacion: %s", exc)
        raise


def run(raw_data):
    """Procesa cada request al endpoint de Azure ML."""
    try:
        data = json.loads(raw_data)
        if "data" not in data:
            return json.dumps({"error": 'Formato invalido. Esperado {"data": [[...]]}.'})

        df_input = pd.DataFrame(data["data"], columns=feature_names)
        if len(df_input.columns) != len(feature_names):
            return json.dumps(
                {"error": f"Se esperaban {len(feature_names)} features, se recibieron {len(df_input.columns)}"}
            )

        X = scaler.transform(df_input) if scaler is not None else df_input.to_numpy()

        probabilities = model.predict_proba(X)
        predictions = model.predict(X)

        def risk_bucket(p: float) -> str:
            if p is None:
                return "desconocido"
            if p >= 0.75:
                return "CRITICO"
            if p >= 0.5:
                return "ALTO"
            if p >= 0.25:
                return "MEDIO"
            return "BAJO"

        results = []
        for idx in range(len(predictions)):
            prob_fire = float(probabilities[idx, 1]) if probabilities.ndim == 2 else None
            results.append(
                {
                    "prediction": int(predictions[idx]),
                    "probability": prob_fire,
                    "risk_level": risk_bucket(prob_fire),
                }
            )

        logger.info("Prediccion completada: %s muestras", len(results))
        return json.dumps(
            {
                "results": results,
                "model_name": type(model).__name__,
                "timestamp": pd.Timestamp.utcnow().isoformat(),
            }
        )

    except Exception as exc:
        logger.error("Error en prediccion: %s", exc)
        return json.dumps({"error": str(exc)})


if __name__ == "__main__":
    # Prueba local rapida (requiere tener el modelo descargado en ./models/fire-model.pkl)
    try:
        model = joblib.load("../models/fire_model.pkl")["model"]
        scaler = joblib.load("../models/fire_model.pkl").get("scaler")
        feature_names = joblib.load("../models/fire_model.pkl").get("feature_names", EXPECTED_FEATURES)
        sample = {"data": [np.ones(len(feature_names)).tolist()]}
        print(run(json.dumps(sample)))
    except Exception as exc:
        print(f"No se pudo probar localmente: {exc}")
