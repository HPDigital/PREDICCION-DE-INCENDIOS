"""
API REST para el sistema de prediccion de incendios descrito en el articulo.
Exponde endpoints basicos para health, prediccion individual y por lotes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).parent.parent / "src"))

from models.predict import EXPECTED_FEATURES, FirePredictionSystem

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_PATH = Path("./models/fire_model.pkl")
SCALER_PATH = Path("./models/scaler.pkl")

predictor = None
if MODEL_PATH.exists():
    try:
        predictor = FirePredictionSystem(
            model_path=str(MODEL_PATH),
            scaler_path=str(SCALER_PATH) if SCALER_PATH.exists() else None,
        )
        logger.info("Modelo cargado para la API")
    except Exception as exc:
        logger.error("No se pudo inicializar el predictor: %s", exc)
else:
    logger.warning("Modelo no encontrado en %s", MODEL_PATH)


def _validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload:
        raise ValueError("No se proporcionaron datos")
    missing = set(EXPECTED_FEATURES) - set(payload.keys())
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    return payload


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "service": "Prediccion de incendios forestales",
            "status": "ready" if predictor else "model_missing",
            "expected_features": EXPECTED_FEATURES,
            "endpoints": ["/", "/health", "/predict", "/predict/batch", "/stats"],
        }
    )


@app.route("/health", methods=["GET"])
def health():
    status = predictor is not None
    return jsonify(
        {
            "status": "healthy" if status else "unhealthy",
            "model_path": str(MODEL_PATH),
            "predictions_recorded": len(predictor.prediction_history) if predictor else 0,
        }
    ), (200 if status else 503)


@app.route("/predict", methods=["POST"])
def predict():
    if predictor is None:
        return jsonify({"error": "Modelo no disponible"}), 503
    try:
        payload = request.get_json(force=True)
        data = _validate_payload(payload)
        result = predictor.predict_with_details(data)
        return jsonify({"result": result})
    except Exception as exc:
        logger.exception("Error al predecir")
        return jsonify({"error": str(exc)}), 400


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    if predictor is None:
        return jsonify({"error": "Modelo no disponible"}), 503
    try:
        payload = request.get_json(force=True)
        if not payload or "data" not in payload:
            raise ValueError('Se espera un JSON con la clave "data"')
        records = payload["data"]
        if not isinstance(records, list) or not records:
            raise ValueError("El contenido de 'data' debe ser una lista con elementos")
        # Validar con la primera fila
        _validate_payload(records[0])
        df_results = predictor.batch_predict(pd.DataFrame(records))
        return jsonify({"count": len(df_results), "results": df_results.to_dict("records")})
    except Exception as exc:
        logger.exception("Error en prediccion batch")
        return jsonify({"error": str(exc)}), 400


@app.route("/stats", methods=["GET"])
def stats():
    if predictor is None:
        return jsonify({"error": "Modelo no disponible"}), 503
    return jsonify(predictor.get_statistics())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
