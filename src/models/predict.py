"""
Sistema de prediccion en tiempo real para deteccion de incendios forestales.
El objetivo es binario: 0 = no se espera incendio en 24h, 1 = posible incendio.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

# Lista alineada con el articulo tecnico: variables meteorologicas + derivadas
EXPECTED_FEATURES: List[str] = [
    "temp_2m_mean",
    "temp_2m_max",
    "temp_2m_min",
    "relative_humidity_mean",
    "relative_humidity_min",
    "relative_humidity_max",
    "soil_moisture_7_28cm",
    "soil_moisture_28_100cm",
    "soil_temp_100_255cm",
    "direct_shortwave_radiation",
    "evapotranspiration",
    "vapor_pressure_deficit",
    "cloud_cover_high",
    "precipitation_sum",
    "wind_speed_mean",
    "wind_speed_max",
    "surface_pressure",
    "dew_point",
    "radiation_diffuse",
    "temperature_range",
    "humidity_range",
    "day_of_year",
    "month",
    "is_dry_season",
]


class FirePredictionSystem:
    """
    Carga un modelo entrenado y ofrece predicciones con metadatos
    acordes al enfoque binario descrito en el articulo.
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ):
        loaded = joblib.load(model_path)
        if isinstance(loaded, dict) and "model" in loaded:
            self.model = loaded["model"]
            self.scaler = loaded.get("scaler")
            self.feature_names = loaded.get("feature_names", feature_names or EXPECTED_FEATURES)
        else:
            self.model = loaded
            self.scaler = joblib.load(scaler_path) if scaler_path else None
            self.feature_names = feature_names or EXPECTED_FEATURES

        self.prediction_history: List[Dict] = []

    def _to_dataframe(self, data: Union[pd.DataFrame, Dict, np.ndarray]) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, dict):
            return pd.DataFrame([data])
        if isinstance(data, np.ndarray):
            return pd.DataFrame(data, columns=self.feature_names)
        raise TypeError("El tipo de datos recibido no es soportado")

    def preprocess_input(self, data: Union[pd.DataFrame, Dict, np.ndarray]) -> np.ndarray:
        df = self._to_dataframe(data)
        missing = set(self.feature_names) - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

        ordered_df = df[self.feature_names]

        if self.scaler is not None:
            return np.asarray(self.scaler.transform(ordered_df))

        return ordered_df.to_numpy()

    def predict(self, data: Union[pd.DataFrame, Dict, np.ndarray]) -> Union[int, np.ndarray]:
        X = self.preprocess_input(data)
        predictions = self.model.predict(X)
        if predictions.shape[0] == 1:
            return int(predictions[0])
        return predictions

    def predict_proba(self, data: Union[pd.DataFrame, Dict, np.ndarray]) -> np.ndarray:
        X = self.preprocess_input(data)
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("El modelo cargado no expone predict_proba")
        return self.model.predict_proba(X)

    def _risk_bucket(self, fire_probability: Optional[float]) -> str:
        if fire_probability is None:
            return "desconocido"
        if fire_probability >= 0.75:
            return "critico"
        if fire_probability >= 0.50:
            return "alto"
        if fire_probability >= 0.25:
            return "medio"
        return "bajo"

    def _label(self, prediction: int) -> str:
        return "incendio" if prediction == 1 else "sin_incendio"

    def predict_with_details(self, data: Union[pd.DataFrame, Dict, np.ndarray]) -> Dict:
        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = self.predict_proba(data)

        prediction = self.predict(data)
        fire_probability = None
        if proba is not None:
            fire_probability = float(proba[0][1]) if np.ndim(proba) == 2 else float(proba[1])

        result = {
            "prediction": int(prediction),
            "prediction_label": self._label(int(prediction)),
            "probability_fire": fire_probability,
            "risk_level": self._risk_bucket(fire_probability),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.prediction_history.append(result)
        return result

    def batch_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self._to_dataframe(data)
        preds = self.predict(df)
        probas = None
        if hasattr(self.model, "predict_proba"):
            probas = self.predict_proba(df)[:, 1]

        risk_levels = (
            [self._risk_bucket(p) for p in probas]
            if probas is not None
            else ["desconocido"] * len(df)
        )

        return pd.DataFrame(
            {
                "prediction": np.atleast_1d(preds),
                "prediction_label": [self._label(int(p)) for p in np.atleast_1d(preds)],
                "probability_fire": probas if probas is not None else [None] * len(df),
                "risk_level": risk_levels,
            }
        )

    def save_prediction_history(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.prediction_history, f, indent=2, ensure_ascii=False)

    def get_statistics(self) -> Dict:
        if not self.prediction_history:
            return {"message": "No hay predicciones registradas"}

        preds = [item["prediction"] for item in self.prediction_history]
        distribution = {
            self._label(value): preds.count(value) for value in sorted(set(preds))
        }
        return {
            "total_predictions": len(preds),
            "prediction_distribution": distribution,
            "most_common_prediction": self._label(max(distribution, key=distribution.get)),
        }


def create_example_input() -> Dict:
    return {
        "temp_2m_mean": 27.5,
        "temp_2m_max": 33.0,
        "temp_2m_min": 18.5,
        "relative_humidity_mean": 42.0,
        "relative_humidity_min": 25.0,
        "relative_humidity_max": 70.0,
        "soil_moisture_7_28cm": 0.18,
        "soil_moisture_28_100cm": 0.21,
        "soil_temp_100_255cm": 19.4,
        "direct_shortwave_radiation": 4200,
        "evapotranspiration": 4.8,
        "vapor_pressure_deficit": 2.4,
        "cloud_cover_high": 18.0,
        "precipitation_sum": 0.6,
        "wind_speed_mean": 12.0,
        "wind_speed_max": 28.0,
        "surface_pressure": 910.0,
        "dew_point": 11.0,
        "radiation_diffuse": 140.0,
        "temperature_range": 14.5,
        "humidity_range": 45.0,
        "day_of_year": 210,
        "month": 7,
        "is_dry_season": 1,
    }


def main():
    model_path = "./models/random_forest_model.pkl"
    scaler_path = "./models/scaler.pkl"

    if not Path(model_path).exists():
        print("Modelo no encontrado. Entrena un modelo antes de predecir.")
        return

    predictor = FirePredictionSystem(
        model_path=model_path,
        scaler_path=scaler_path if Path(scaler_path).exists() else None,
    )

    example = create_example_input()
    print("Ejemplo de datos de entrada:")
    for key, value in example.items():
        print(f"  {key}: {value}")

    print("\nResultado de la prediccion:")
    result = predictor.predict_with_details(example)
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
