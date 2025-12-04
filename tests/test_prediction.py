"""
Tests unitarios para el sistema de prediccion binaria de incendios.
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import unittest

sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.predict import EXPECTED_FEATURES, FirePredictionSystem, create_example_input


def make_mock_model():
    mock = Mock()
    mock.predict.side_effect = lambda X: np.ones((len(X),), dtype=int)
    mock.predict_proba.side_effect = lambda X: np.tile(np.array([[0.2, 0.8]]), (len(X), 1))
    return mock


class TestFirePredictionSystem(unittest.TestCase):
    def setUp(self):
        with patch("joblib.load", return_value=make_mock_model()):
            self.system = FirePredictionSystem(model_path="dummy.pkl", scaler_path=None)

    def _minimal_payload(self) -> Dict:
        data = create_example_input()
        return {k: data[k] for k in EXPECTED_FEATURES}

    def test_preprocess_input_valid(self):
        payload = self._minimal_payload()
        arr = self.system.preprocess_input(payload)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape[1], len(EXPECTED_FEATURES))

    def test_preprocess_input_missing_cols(self):
        with self.assertRaises(ValueError):
            self.system.preprocess_input({"temp_2m_mean": 25})

    def test_predict_single(self):
        payload = self._minimal_payload()
        pred = self.system.predict(payload)
        self.assertIsInstance(pred, int)
        self.assertIn(pred, [0, 1])

    def test_predict_proba(self):
        payload = self._minimal_payload()
        proba = self.system.predict_proba(payload)
        self.assertAlmostEqual(float(proba.sum()), 1.0, places=4)
        self.assertGreaterEqual(proba[0, 1], 0.0)

    def test_predict_with_details(self):
        result = self.system.predict_with_details(self._minimal_payload())
        self.assertIn("risk_level", result)
        self.assertEqual(result["prediction_label"], "incendio")
        self.assertAlmostEqual(result["probability_fire"], 0.8, places=3)

    def test_batch_predict(self):
        df = pd.DataFrame([self._minimal_payload(), self._minimal_payload()])
        res = self.system.batch_predict(df)
        self.assertEqual(len(res), 2)
        self.assertIn("risk_level", res.columns)

    def test_statistics(self):
        self.system.predict_with_details(self._minimal_payload())
        stats = self.system.get_statistics()
        self.assertEqual(stats["total_predictions"], 1)
        self.assertIn("incendio", stats["prediction_distribution"])


class TestCreateExampleInput(unittest.TestCase):
    def test_contains_expected_features(self):
        example = create_example_input()
        for feat in EXPECTED_FEATURES:
            self.assertIn(feat, example)


if __name__ == "__main__":
    unittest.main()
