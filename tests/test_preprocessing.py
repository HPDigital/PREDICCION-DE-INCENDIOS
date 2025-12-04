"""
Tests del preprocesamiento para el pipeline descrito en el articulo tecnico.
"""

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.preprocesamiento_de_datos import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.pre = DataPreprocessor()
        self.sample = pd.DataFrame(
            {
                "temp_2m_mean": [20, 25, np.nan, 30],
                "relative_humidity_mean": [60, np.nan, 55, 50],
                "wind_speed_mean": [10, 12, 14, 16],
                "Incendio": [0, 1, 0, 1],
            }
        )

    def test_handle_missing_values(self):
        filled = self.pre.handle_missing_values(self.sample, strategy="mean")
        self.assertEqual(filled.isnull().sum().sum(), 0)

    def test_remove_outliers(self):
        df = self.sample.copy()
        df.loc[3, "wind_speed_mean"] = 100
        clean = self.pre.remove_outliers(df, threshold=2.5)
        self.assertLess(len(clean), len(df))

    def test_normalize_features(self):
        features = self.sample.drop(columns=["Incendio"])
        features = self.pre.handle_missing_values(features, strategy="mean")
        norm, scaler = self.pre.normalize_features(features, method="standard")
        self.assertIsNotNone(scaler)
        self.assertAlmostEqual(float(norm.mean().mean()), 0.0, places=1)

    def test_split_and_pipeline(self):
        result = self.pre.preprocess_pipeline(self.sample, target_column="Incendio", test_size=0.5)
        self.assertIn("X_train", result)
        self.assertEqual(len(result["X_train"]) + len(result["X_test"]), len(self.sample))


class TestDataIntegrity(unittest.TestCase):
    def test_types(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [0.5, 0.6, 0.7]})
        self.assertTrue(np.issubdtype(df["a"].dtype, np.number))
        self.assertTrue(np.issubdtype(df["b"].dtype, np.number))


if __name__ == "__main__":
    unittest.main()
