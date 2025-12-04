"""
Entrenamiento y exporte de artefacto RandomForest listo para Azure/REST.
Genera un archivo con modelo, scaler y orden de features.
"""

import logging
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestTrainer:
    """Entrena y guarda un artefacto (modelo + scaler + features)."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.feature_importance = None

    def load_and_split_data(
        self, filepath: str, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler, list]:
        logger.info("Cargando datos desde %s", filepath)
        df = pd.read_csv(filepath)

        X = df.drop(["Incendio", "date"], axis=1, errors="ignore")
        y = df["Incendio"]
        feature_names = X.columns.tolist()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

        logger.info("Train: %s muestras | Test: %s muestras", len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test, scaler, feature_names

    def train_baseline(self, X_train, y_train):
        self.model = RandomForestClassifier(
            n_estimators=200,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        return self.model

    def hyperparameter_tuning(self, X_train, y_train, n_iter: int = 30, cv: int = 4):
        param_distributions = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
            "class_weight": ["balanced", "balanced_subsample"],
        }

        cv_strategy = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=self.random_state
        )

        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_strategy,
            scoring="f1",
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1,
            return_train_score=True,
        )

        random_search.fit(X_train, y_train)
        self.model = random_search.best_estimator_
        logger.info("Mejor F1 en CV: %.4f", random_search.best_score_)
        logger.info("Hiperparametros seleccionados: %s", random_search.best_params_)
        return self.model

    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Entrena el modelo antes de evaluar.")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, digits=3)
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        logger.info("Classification report:\n%s", report)
        logger.info("Confusion matrix:\n%s", cm)
        logger.info("AUC-ROC: %.3f", auc)

        self.feature_importance = pd.DataFrame(
            {"feature": range(len(self.model.feature_importances_)), "importance": self.model.feature_importances_}
        )

        return {
            "report": report,
            "confusion_matrix": cm.tolist(),
            "auc": auc,
            "precision_recall_curve": precision_recall_curve(y_test, y_pred_proba),
            "roc_curve": roc_curve(y_test, y_pred_proba),
        }

    def save_artifact(self, filepath: str, scaler: StandardScaler, feature_names: list):
        if self.model is None:
            raise ValueError("No hay modelo entrenado para guardar.")
        artifact = {
            "model": self.model,
            "scaler": scaler,
            "feature_names": feature_names,
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, filepath)
        logger.info("Artefacto guardado en %s", filepath)


def main():
    trainer = RandomForestTrainer(random_state=42)

    X_train, X_test, y_train, y_test, scaler, feature_names = trainer.load_and_split_data(
        "../../data/processed/datos_de_incendios_clean.csv",
        test_size=0.2,
    )

    trainer.hyperparameter_tuning(X_train, y_train, n_iter=20, cv=4)
    trainer.evaluate_model(X_test, y_test)
    trainer.save_artifact("../../models/fire_model.pkl", scaler, feature_names)
    logger.info("Entrenamiento y guardado completados.")


if __name__ == "__main__":
    main()
