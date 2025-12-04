"""
Script para evaluación de modelos de predicción de incendios.
Calcula métricas de rendimiento y genera reportes de evaluación.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_model(model_path: str):
    """
    Carga un modelo guardado desde disco.
    
    Args:
        model_path: Ruta al archivo del modelo
        
    Returns:
        Modelo cargado
    """
    print(f"Cargando modelo desde: {model_path}")
    obj = joblib.load(model_path)
    if isinstance(obj, dict) and 'model' in obj:
        model = obj['model']
        print("Modelo y artefactos cargados")
        return model
    print("Modelo cargado exitosamente")
    return obj


def calculate_metrics(y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      y_proba: Optional[np.ndarray] = None) -> Dict:
    """
    Calcula métricas de evaluación del modelo.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        y_proba: Probabilidades predichas (opcional)
        
    Returns:
        Diccionario con métricas calculadas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Calcular métricas adicionales si se proporcionan probabilidades
    if y_proba is not None and len(np.unique(y_true)) == 2:
        # Para clasificación binaria
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['average_precision'] = average_precision_score(y_true, y_proba[:, 1])
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         save_path: str,
                         classes: Optional[list] = None):
    """
    Genera y guarda la matriz de confusión.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        save_path: Ruta donde guardar la imagen
        classes: Nombres de las clases (opcional)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes if classes else 'auto',
                yticklabels=classes if classes else 'auto')
    plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
    plt.ylabel('Etiqueta Verdadera', fontsize=12)
    plt.xlabel('Etiqueta Predicha', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matriz de confusión guardada en: {save_path}")


def plot_roc_curve(y_true: np.ndarray,
                   y_proba: np.ndarray,
                   save_path: str):
    """
    Genera y guarda la curva ROC.
    
    Args:
        y_true: Etiquetas verdaderas
        y_proba: Probabilidades predichas
        save_path: Ruta donde guardar la imagen
    """
    if len(np.unique(y_true)) != 2:
        print("Advertencia: La curva ROC solo está disponible para clasificación binaria")
        return
    
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Clasificador Aleatorio')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Curva ROC (Receiver Operating Characteristic)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Curva ROC guardada en: {save_path}")


def plot_precision_recall_curve(y_true: np.ndarray,
                                y_proba: np.ndarray,
                                save_path: str):
    """
    Genera y guarda la curva Precisión-Recall.
    
    Args:
        y_true: Etiquetas verdaderas
        y_proba: Probabilidades predichas
        save_path: Ruta donde guardar la imagen
    """
    if len(np.unique(y_true)) != 2:
        print("Advertencia: La curva Precisión-Recall solo está disponible para clasificación binaria")
        return
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    avg_precision = average_precision_score(y_true, y_proba[:, 1])
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'Precisión Promedio = {avg_precision:.2f}')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precisión', fontsize=12)
    plt.title('Curva Precisión-Recall', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Curva Precisión-Recall guardada en: {save_path}")


def generate_classification_report(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   save_path: str,
                                   classes: Optional[list] = None):
    """
    Genera y guarda un reporte de clasificación detallado.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones del modelo
        save_path: Ruta donde guardar el reporte
        classes: Nombres de las clases (opcional)
    """
    report = classification_report(y_true, y_pred, 
                                   target_names=classes if classes else None,
                                   digits=4)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE DE CLASIFICACIÓN\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"Reporte de clasificación guardado en: {save_path}")
    print("\nReporte de Clasificación:")
    print(report)


def save_metrics_json(metrics: Dict, save_path: str):
    """
    Guarda las métricas en formato JSON.
    
    Args:
        metrics: Diccionario con métricas
        save_path: Ruta donde guardar el archivo JSON
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"Métricas guardadas en: {save_path}")


def evaluate_model(model_path: str,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  output_dir: str = './results',
                  model_name: str = 'model',
                  class_names: Optional[list] = None) -> Dict:
    """
    Función principal para evaluar un modelo completo.
    
    Args:
        model_path: Ruta al modelo guardado
        X_test: Datos de prueba
        y_test: Etiquetas de prueba
        output_dir: Directorio para guardar resultados
        model_name: Nombre del modelo
        class_names: Nombres de las clases (opcional)
        
    Returns:
        Diccionario con todas las métricas calculadas
    """
    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Cargar modelo
    model = load_model(model_path)
    
    # Realizar predicciones
    print("\nRealizando predicciones...")
    y_pred = model.predict(X_test)
    
    # Obtener probabilidades si el modelo lo soporta
    try:
        y_proba = model.predict_proba(X_test)
    except AttributeError:
        y_proba = None
        print("El modelo no soporta predict_proba, se omitirán algunas métricas")
    
    # Calcular métricas
    print("\nCalculando métricas...")
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    
    # Imprimir métricas principales
    print("\n" + "=" * 60)
    print("MÉTRICAS DE EVALUACIÓN")
    print("=" * 60)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper():<20}: {metric_value:.4f}")
    print("=" * 60)
    
    # Generar visualizaciones
    print("\nGenerando visualizaciones...")
    
    # Matriz de confusión
    cm_path = output_path / f'{model_name}_confusion_matrix.png'
    plot_confusion_matrix(y_test, y_pred, str(cm_path), class_names)
    
    # Curva ROC (solo para clasificación binaria)
    if y_proba is not None and len(np.unique(y_test)) == 2:
        roc_path = output_path / f'{model_name}_roc_curve.png'
        plot_roc_curve(y_test, y_proba, str(roc_path))
        
        # Curva Precisión-Recall
        pr_path = output_path / f'{model_name}_precision_recall_curve.png'
        plot_precision_recall_curve(y_test, y_proba, str(pr_path))
    
    # Reporte de clasificación
    report_path = output_path / f'{model_name}_classification_report.txt'
    generate_classification_report(y_test, y_pred, str(report_path), class_names)
    
    # Guardar métricas en JSON
    metrics_path = output_path / f'{model_name}_metrics.json'
    save_metrics_json(metrics, str(metrics_path))
    
    print("\n✓ Evaluación completada exitosamente")
    
    return metrics


if __name__ == "__main__":
    # Ejemplo de uso
    print("Script de evaluación de modelos")
    print("Este script debe ser importado o ejecutado con argumentos específicos")
    print("\nEjemplo de uso:")
    print("python evaluate_model.py --model path/to/model.pkl --data path/to/test_data.csv")
