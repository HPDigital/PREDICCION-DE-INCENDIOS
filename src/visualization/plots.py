"""
Script de visualización - Predicción de Incendios Forestales
Genera los gráficos y visualizaciones descritas en el artículo técnico
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class WildfireVisualizer:
    """
    Clase para crear visualizaciones del proyecto de predicción de incendios
    """
    
    def __init__(self, figsize=(12, 8), save_path='../outputs/'):
        self.figsize = figsize
        self.save_path = save_path
        
    def plot_time_series(self, df, date_col='fecha', value_cols=None, 
                        target_col='incendio', save_name=None):
        """
        Visualiza series temporales de variables con eventos de incendio
        
        Args:
            df: DataFrame con datos
            date_col: Columna de fecha
            value_cols: Columnas a graficar
            target_col: Columna de incendios
            save_name: Nombre para guardar figura
        """
        if value_cols is None:
            value_cols = ['temperatura', 'humedad', 'precipitacion', 'velocidad_viento']
        
        fig, axes = plt.subplots(len(value_cols), 1, figsize=(14, 3*len(value_cols)))
        
        if len(value_cols) == 1:
            axes = [axes]
        
        for idx, col in enumerate(value_cols):
            if col in df.columns:
                # Graficar serie temporal
                axes[idx].plot(df[date_col], df[col], label=col, linewidth=1, alpha=0.7)
                
                # Marcar eventos de incendio
                if target_col in df.columns:
                    fire_dates = df[df[target_col] == 1][date_col]
                    fire_values = df[df[target_col] == 1][col]
                    axes[idx].scatter(fire_dates, fire_values, color='red', 
                                    label='Incendio', s=50, alpha=0.6, marker='^')
                
                axes[idx].set_xlabel('Fecha')
                axes[idx].set_ylabel(col.replace('_', ' ').title())
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_path}{save_name}.png", dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico guardado: {save_name}.png")
        
        plt.show()
    
    def plot_correlation_matrix(self, df, figsize=(12, 10), save_name=None):
        """
        Visualiza matriz de correlación de características
        
        Args:
            df: DataFrame con características numéricas
            figsize: Tamaño de la figura
            save_name: Nombre para guardar figura
        """
        # Seleccionar solo columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        plt.title('Matriz de Correlación de Características', fontsize=16, pad=20)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_path}{save_name}.png", dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico guardado: {save_name}.png")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names, importances, top_n=20, save_name=None):
        """
        Visualiza importancia de características
        
        Args:
            feature_names: Nombres de las características
            importances: Valores de importancia
            top_n: Top N características a mostrar
            save_name: Nombre para guardar figura
        """
        # Ordenar por importancia
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices][::-1])
        plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
        plt.xlabel('Importancia')
        plt.title(f'Top {top_n} Características Más Importantes', fontsize=14)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_path}{save_name}.png", dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico guardado: {save_name}.png")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False, save_name=None):
        """
        Visualiza matriz de confusión
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            normalize: Si normalizar la matriz
            save_name: Nombre para guardar figura
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                   square=True, cbar_kws={"shrink": 0.8})
        
        plt.title('Matriz de Confusión' + (' (Normalizada)' if normalize else ''), 
                 fontsize=14)
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        
        # Añadir etiquetas
        plt.xticks([0.5, 1.5], ['No Incendio', 'Incendio'])
        plt.yticks([0.5, 1.5], ['No Incendio', 'Incendio'])
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_path}{save_name}.png", dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico guardado: {save_name}.png")
        
        plt.show()
    
    def plot_roc_curve(self, y_true, y_proba, save_name=None):
        """
        Visualiza curva ROC
        
        Args:
            y_true: Etiquetas verdaderas
            y_proba: Probabilidades predichas
            save_name: Nombre para guardar figura
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'Curva ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatorio')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
        plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
        plt.title('Curva ROC - Característica Operativa del Receptor', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_path}{save_name}.png", dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico guardado: {save_name}.png")
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_proba, save_name=None):
        """
        Visualiza curva Precision-Recall
        
        Args:
            y_true: Etiquetas verdaderas
            y_proba: Probabilidades predichas
            save_name: Nombre para guardar figura
        """
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Curva PR (AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall (Sensibilidad)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Curva Precision-Recall', fontsize=14)
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_path}{save_name}.png", dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico guardado: {save_name}.png")
        
        plt.show()
    
    def plot_pca_variance(self, pca_model, save_name=None):
        """
        Visualiza varianza explicada por componentes principales
        
        Args:
            pca_model: Modelo PCA entrenado
            save_name: Nombre para guardar figura
        """
        variance_ratio = pca_model.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico de barras de varianza individual
        ax1.bar(range(1, len(variance_ratio) + 1), variance_ratio)
        ax1.set_xlabel('Componente Principal')
        ax1.set_ylabel('Varianza Explicada')
        ax1.set_title('Varianza Explicada por Componente')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de línea de varianza acumulada
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                marker='o', linestyle='-', linewidth=2)
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% varianza')
        ax2.set_xlabel('Número de Componentes')
        ax2.set_ylabel('Varianza Acumulada')
        ax2.set_title('Varianza Acumulada Explicada')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_path}{save_name}.png", dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico guardado: {save_name}.png")
        
        plt.show()
    
    def plot_distribution_comparison(self, df, col, target_col='incendio', save_name=None):
        """
        Compara distribuciones de una variable entre clases
        
        Args:
            df: DataFrame con datos
            col: Columna a analizar
            target_col: Columna objetivo
            save_name: Nombre para guardar figura
        """
        plt.figure(figsize=(10, 6))
        
        # Histogramas superpuestos
        df[df[target_col] == 0][col].hist(alpha=0.5, bins=30, label='No Incendio', density=True)
        df[df[target_col] == 1][col].hist(alpha=0.5, bins=30, label='Incendio', density=True)
        
        plt.xlabel(col.replace('_', ' ').title())
        plt.ylabel('Densidad')
        plt.title(f'Distribución de {col.replace("_", " ").title()} por Clase', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_path}{save_name}.png", dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico guardado: {save_name}.png")
        
        plt.show()
    
    def plot_seasonal_analysis(self, df, date_col='fecha', target_col='incendio', save_name=None):
        """
        Analiza patrones estacionales de incendios
        
        Args:
            df: DataFrame con datos
            date_col: Columna de fecha
            target_col: Columna objetivo
            save_name: Nombre para guardar figura
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df['mes'] = df[date_col].dt.month
        df['año'] = df[date_col].dt.year
        
        # Agrupar por mes
        monthly_fires = df.groupby('mes')[target_col].sum()
        monthly_total = df.groupby('mes')[target_col].count()
        monthly_rate = (monthly_fires / monthly_total) * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Número de incendios por mes
        ax1.bar(monthly_fires.index, monthly_fires.values)
        ax1.set_xlabel('Mes')
        ax1.set_ylabel('Número de Incendios')
        ax1.set_title('Incendios por Mes')
        ax1.set_xticks(range(1, 13))
        ax1.grid(True, alpha=0.3)
        
        # Tasa de incendios por mes
        ax2.plot(monthly_rate.index, monthly_rate.values, marker='o', linewidth=2)
        ax2.set_xlabel('Mes')
        ax2.set_ylabel('Tasa de Incendios (%)')
        ax2.set_title('Tasa de Incendios por Mes')
        ax2.set_xticks(range(1, 13))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(f"{self.save_path}{save_name}.png", dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico guardado: {save_name}.png")
        
        plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    # Crear datos de ejemplo
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    df_ejemplo = pd.DataFrame({
        'fecha': dates,
        'temperatura': np.random.uniform(15, 35, n_samples),
        'humedad': np.random.uniform(20, 90, n_samples),
        'velocidad_viento': np.random.uniform(0, 40, n_samples),
        'precipitacion': np.random.exponential(2, n_samples),
        'incendio': np.random.binomial(1, 0.1, n_samples)
    })
    
    # Crear visualizador
    viz = WildfireVisualizer(save_path='../outputs/')
    
    # Ejemplos de visualizaciones
    print("Generando visualizaciones de ejemplo...")
    
    # 1. Series temporales
    viz.plot_time_series(df_ejemplo)
    
    # 2. Análisis estacional
    viz.plot_seasonal_analysis(df_ejemplo)
    
    # 3. Comparación de distribuciones
    viz.plot_distribution_comparison(df_ejemplo, 'temperatura')
    
    print("\n✓ Visualizaciones de ejemplo generadas")
