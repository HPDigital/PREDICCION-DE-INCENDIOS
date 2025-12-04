"""
preprocesamiento_de_datos.py
=====================
Pipeline de limpieza y preprocesamiento de datos.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocesador:
    """Preprocesador de datos con validación robusta"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def cargar_datos(self, filepath: str) -> pd.DataFrame:
        """Cargar datos crudos desde CSV"""
        logger.info(f"Cargando datos desde {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"✓ Cargados {len(df)} registros")
        
        return df
    
    def quitar_duplicados(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remover registros duplicados basados en fecha"""
        initial_count = len(df)
        
        df = df.drop_duplicates(subset=['date'], keep='first')
        
        removed = initial_count - len(df)
        if removed > 0:
            logger.warning(f"⚠ Removidos {removed} registros duplicados")
        else:
            logger.info("✓ Sin duplicados encontrados")
        
        return df
    
    def salida_cronologica(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordenar por fecha ascendente"""
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"✓ Datos ordenados: {df['date'].min()} a {df['date'].max()}")
        
        return df
    
    def valores_faltantes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputar valores faltantes usando forward fill.
        
        Estrategia:
        1. Forward fill (usar valor anterior)
        2. Backward fill para primeros registros
        3. Si aún hay NaN, usar mediana de la columna
        """
        inicialmente_nulos = df.isnull().sum().sum()
        logger.info(f"Missing values iniciales: {inicialmente_nulos}")
        
        if inicialmente_nulos == 0:
            logger.info("✓ Sin valores faltantes")
            return df
        
        # Identificar columnas numéricas (excluir date e Incendio)
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Incendio' in columnas_numericas:
            columnas_numericas.remove('Incendio')
        
        # Forward fill
        df[columnas_numericas] = df[columnas_numericas].fillna(method='ffill')
        
        # Backward fill para inicio
        df[columnas_numericas] = df[columnas_numericas].fillna(method='bfill')
        
        # Mediana como último recurso
        for col in columnas_numericas:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.warning(f"⚠ Columna {col}: usada mediana ({median_val:.2f})")
        
        final_nulls = df.isnull().sum().sum()
        logger.info(f"✓ Missing values finales: {final_nulls}")
        
        return df
    
    def detectar_outliers(
        self, 
        df: pd.DataFrame, 
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detectar outliers usando Z-score, pero NO removerlos.
        Solo marcar para análisis.
        """
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Incendio' in columnas_numericas:
            columnas_numericas.remove('Incendio')
        
        contador_outliers = {}
        
        for col in columnas_numericas:
            z_scores = np.abs(
                (df[col] - df[col].mean()) / df[col].std()
            )
            outlier_mask = z_scores > threshold
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                contador_outliers[col] = outlier_count
                # Marcar outliers en nueva columna
                df[f'{col}_is_outlier'] = outlier_mask
        
        total_outliers = sum(contador_outliers.values())
        logger.info(f"✓ Detectados {total_outliers} outliers en {len(contador_outliers)} variables")
        
        for col, count in sorted(contador_outliers.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  • {col}: {count} outliers")
        
        return df
    
    def validar_variable_objetivo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validar que variable objetivo esté correcta"""
        
        # Verificar que solo contenga 0 y 1
        unique_values = df['Incendio'].unique()
        assert set(unique_values).issubset({0, 1}), \
            f"Variable objetivo tiene valores inválidos: {unique_values}"
        
        # Verificar que no haya NaN
        assert df['Incendio'].isnull().sum() == 0, \
            "Variable objetivo contiene valores NaN"
        
        # Estadísticas
        class_distribution = df['Incendio'].value_counts()
        class_0_pct = class_distribution[0] / len(df) * 100
        class_1_pct = class_distribution[1] / len(df) * 100
        
        logger.info(f"✓ Variable objetivo validada:")
        logger.info(f"  • Clase 0 (No fuego): {class_distribution[0]} ({class_0_pct:.1f}%)")
        logger.info(f"  • Clase 1 (Fuego): {class_distribution[1]} ({class_1_pct:.1f}%)")
        
        return df
    
    def crear_features_temporales(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear features temporales adicionales.
        """
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['quarter'] = df['date'].dt.quarter
        df['is_dry_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
        
        # Features cíclicos
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        logger.info("✓ Features temporales creados")
        
        return df
    
    def ajustar_escalamiento(self, df: pd.DataFrame) -> 'DataPreprocesador':
        """Ajustar StandardScaler en datos de entrenamiento"""
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Excluir target y columnas auxiliares
        exclude = ['Incendio', 'year', 'month', 'day_of_year', 'quarter']
        feature_cols = [col for col in columnas_numericas if col not in exclude]
        
        self.feature_names = feature_cols
        self.scaler.fit(df[feature_cols])
        
        logger.info(f"✓ Scaler ajustado en {len(feature_cols)} features")
        
        return self
    
    def transformar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplicar transformación de escalado"""
        if self.feature_names is None:
            raise ValueError("Scaler no ha sido ajustado. Llamar ajustar_escalamiento() primero.")
        
        df_scaled = df.copy()
        df_scaled[self.feature_names] = self.scaler.transform(df[self.feature_names])
        
        logger.info("✓ Transformación aplicada")
        
        return df_scaled
    
    def guardar_data_limpia(self, df: pd.DataFrame, filepath: str):
        """Guardar datos limpios"""
        df.to_csv(filepath, index=False)
        logger.info(f"✓ Datos limpios guardados en {filepath}")
    
    def full_pipeline(self, input_path: str, output_path: str) -> pd.DataFrame:
        """
        Ejecutar pipeline completo de limpieza.
        """
        logger.info("="*60)
        logger.info("INICIANDO PIPELINE DE LIMPIEZA")
        logger.info("="*60)
        
        # Cargar
        df = self.cargar_datos(input_path)
        
        # Limpieza
        df = self.quitar_duplicados(df)
        df = self.salida_cronologica(df)
        df = self.valores_faltantes(df)
        df = self.detectar_outliers(df)
        df = self.validar_variable_objetivo(df)
        df = self.crear_features_temporales(df)
        
        # Guardar
        self.guardar_data_limpia(df, output_path)
        
        logger.info(f"✓ Pipeline completado: {len(df)} registros listos")
        logger.info("="*60)
        
        return df


class DataPreprocessor:
    """
    Wrapper en ingles alineado con el articulo para limpiar, normalizar y dividir
    el dataset (target binario Incendio).
    """

    def __init__(self):
        self.scaler = None

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        if strategy == "mean":
            return df.fillna(df.mean(numeric_only=True))
        if strategy == "median":
            return df.fillna(df.median(numeric_only=True))
        if strategy == "drop":
            return df.dropna()
        raise ValueError("Estrategia no soportada")

    def remove_outliers(self, df: pd.DataFrame, method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
        if method != "zscore":
            raise ValueError("Solo se soporta zscore en este wrapper")
        numeric = df.select_dtypes(include=[np.number])
        z = np.abs((numeric - numeric.mean()) / numeric.std(ddof=0))
        mask = (z < threshold).all(axis=1)
        return df.loc[mask].reset_index(drop=True)

    def normalize_features(self, df: pd.DataFrame, method: str = "standard"):
        if method == "standard":
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()
        else:
            raise ValueError("Metodo de normalizacion no soportado")

        transformed = scaler.fit_transform(df)
        self.scaler = scaler
        return pd.DataFrame(transformed, columns=df.columns), scaler

    def split_data(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
    ):
        from sklearn.model_selection import train_test_split

        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        target_column: str = "Incendio",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        if df.empty:
            raise ValueError("El dataframe esta vacio")
        if target_column not in df.columns:
            raise ValueError("No se encontro la columna objetivo")

        df_clean = self.handle_missing_values(df, strategy="mean")
        df_clean = self.remove_outliers(df_clean, method="zscore", threshold=3.5)

        y = df_clean[target_column].astype(int)
        X = df_clean.drop(columns=[target_column])

        X_train, X_test, y_train, y_test = self.split_data(
            X, y, test_size=test_size, random_state=random_state
        )

        X_train_scaled, scaler = self.normalize_features(X_train, method="standard")
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train.reset_index(drop=True),
            "y_test": y_test.reset_index(drop=True),
            "scaler": scaler,
        }


if __name__ == "__main__":
    # Ejemplo de uso
    preprocessor = DataPreprocesador()
    
    df_clean = preprocessor.full_pipeline(
        input_path='../../data/raw/Data_clima_Clasificacion.csv',
        output_path='../../data/processed/datos_de_incendios_clean.csv'
    )
    
    print(f"\n✓ Dataset limpio: {df_clean.shape}")
    print(f"✓ Rango de fechas: {df_clean['date'].min()} a {df_clean['date'].max()}")
