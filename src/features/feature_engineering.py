"""
Script para ingeniería de características - Predicción de Incendios Forestales
Implementa las transformaciones y creación de features descritas en el artículo técnico
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Clase para realizar ingeniería de características sobre datos de incendios forestales
    """
    
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.feature_names = None
        
    def create_temporal_features(self, df):
        """
        Crea características temporales a partir de la fecha
        
        Args:
            df: DataFrame con columna 'fecha'
            
        Returns:
            DataFrame con nuevas características temporales
        """
        df = df.copy()
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # Extraer componentes temporales
        df['año'] = df['fecha'].dt.year
        df['mes'] = df['fecha'].dt.month
        df['dia'] = df['fecha'].dt.day
        df['dia_año'] = df['fecha'].dt.dayofyear
        df['semana_año'] = df['fecha'].dt.isocalendar().week
        df['trimestre'] = df['fecha'].dt.quarter
        
        # Características cíclicas para mes (1-12)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        
        # Características cíclicas para día del año (1-365)
        df['dia_año_sin'] = np.sin(2 * np.pi * df['dia_año'] / 365)
        df['dia_año_cos'] = np.cos(2 * np.pi * df['dia_año'] / 365)
        
        # Estacionalidad
        df['estacion'] = df['mes'].apply(self._get_season)
        
        print(f"✓ Características temporales creadas: {df.shape[1]} columnas totales")
        return df
    
    @staticmethod
    def _get_season(month):
        """Determina la estación del año según el mes"""
        if month in [12, 1, 2]:
            return 'verano'  # Hemisferio Sur
        elif month in [3, 4, 5]:
            return 'otoño'
        elif month in [6, 7, 8]:
            return 'invierno'
        else:
            return 'primavera'
    
    def create_meteorological_features(self, df):
        """
        Crea características derivadas de variables meteorológicas
        
        Args:
            df: DataFrame con variables meteorológicas
            
        Returns:
            DataFrame con nuevas características meteorológicas
        """
        df = df.copy()
        
        # Índice de sequía (función inversa de precipitación)
        if 'precipitacion' in df.columns:
            df['indice_sequia'] = 1 / (df['precipitacion'] + 1)
        
        # Índice de calor (temperatura * humedad relativa)
        if 'temperatura' in df.columns and 'humedad' in df.columns:
            df['indice_calor'] = df['temperatura'] * (1 - df['humedad'] / 100)
        
        # Déficit de presión de vapor (VPD)
        if 'temperatura' in df.columns and 'humedad' in df.columns:
            # Fórmula simplificada de VPD
            es = 0.6108 * np.exp((17.27 * df['temperatura']) / (df['temperatura'] + 237.3))
            ea = es * (df['humedad'] / 100)
            df['vpd'] = es - ea
        
        # Amplitud térmica (si hay temperatura máxima y mínima)
        if 'temp_max' in df.columns and 'temp_min' in df.columns:
            df['amplitud_termica'] = df['temp_max'] - df['temp_min']
        
        # Razón entre viento y humedad (indicador de riesgo)
        if 'velocidad_viento' in df.columns and 'humedad' in df.columns:
            df['ratio_viento_humedad'] = df['velocidad_viento'] / (df['humedad'] + 1)
        
        print(f"✓ Características meteorológicas creadas: {df.shape[1]} columnas totales")
        return df
    
    def create_aggregated_features(self, df, temporal_cols=['fecha'], value_cols=None):
        """
        Crea características agregadas (promedios móviles, acumulados)
        
        Args:
            df: DataFrame ordenado por fecha
            temporal_cols: Columnas para agrupar (por defecto fecha)
            value_cols: Columnas numéricas para agregar
            
        Returns:
            DataFrame con características agregadas
        """
        df = df.copy()
        df = df.sort_values('fecha')
        
        if value_cols is None:
            value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            value_cols = [col for col in value_cols if col not in ['año', 'mes', 'dia']]
        
        # Promedios móviles de 7 y 30 días
        for col in value_cols[:5]:  # Limitamos para evitar demasiadas columnas
            if col in df.columns:
                df[f'{col}_ma7'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_ma30'] = df[col].rolling(window=30, min_periods=1).mean()
        
        # Valores acumulados
        if 'precipitacion' in df.columns:
            df['precip_acum_7d'] = df['precipitacion'].rolling(window=7, min_periods=1).sum()
            df['precip_acum_30d'] = df['precipitacion'].rolling(window=30, min_periods=1).sum()
        
        print(f"✓ Características agregadas creadas: {df.shape[1]} columnas totales")
        return df
    
    def create_interaction_features(self, df, max_interactions=5):
        """
        Crea características de interacción entre variables
        
        Args:
            df: DataFrame con características
            max_interactions: Número máximo de interacciones a crear
            
        Returns:
            DataFrame con características de interacción
        """
        df = df.copy()
        
        # Interacciones importantes basadas en el dominio
        interactions = []
        
        if 'temperatura' in df.columns and 'velocidad_viento' in df.columns:
            df['temp_x_viento'] = df['temperatura'] * df['velocidad_viento']
            interactions.append('temp_x_viento')
        
        if 'temperatura' in df.columns and 'humedad' in df.columns:
            df['temp_x_humedad'] = df['temperatura'] * (100 - df['humedad'])
            interactions.append('temp_x_humedad')
        
        if 'velocidad_viento' in df.columns and 'precipitacion' in df.columns:
            df['viento_x_precip'] = df['velocidad_viento'] / (df['precipitacion'] + 1)
            interactions.append('viento_x_precip')
        
        if 'temperatura' in df.columns and 'precipitacion' in df.columns:
            df['temp_x_precip'] = df['temperatura'] / (df['precipitacion'] + 1)
            interactions.append('temp_x_precip')
        
        print(f"✓ {len(interactions)} características de interacción creadas")
        return df
    
    def scale_features(self, df, method='standard', columns=None):
        """
        Normaliza o estandariza características
        
        Args:
            df: DataFrame con características
            method: 'standard' o 'minmax'
            columns: Lista de columnas a escalar (None = todas las numéricas)
            
        Returns:
            DataFrame con características escaladas
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Excluir columnas que no deberían escalarse
            exclude = ['año', 'mes', 'dia', 'incendio']
            columns = [col for col in columns if col not in exclude]
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("method debe ser 'standard' o 'minmax'")
        
        df[columns] = self.scaler.fit_transform(df[columns])
        
        print(f"✓ {len(columns)} características escaladas con {method}")
        return df
    
    def apply_pca(self, df, n_components=0.95, exclude_cols=None):
        """
        Aplica PCA para reducción de dimensionalidad
        
        Args:
            df: DataFrame con características
            n_components: Número de componentes o varianza a retener
            exclude_cols: Columnas a excluir del PCA
            
        Returns:
            DataFrame con componentes principales
        """
        df = df.copy()
        
        if exclude_cols is None:
            exclude_cols = ['fecha', 'incendio', 'año', 'mes', 'dia']
        
        # Seleccionar columnas numéricas para PCA
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        pca_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Aplicar PCA
        self.pca = PCA(n_components=n_components)
        pca_features = self.pca.fit_transform(df[pca_cols])
        
        # Crear DataFrame con componentes
        pca_df = pd.DataFrame(
            pca_features,
            columns=[f'PC{i+1}' for i in range(pca_features.shape[1])],
            index=df.index
        )
        
        # Mantener columnas excluidas
        for col in exclude_cols:
            if col in df.columns:
                pca_df[col] = df[col]
        
        variance_explained = self.pca.explained_variance_ratio_.sum()
        print(f"✓ PCA aplicado: {pca_features.shape[1]} componentes "
              f"(varianza explicada: {variance_explained:.2%})")
        
        return pca_df
    
    def transform_target(self, df, target_col='incendio'):
        """
        Transforma la variable objetivo si es necesario
        
        Args:
            df: DataFrame con variable objetivo
            target_col: Nombre de la columna objetivo
            
        Returns:
            DataFrame con target transformado
        """
        df = df.copy()
        
        if target_col in df.columns:
            # Asegurar que es binario (0 o 1)
            df[target_col] = df[target_col].astype(int)
            
            # Imprimir balance de clases
            value_counts = df[target_col].value_counts()
            print(f"\n✓ Balance de clases:")
            print(f"  Clase 0 (No incendio): {value_counts.get(0, 0)} ({value_counts.get(0, 0)/len(df):.2%})")
            print(f"  Clase 1 (Incendio): {value_counts.get(1, 0)} ({value_counts.get(1, 0)/len(df):.2%})")
        
        return df
    
    def pipeline_completo(self, df, apply_pca_flag=False, n_components=0.95):
        """
        Ejecuta el pipeline completo de ingeniería de características
        
        Args:
            df: DataFrame crudo
            apply_pca_flag: Si aplicar PCA o no
            n_components: Componentes para PCA
            
        Returns:
            DataFrame procesado
        """
        print("\n" + "="*60)
        print("PIPELINE DE INGENIERÍA DE CARACTERÍSTICAS")
        print("="*60 + "\n")
        
        # 1. Características temporales
        df = self.create_temporal_features(df)
        
        # 2. Características meteorológicas
        df = self.create_meteorological_features(df)
        
        # 3. Características agregadas
        df = self.create_aggregated_features(df)
        
        # 4. Características de interacción
        df = self.create_interaction_features(df)
        
        # 5. Escalar características
        df = self.scale_features(df, method='standard')
        
        # 6. Transformar target
        df = self.transform_target(df)
        
        # 7. PCA (opcional)
        if apply_pca_flag:
            df = self.apply_pca(df, n_components=n_components)
        
        print("\n" + "="*60)
        print(f"PIPELINE COMPLETADO - Shape final: {df.shape}")
        print("="*60 + "\n")
        
        return df


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
    
    # Aplicar pipeline
    engineer = FeatureEngineer()
    df_procesado = engineer.pipeline_completo(df_ejemplo, apply_pca_flag=False)
    
    print("\nPrimeras 5 filas del dataset procesado:")
    print(df_procesado.head())
    
    print("\nColumnas finales:")
    print(df_procesado.columns.tolist())
