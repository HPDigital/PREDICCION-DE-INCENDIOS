"""
data_collection.py
==================
Script para recolección automatizada de datos meteorológicos y satelitales.
Ejecutar diariamente mediante cron job o Azure Functions.

Autor: Herwig Luk Poleyn Paz
Fecha: Diciembre 2023 - Actualizado 2025
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import os
from dotenv import load_dotenv

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
BLEU_API_KEY = os.getenv('BLEU_API_KEY')
NASA_API_KEY = os.getenv('NASA_API_KEY')


class RecolectorDeDatos:
    """Recolector de datos meteorológicos desde BLEU API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.bleu-weather.com/v1"
        self.location = {
            'latitude': -17.3935,
            'longitude': -66.1570,
            'name': 'Cochabamba'
        }
    
    def obtener_datos_diarios(self, date: str) -> Dict:
        """
        Obtener datos meteorológicos para una fecha específica.
        
        Args:
            date: Fecha en formato 'YYYY-MM-DD'
            
        Returns:
            Diccionario con variables meteorológicas
        """
        endpoint = f"{self.base_url}/historical"
        params = {
            'lat': self.location['latitude'],
            'lon': self.location['longitude'],
            'date': date,
            'variables': [
                'temperature_2m',
                'relative_humidity_2m',
                'soil_moisture_7_28cm',
                'soil_moisture_28_100cm',
                'soil_temperature_100_255cm',
                'direct_shortwave_radiation',
                'evapotranspiration',
                'vapor_pressure_deficit',
                'cloud_cover_high'
            ],
            'api_key': self.api_key
        }
        
        try:
            response = requests.get(
                endpoint, 
                params=params, 
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            logger.info(f"Datos meteorológicos obtenidos para {date}")
            return self._procesar_respuesta(data, date)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error obteniendo datos para {date}: {e}")
            return None
    
    def _procesar_respuesta(self, data: Dict, date: str) -> Dict:
        """Procesar respuesta de la API y extraer variables"""
        processed = {'date': date}
        
        # Extraer promedios diarios de cada variable
        for variable in data['hourly']['variables']:
            var_name = variable['name']
            var_values = variable['values']
            
            # Calcular estadísticas
            processed[f'{var_name}_mean'] = sum(var_values) / len(var_values)
            processed[f'{var_name}_max'] = max(var_values)
            processed[f'{var_name}_min'] = min(var_values)
            processed[f'{var_name}_std'] = pd.Series(var_values).std()
        
        return processed
    
    def obtener_datos_de_fechas(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Obtener datos para un rango de fechas.
        
        Args:
            start_date: Fecha inicial 'YYYY-MM-DD'
            end_date: Fecha final 'YYYY-MM-DD'
            
        Returns:
            DataFrame con datos de todas las fechas
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_data = []
        current = start
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            daily_data = self.obtener_datos_diarios(date_str)
            
            if daily_data:
                all_data.append(daily_data)
            
            current += timedelta(days=1)
            # Rate limiting: respetar límites de API
            time.sleep(0.5)
        
        df = pd.DataFrame(all_data)
        logger.info(f"Recolectados {len(df)} días de datos meteorológicos")
        return df


class ColectorDatos:
    """Recolector de datos de incendios desde NASA MODIS"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://firms.modaps.eosdis.nasa.gov/api"
        self.bbox = {
            'west': -66.5,
            'south': -17.6,
            'east': -65.8,
            'north': -17.1
        }
    
    def fetch_datos_de_incendios(self, date: str) -> List[Dict]:
        """
        Obtener focos de calor para una fecha específica.
        
        Args:
            date: Fecha en formato 'YYYY-MM-DD'
            
        Returns:
            Lista de focos de calor detectados
        """
        endpoint = f"{self.base_url}/country/csv/{self.api_key}/MODIS_NRT/BOL/1"
        
        try:
            response = requests.get(endpoint, timeout=30)
            response.raise_for_status()
            
            # Parsear CSV response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            # Filtrar por bounding box y fecha
            df_filtered = df[
                (df['latitude'] >= self.bbox['south']) &
                (df['latitude'] <= self.bbox['north']) &
                (df['longitude'] >= self.bbox['west']) &
                (df['longitude'] <= self.bbox['east']) &
                (df['acq_date'] == date) &
                (df['confidence'] >= 80)  # Solo alta confianza
            ]
            
            logger.info(f"✓ {len(df_filtered)} focos detectados para {date}")
            return df_filtered.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error obteniendo focos para {date}: {e}")
            return []
    
    def agregar_al_diario(self, fires: List[Dict]) -> Dict:
        """
        Agregar focos a nivel diario.
        
        Returns:
            Diccionario con estadísticas diarias
        """
        if not fires:
            return {
                'fire_count': 0,
                'Incendio': 0,
                'avg_brightness': None,
                'avg_frp': None
            }
        
        df = pd.DataFrame(fires)
        return {
            'fire_count': len(df),
            'Incendio': 1,  # Variable binaria
            'avg_brightness': df['brightness'].mean(),
            'avg_frp': df['frp'].mean(),  # Fire Radiative Power
            'max_confidence': df['confidence'].max()
        }


def main_pipeline(date: str = None):
    """
    Pipeline principal de recolección de datos.
    
    Args:
        date: Fecha a procesar. Si None, usa fecha de ayer.
    """
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    logger.info(f"=== Iniciando pipeline para {date} ===")
    
    # Inicializar colectores
    colector_de_datos = RecolectorDeDatos(BLEU_API_KEY)
    colector_de_incendios = ColectorDatos(NASA_API_KEY)
    
    # Recolectar datos
    datos_de_clima = colector_de_datos.obtener_datos_diarios(date)
    datos_de_incendios = colector_de_incendios.fetch_datos_de_incendios(date)
    incendios_agregados = colector_de_incendios.agregar_al_diario(datos_de_incendios)
    
    # Combinar datos
    combinado = {**datos_de_clima, **incendios_agregados}
    
    # Guardar en CSV
    output_file = f"data/raw/daily_data_{date}.csv"
    pd.DataFrame([combinado]).to_csv(output_file, index=False)
    
    logger.info(f"✓ Datos guardados en {output_file}")
    logger.info(f"=== Pipeline completado exitosamente ===")
    
    return combinado


if __name__ == "__main__":
    # Ejemplo de uso
    main_pipeline()
