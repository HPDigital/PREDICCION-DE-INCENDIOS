"""
data_collection.py
==================
Script para recolección automatizada de datos meteorológicos y satelitales.
Ejecutar diariamente mediante cron job o Azure Functions.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import os
import time
from dotenv import load_dotenv

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()
BLEU_API_KEY = os.getenv('BLEU_API_KEY', 'YOUR_KEY_HERE')
NASA_API_KEY = os.getenv('NASA_API_KEY', 'YOUR_KEY_HERE')


class RecolectorDatos:
    """Recolector de datos meteorológicos desde BLEU API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
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
        params = {
            'latitude': self.location['latitude'],
            'longitude': self.location['longitude'],
            'start_date': date,
            'end_date': date,
            'daily': [
                'temperature_2m_mean',
                'temperature_2m_max',
                'temperature_2m_min',
                'relative_humidity_2m',
                'precipitation_sum',
                'shortwave_radiation_sum',
                'et0_fao_evapotranspiration',
                'vapor_pressure_deficit_mean'
            ],
            'timezone': 'America/La_Paz'
        }
        
        try:
            response = requests.get(
                self.base_url, 
                params=params, 
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"✓ Datos meteorológicos obtenidos para {date}")
            
            return self._procesar_respuesta(data, date)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error obteniendo datos para {date}: {e}")
            return None
    
    def _procesar_respuesta(self, data: Dict, date: str) -> Dict:
        """Procesar respuesta de la API y extraer variables"""
        
        processed = {'date': date}
        
        # Extraer datos diarios
        daily_data = data['daily']
        
        processed['Temperature'] = daily_data['temperature_2m_mean'][0]
        processed['Temperature_Max'] = daily_data['temperature_2m_max'][0]
        processed['Temperature_Min'] = daily_data['temperature_2m_min'][0]
        processed['Relative_Humidity'] = daily_data['relative_humidity_2m'][0]
        processed['Precipitation'] = daily_data['precipitation_sum'][0]
        processed['Solar_Radiation'] = daily_data['shortwave_radiation_sum'][0]
        processed['Evapotranspiration'] = daily_data['et0_fao_evapotranspiration'][0]
        processed['Vapor_Pressure_Deficit'] = daily_data['vapor_pressure_deficit_mean'][0]
        
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
        logger.info(f"✓ Recolectados {len(df)} días de datos meteorológicos")
        
        return df


class ColectorIncendios:
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
        # Nota: Esta es una implementación simplificada
        # En producción, usar la API real de NASA FIRMS
        
        logger.info(f"Buscando focos de calor para {date}")
        
        # Por ahora retornar estructura vacía
        # En producción, hacer request real a NASA FIRMS API
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
    colector_clima = RecolectorDatos(BLEU_API_KEY)
    colector_incendios = ColectorIncendios(NASA_API_KEY)
    
    # Recolectar datos
    datos_clima = colector_clima.obtener_datos_diarios(date)
    datos_incendios = colector_incendios.fetch_datos_de_incendios(date)
    incendios_agregados = colector_incendios.agregar_al_diario(datos_incendios)
    
    # Combinar datos
    combinado = {**datos_clima, **incendios_agregados}
    
    # Guardar en CSV
    output_file = f"../../data/raw/daily_data_{date}.csv"
    pd.DataFrame([combinado]).to_csv(output_file, index=False)
    
    logger.info(f"✓ Datos guardados en {output_file}")
    logger.info(f"=== Pipeline completado exitosamente ===")
    
    return combinado


if __name__ == "__main__":
    # Ejecutar para ayer
    main_pipeline()
