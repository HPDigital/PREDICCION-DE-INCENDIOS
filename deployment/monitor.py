"""
Sistema de monitoreo en tiempo real para predicción de incendios forestales.
Simula la adquisición de datos en tiempo real y realiza predicciones continuas.
"""

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.predict import FirePredictionSystem


class RealTimeMonitor:
    """
    Sistema de monitoreo en tiempo real para detección de incendios.
    Simula la captura de datos meteorológicos y realiza predicciones continuas.
    """
    
    def __init__(self, predictor: FirePredictionSystem, 
                 max_history: int = 100,
                 alert_threshold: float = 0.7):
        """
        Inicializa el monitor en tiempo real.
        
        Args:
            predictor: Sistema de predicción inicializado
            max_history: Número máximo de observaciones en historial
            alert_threshold: Umbral de confianza para generar alertas
        """
        self.predictor = predictor
        self.max_history = max_history
        self.alert_threshold = alert_threshold
        
        # Historial de datos y predicciones
        self.temperature_history = deque(maxlen=max_history)
        self.humidity_history = deque(maxlen=max_history)
        self.wind_speed_history = deque(maxlen=max_history)
        self.prediction_history = deque(maxlen=max_history)
        self.confidence_history = deque(maxlen=max_history)
        self.timestamp_history = deque(maxlen=max_history)
        
        # Estadísticas
        self.total_readings = 0
        self.total_alerts = 0
        self.start_time = datetime.now()
        
        print("✓ Monitor en tiempo real inicializado")
    
    def generate_sensor_data(self) -> dict:
        """
        Simula la lectura de sensores meteorológicos.
        En producción, esto se reemplazaría con lecturas reales de sensores.
        
        Returns:
            Diccionario con datos simulados de sensores
        """
        # Simular variación temporal (hora del día afecta temperatura)
        hour = datetime.now().hour
        base_temp = 20 + 10 * np.sin((hour - 6) * np.pi / 12)
        
        # Generar datos con algo de ruido aleatorio
        data = {
            'temperatura': base_temp + np.random.normal(0, 3),
            'humedad': max(20, min(100, 50 + np.random.normal(0, 15))),
            'velocidad_viento': max(0, np.random.exponential(8)),
            'precipitacion': max(0, np.random.exponential(0.5) if np.random.random() < 0.2 else 0),
            'indice_sequedad': max(0, min(100, 50 + np.random.normal(0, 20))),
            'vegetacion_seca': max(0, min(1, np.random.beta(2, 5))),
            'mes': datetime.now().month,
            'dia_semana': datetime.now().weekday()
        }
        
        return data
    
    def process_reading(self, data: dict) -> dict:
        """
        Procesa una lectura de sensores y realiza predicción.
        
        Args:
            data: Datos de sensores
            
        Returns:
            Resultado de la predicción con metadatos
        """
        # Realizar predicción
        result = self.predictor.predict_with_details(data)
        
        # Actualizar historial
        self.temperature_history.append(data['temperatura'])
        self.humidity_history.append(data['humedad'])
        self.wind_speed_history.append(data['velocidad_viento'])
        self.prediction_history.append(result['prediction'])
        self.confidence_history.append(result['confidence'] if result['confidence'] else 0)
        self.timestamp_history.append(datetime.now())
        
        # Actualizar estadísticas
        self.total_readings += 1
        
        # Verificar si se debe generar alerta
        if result['confidence'] and result['confidence'] > self.alert_threshold and result['prediction'] >= 2:
            self.total_alerts += 1
            result['alert'] = True
        else:
            result['alert'] = False
        
        return result
    
    def display_reading(self, data: dict, result: dict):
        """
        Muestra la lectura actual en consola.
        
        Args:
            data: Datos de sensores
            result: Resultado de predicción
        """
        print("\n" + "=" * 70)
        print(f"LECTURA #{self.total_readings} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Datos de sensores
        print("\n📊 DATOS DE SENSORES:")
        print(f"  🌡️  Temperatura: {data['temperatura']:.1f}°C")
        print(f"  💧 Humedad: {data['humedad']:.1f}%")
        print(f"  💨 Velocidad del viento: {data['velocidad_viento']:.1f} km/h")
        print(f"  🌧️  Precipitación: {data['precipitacion']:.1f} mm")
        print(f"  🌵 Índice de sequedad: {data['indice_sequedad']:.1f}")
        
        # Resultado de predicción
        print("\n🔮 PREDICCIÓN:")
        print(f"  Clase: {result['prediction_label']}")
        if result['confidence']:
            print(f"  Confianza: {result['confidence']:.1%}")
        print(f"  Nivel de Riesgo: {result['risk_level']}")
        
        # Alerta si es necesario
        if result.get('alert', False):
            print("\n⚠️  ¡ALERTA! Se detectó riesgo elevado de incendio")
            print(f"  Total de alertas: {self.total_alerts}")
        
        print("=" * 70)
    
    def get_statistics(self) -> dict:
        """
        Obtiene estadísticas del monitoreo.
        
        Returns:
            Diccionario con estadísticas
        """
        uptime = datetime.now() - self.start_time
        
        stats = {
            'total_readings': self.total_readings,
            'total_alerts': self.total_alerts,
            'alert_rate': self.total_alerts / self.total_readings if self.total_readings > 0 else 0,
            'uptime_seconds': uptime.total_seconds(),
            'uptime_formatted': str(uptime),
            'avg_temperature': np.mean(list(self.temperature_history)) if self.temperature_history else 0,
            'avg_humidity': np.mean(list(self.humidity_history)) if self.humidity_history else 0,
            'avg_wind_speed': np.mean(list(self.wind_speed_history)) if self.wind_speed_history else 0,
        }
        
        return stats
    
    def plot_realtime_dashboard(self, save_path: str = None):
        """
        Genera un dashboard con los datos en tiempo real.
        
        Args:
            save_path: Ruta donde guardar la imagen (opcional)
        """
        if len(self.timestamp_history) < 2:
            print("No hay suficientes datos para generar el dashboard")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dashboard de Monitoreo en Tiempo Real', fontsize=16, fontweight='bold')
        
        # Convertir timestamps a formato numérico para graficar
        time_axis = [(t - self.timestamp_history[0]).total_seconds() / 60 
                     for t in self.timestamp_history]
        
        # Temperatura
        axes[0, 0].plot(time_axis, list(self.temperature_history), 'r-', linewidth=2)
        axes[0, 0].set_title('Temperatura', fontweight='bold')
        axes[0, 0].set_ylabel('°C')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlabel('Tiempo (minutos)')
        
        # Humedad
        axes[0, 1].plot(time_axis, list(self.humidity_history), 'b-', linewidth=2)
        axes[0, 1].set_title('Humedad', fontweight='bold')
        axes[0, 1].set_ylabel('%')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlabel('Tiempo (minutos)')
        
        # Velocidad del viento
        axes[1, 0].plot(time_axis, list(self.wind_speed_history), 'g-', linewidth=2)
        axes[1, 0].set_title('Velocidad del Viento', fontweight='bold')
        axes[1, 0].set_ylabel('km/h')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlabel('Tiempo (minutos)')
        
        # Nivel de confianza de predicciones
        axes[1, 1].plot(time_axis, list(self.confidence_history), 'orange', linewidth=2)
        axes[1, 1].axhline(y=self.alert_threshold, color='r', linestyle='--', 
                          label=f'Umbral de alerta ({self.alert_threshold})')
        axes[1, 1].set_title('Confianza de Predicción', fontweight='bold')
        axes[1, 1].set_ylabel('Confianza')
        axes[1, 1].set_xlabel('Tiempo (minutos)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Dashboard guardado en: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def run_continuous(self, interval: int = 5, duration: int = 60):
        """
        Ejecuta el monitoreo continuo por un período determinado.
        
        Args:
            interval: Intervalo entre lecturas en segundos
            duration: Duración total del monitoreo en segundos
        """
        print("\n" + "=" * 70)
        print("INICIANDO MONITOREO EN TIEMPO REAL")
        print("=" * 70)
        print(f"Intervalo: {interval} segundos")
        print(f"Duración: {duration} segundos")
        print("Presione Ctrl+C para detener")
        print("=" * 70)
        
        end_time = datetime.now() + timedelta(seconds=duration)
        
        try:
            while datetime.now() < end_time:
                # Generar datos de sensores
                sensor_data = self.generate_sensor_data()
                
                # Procesar lectura
                result = self.process_reading(sensor_data)
                
                # Mostrar resultados
                self.display_reading(sensor_data, result)
                
                # Esperar antes de la siguiente lectura
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Monitoreo detenido por el usuario")
        
        # Mostrar estadísticas finales
        print("\n" + "=" * 70)
        print("ESTADÍSTICAS FINALES")
        print("=" * 70)
        stats = self.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("=" * 70)
        
        # Generar dashboard final
        print("\nGenerando dashboard final...")
        self.plot_realtime_dashboard('./results/realtime_dashboard.png')
        
        print("\n✓ Monitoreo completado")


def main():
    """Función principal para ejecutar el monitor."""
    print("=" * 70)
    print("SISTEMA DE MONITOREO EN TIEMPO REAL")
    print("=" * 70)
    
    # Configuración
    model_path = './models/random_forest_model.pkl'
    scaler_path = './models/scaler.pkl'
    
    # Verificar que el modelo existe
    if not Path(model_path).exists():
        print(f"\n⚠️  Error: No se encontró el modelo en {model_path}")
        print("Por favor, entrena un modelo primero usando train_random_forest.py")
        return
    
    # Inicializar sistema de predicción
    print("\nInicializando sistema de predicción...")
    try:
        predictor = FirePredictionSystem(
            model_path=model_path,
            scaler_path=scaler_path if Path(scaler_path).exists() else None
        )
    except Exception as e:
        print(f"\n⚠️  Error al inicializar el sistema: {e}")
        return
    
    # Crear monitor
    monitor = RealTimeMonitor(
        predictor=predictor,
        max_history=100,
        alert_threshold=0.7
    )
    
    # Ejecutar monitoreo continuo
    monitor.run_continuous(interval=5, duration=60)


if __name__ == "__main__":
    main()
