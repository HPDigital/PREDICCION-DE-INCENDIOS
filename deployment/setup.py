"""
Script de configuración y despliegue del proyecto.
Automatiza la preparación del entorno y el despliegue de los servicios.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json


class DeploymentManager:
    """Gestor de despliegue del proyecto."""
    
    def __init__(self, project_root: str = None):
        """
        Inicializa el gestor de despliegue.
        
        Args:
            project_root: Directorio raíz del proyecto
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.models_dir = self.project_root / 'models'
        self.results_dir = self.project_root / 'results'
        self.logs_dir = self.project_root / 'logs'
        self.deployment_dir = self.project_root / 'deployment'
        
        print(f"📂 Directorio del proyecto: {self.project_root}")
    
    def check_prerequisites(self) -> bool:
        """
        Verifica que todos los prerequisitos estén instalados.
        
        Returns:
            bool: True si todos los prerequisitos están disponibles
        """
        print("\n" + "=" * 70)
        print("VERIFICANDO PREREQUISITOS")
        print("=" * 70)
        
        prerequisites = {
            'python': ['python', '--version'],
            'pip': ['pip', '--version'],
            'docker': ['docker', '--version'],
            'docker-compose': ['docker-compose', '--version']
        }
        
        all_ok = True
        
        for name, command in prerequisites.items():
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True
                )
                version = result.stdout.strip()
                print(f"✓ {name}: {version}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"✗ {name}: NO DISPONIBLE")
                all_ok = False
        
        return all_ok
    
    def create_directories(self):
        """Crea los directorios necesarios si no existen."""
        print("\n" + "=" * 70)
        print("CREANDO ESTRUCTURA DE DIRECTORIOS")
        print("=" * 70)
        
        directories = [
            self.models_dir,
            self.results_dir,
            self.logs_dir,
            self.deployment_dir
        ]
        
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                print(f"✓ Creado: {directory}")
            else:
                print(f"✓ Existe: {directory}")
    
    def check_model_files(self) -> bool:
        """
        Verifica que los archivos del modelo existan.
        
        Returns:
            bool: True si los archivos existen
        """
        print("\n" + "=" * 70)
        print("VERIFICANDO ARCHIVOS DEL MODELO")
        print("=" * 70)
        
        model_file = self.models_dir / 'random_forest_model.pkl'
        scaler_file = self.models_dir / 'scaler.pkl'
        
        model_exists = model_file.exists()
        scaler_exists = scaler_file.exists()
        
        if model_exists:
            print(f"✓ Modelo encontrado: {model_file}")
        else:
            print(f"✗ Modelo NO encontrado: {model_file}")
            print("  Ejecuta 'train_random_forest.py' para entrenar el modelo")
        
        if scaler_exists:
            print(f"✓ Scaler encontrado: {scaler_file}")
        else:
            print(f"⚠ Scaler NO encontrado: {scaler_file}")
            print("  El sistema funcionará sin normalización")
        
        return model_exists
    
    def install_dependencies(self, dev: bool = False):
        """
        Instala las dependencias del proyecto.
        
        Args:
            dev: Si True, instala también las dependencias de desarrollo
        """
        print("\n" + "=" * 70)
        print("INSTALANDO DEPENDENCIAS")
        print("=" * 70)
        
        requirements_file = self.project_root / 'requirements.txt'
        
        if not requirements_file.exists():
            print(f"✗ No se encontró {requirements_file}")
            return False
        
        try:
            print(f"📦 Instalando desde {requirements_file}...")
            subprocess.run(
                ['pip', 'install', '-r', str(requirements_file)],
                check=True
            )
            print("✓ Dependencias instaladas correctamente")
            
            if dev:
                print("\n📦 Instalando dependencias de desarrollo...")
                subprocess.run(
                    ['pip', 'install', 'pytest', 'pytest-cov', 'black', 'flake8'],
                    check=True
                )
                print("✓ Dependencias de desarrollo instaladas")
            
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"✗ Error al instalar dependencias: {e}")
            return False
    
    def run_tests(self) -> bool:
        """
        Ejecuta los tests del proyecto.
        
        Returns:
            bool: True si todos los tests pasan
        """
        print("\n" + "=" * 70)
        print("EJECUTANDO TESTS")
        print("=" * 70)
        
        test_script = self.project_root / 'tests' / 'run_tests.py'
        
        if not test_script.exists():
            print(f"✗ No se encontró el script de tests: {test_script}")
            return False
        
        try:
            result = subprocess.run(
                ['python', str(test_script)],
                capture_output=False,
                check=True
            )
            print("\n✓ Todos los tests pasaron")
            return True
        
        except subprocess.CalledProcessError:
            print("\n✗ Algunos tests fallaron")
            return False
    
    def build_docker_images(self):
        """Construye las imágenes de Docker."""
        print("\n" + "=" * 70)
        print("CONSTRUYENDO IMÁGENES DE DOCKER")
        print("=" * 70)
        
        try:
            print("🐳 Construyendo imagen de la API...")
            subprocess.run(
                ['docker', 'build', '-t', 'fire-prediction-api', '-f', 'Dockerfile', '.'],
                cwd=self.project_root,
                check=True
            )
            print("✓ Imagen de API construida")
            
            print("\n🐳 Construyendo imagen del dashboard...")
            subprocess.run(
                ['docker', 'build', '-t', 'fire-prediction-dashboard', 
                 '-f', 'Dockerfile.streamlit', '.'],
                cwd=self.project_root,
                check=True
            )
            print("✓ Imagen del dashboard construida")
            
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"✗ Error al construir imágenes: {e}")
            return False
    
    def deploy_docker_compose(self):
        """Despliega usando docker-compose."""
        print("\n" + "=" * 70)
        print("DESPLEGANDO CON DOCKER COMPOSE")
        print("=" * 70)
        
        try:
            print("🚀 Iniciando servicios...")
            subprocess.run(
                ['docker-compose', 'up', '-d'],
                cwd=self.project_root,
                check=True
            )
            
            print("\n✓ Servicios desplegados correctamente")
            print("\n📡 Servicios disponibles:")
            print("   - API: http://localhost:5000")
            print("   - Dashboard: http://localhost:8501")
            print("   - Health Check: http://localhost:5000/health")
            
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"✗ Error al desplegar: {e}")
            return False
    
    def stop_services(self):
        """Detiene los servicios de Docker."""
        print("\n" + "=" * 70)
        print("DETENIENDO SERVICIOS")
        print("=" * 70)
        
        try:
            subprocess.run(
                ['docker-compose', 'down'],
                cwd=self.project_root,
                check=True
            )
            print("✓ Servicios detenidos")
            return True
        
        except subprocess.CalledProcessError as e:
            print(f"✗ Error al detener servicios: {e}")
            return False
    
    def view_logs(self, service: str = None):
        """
        Muestra los logs de los servicios.
        
        Args:
            service: Nombre del servicio específico (opcional)
        """
        print("\n" + "=" * 70)
        print("LOGS DE SERVICIOS")
        print("=" * 70)
        
        command = ['docker-compose', 'logs', '-f']
        if service:
            command.append(service)
        
        try:
            subprocess.run(
                command,
                cwd=self.project_root
            )
        except KeyboardInterrupt:
            print("\n\n✓ Visualización de logs detenida")
    
    def create_env_file(self):
        """Crea archivo .env con configuración por defecto."""
        print("\n" + "=" * 70)
        print("CREANDO ARCHIVO DE CONFIGURACIÓN")
        print("=" * 70)
        
        env_file = self.project_root / '.env'
        
        if env_file.exists():
            print(f"⚠ El archivo .env ya existe")
            return
        
        env_content = """# Configuración del proyecto
FLASK_ENV=production
MODEL_PATH=./models/random_forest_model.pkl
SCALER_PATH=./models/scaler.pkl
API_HOST=0.0.0.0
API_PORT=5000
MONITOR_INTERVAL=60
"""
        
        env_file.write_text(env_content)
        print(f"✓ Archivo .env creado: {env_file}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Gestor de despliegue del proyecto de predicción de incendios'
    )
    
    parser.add_argument(
        'action',
        choices=['setup', 'test', 'build', 'deploy', 'stop', 'logs', 'full'],
        help='Acción a realizar'
    )
    
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Instalar dependencias de desarrollo'
    )
    
    parser.add_argument(
        '--service',
        type=str,
        help='Servicio específico para ver logs'
    )
    
    args = parser.parse_args()
    
    # Crear gestor de despliegue
    manager = DeploymentManager()
    
    # Ejecutar acción
    if args.action == 'setup':
        manager.check_prerequisites()
        manager.create_directories()
        manager.create_env_file()
        manager.install_dependencies(dev=args.dev)
        manager.check_model_files()
    
    elif args.action == 'test':
        if manager.check_model_files():
            manager.run_tests()
        else:
            print("\n⚠ No se pueden ejecutar tests sin el modelo entrenado")
    
    elif args.action == 'build':
        if manager.check_prerequisites():
            manager.build_docker_images()
    
    elif args.action == 'deploy':
        if manager.check_model_files():
            manager.deploy_docker_compose()
        else:
            print("\n⚠ No se puede desplegar sin el modelo entrenado")
    
    elif args.action == 'stop':
        manager.stop_services()
    
    elif args.action == 'logs':
        manager.view_logs(service=args.service)
    
    elif args.action == 'full':
        # Despliegue completo
        print("\n🚀 INICIANDO DESPLIEGUE COMPLETO")
        
        if not manager.check_prerequisites():
            print("\n✗ Faltan prerequisitos necesarios")
            sys.exit(1)
        
        manager.create_directories()
        manager.create_env_file()
        
        if not manager.install_dependencies(dev=args.dev):
            print("\n✗ Error al instalar dependencias")
            sys.exit(1)
        
        if not manager.check_model_files():
            print("\n✗ No se encontró el modelo entrenado")
            print("  Ejecuta primero: python src/models/train_random_forest.py")
            sys.exit(1)
        
        print("\n⚠ Omitiendo tests para despliegue rápido")
        print("  Ejecuta 'python deployment/setup.py test' para correr tests")
        
        if not manager.build_docker_images():
            print("\n✗ Error al construir imágenes")
            sys.exit(1)
        
        if not manager.deploy_docker_compose():
            print("\n✗ Error al desplegar servicios")
            sys.exit(1)
        
        print("\n" + "=" * 70)
        print("✓ DESPLIEGUE COMPLETO EXITOSO")
        print("=" * 70)
        print("\n🌐 Accede a los servicios:")
        print("   - API: http://localhost:5000")
        print("   - Dashboard: http://localhost:8501")
        print("\n💡 Comandos útiles:")
        print("   - Ver logs: python deployment/setup.py logs")
        print("   - Detener servicios: python deployment/setup.py stop")
        print("=" * 70)


if __name__ == '__main__':
    main()
