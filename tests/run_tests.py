"""
Script para ejecutar todos los tests del proyecto.
"""

import sys
import unittest
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from tests.test_preprocessing import TestDataPreprocessor, TestDataIntegrity
from tests.test_prediction import TestFirePredictionSystem, TestCreateExampleInput


def run_all_tests():
    """
    Ejecuta todos los tests del proyecto y genera un reporte.
    
    Returns:
        bool: True si todos los tests pasaron, False en caso contrario
    """
    print("=" * 80)
    print("EJECUTANDO SUITE COMPLETA DE TESTS")
    print("=" * 80)
    print()
    
    # Crear suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Tests de preprocesamiento
    print("📦 Cargando tests de preprocesamiento...")
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrity))
    
    # Tests de predicción
    print("📦 Cargando tests de predicción...")
    suite.addTests(loader.loadTestsFromTestCase(TestFirePredictionSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestCreateExampleInput))
    
    
    print()
    print("=" * 80)
    print("INICIANDO EJECUCIÓN DE TESTS")
    print("=" * 80)
    print()
    
    # Ejecutar tests con verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generar reporte final
    print()
    print("=" * 80)
    print("REPORTE FINAL")
    print("=" * 80)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"✓ Exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"✗ Fallos: {len(result.failures)}")
    print(f"⚠ Errores: {len(result.errors)}")
    print(f"⊘ Omitidos: {len(result.skipped)}")
    print("=" * 80)
    
    # Mostrar detalles de fallos si los hay
    if result.failures:
        print()
        print("DETALLES DE FALLOS:")
        print("-" * 80)
        for test, traceback in result.failures:
            print(f"\n{test}:")
            print(traceback)
    
    # Mostrar detalles de errores si los hay
    if result.errors:
        print()
        print("DETALLES DE ERRORES:")
        print("-" * 80)
        for test, traceback in result.errors:
            print(f"\n{test}:")
            print(traceback)
    
    # Determinar si todos los tests pasaron
    success = result.wasSuccessful()
    
    if success:
        print()
        print("🎉 ¡TODOS LOS TESTS PASARON EXITOSAMENTE!")
    else:
        print()
        print("❌ ALGUNOS TESTS FALLARON")
    
    return success


def run_specific_test(test_class_name: str = None):
    """
    Ejecuta un test específico.
    
    Args:
        test_class_name: Nombre de la clase de test a ejecutar
    """
    if not test_class_name:
        print("Por favor especifica el nombre de la clase de test")
        return False
    
    # Mapeo de nombres a clases
    test_classes = {
        'TestDataPreprocessor': TestDataPreprocessor,
        'TestDataIntegrity': TestDataIntegrity,
        'TestFirePredictionSystem': TestFirePredictionSystem,
        'TestCreateExampleInput': TestCreateExampleInput,
    }
    
    if test_class_name not in test_classes:
        print(f"Test '{test_class_name}' no encontrado")
        print(f"Tests disponibles: {', '.join(test_classes.keys())}")
        return False
    
    print(f"Ejecutando {test_class_name}...")
    print("=" * 80)
    
    # Ejecutar test específico
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(test_classes[test_class_name])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_tests_by_module(module: str):
    """
    Ejecuta tests de un módulo específico.
    
    Args:
        module: 'preprocessing' o 'prediction'
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if module == 'preprocessing':
        print("Ejecutando tests de preprocesamiento...")
        suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessor))
        suite.addTests(loader.loadTestsFromTestCase(TestDataIntegrity))
    elif module == 'prediction':
        print("Ejecutando tests de predicción...")
        suite.addTests(loader.loadTestsFromTestCase(TestFirePredictionSystem))
        suite.addTests(loader.loadTestsFromTestCase(TestCreateExampleInput))
    else:
        print(f"Módulo '{module}' no reconocido")
        print("Módulos disponibles: preprocessing, prediction")
        return False
    
    print("=" * 80)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Ejecutar tests del proyecto de predicción de incendios'
    )
    parser.add_argument(
        '--test',
        type=str,
        help='Nombre de la clase de test específica a ejecutar'
    )
    parser.add_argument(
        '--module',
        type=str,
        choices=['preprocessing', 'prediction'],
        help='Ejecutar tests de un módulo específico'
    )
    
    args = parser.parse_args()
    
    # Determinar qué tests ejecutar
    if args.test:
        success = run_specific_test(args.test)
    elif args.module:
        success = run_tests_by_module(args.module)
    else:
        success = run_all_tests()
    
    # Salir con código apropiado
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
