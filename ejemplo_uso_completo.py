"""
ejemplo_uso_completo.py
=======================
Script de ejemplo mostrando el uso completo del proyecto
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("EJEMPLO DE USO COMPLETO - PREDICCIÓN DE INCENDIOS FORESTALES")
print("="*70)

# ============================================================
# PARTE 1: CARGA Y PREPARACIÓN DE DATOS
# ============================================================

print("\n📊 PARTE 1: CARGA Y PREPARACIÓN DE DATOS")
print("-" * 70)

# Importar módulos personalizados
import sys
sys.path.append('src')

from data.preprocesamiento_de_datos import DataPreprocesador

# Inicializar preprocesador
print("\n1. Inicializando preprocesador...")
preprocessor = DataPreprocesador()

# Simular datos de ejemplo (en producción, usar datos reales)
print("2. Creando datos de ejemplo...")
n_samples = 100
ejemplo_data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=n_samples),
    'Temperature': np.random.uniform(15, 35, n_samples),
    'Relative_Humidity': np.random.uniform(20, 90, n_samples),
    'Solar_Radiation': np.random.uniform(2000, 6000, n_samples),
    'Soil_Moisture': np.random.uniform(0.1, 0.3, n_samples),
    'Cloud_Cover': np.random.uniform(0, 100, n_samples),
    'Precipitation': np.random.uniform(0, 50, n_samples),
    'Evapotranspiration': np.random.uniform(0, 10, n_samples),
    'Vapor_Pressure_Deficit': np.random.uniform(0, 5, n_samples),
    'Incendio': np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
})

print(f"✓ Datos de ejemplo creados: {len(ejemplo_data)} registros")
print(f"✓ Variables: {list(ejemplo_data.columns)}")

# Guardar datos de ejemplo
ejemplo_data.to_csv('data/raw/ejemplo_data.csv', index=False)
print("✓ Datos guardados en: data/raw/ejemplo_data.csv")

# ============================================================
# PARTE 2: ANÁLISIS EXPLORATORIO BÁSICO
# ============================================================

print("\n📈 PARTE 2: ANÁLISIS EXPLORATORIO BÁSICO")
print("-" * 70)

print("\nEstadísticas Descriptivas:")
print(ejemplo_data.describe().round(2))

print(f"\nDistribución de la Variable Objetivo:")
print(ejemplo_data['Incendio'].value_counts())
print(f"Proporción de incendios: {ejemplo_data['Incendio'].mean():.2%}")

# Correlación con target
print("\nCorrelaciones con Variable Objetivo:")
correlaciones = ejemplo_data.corr()['Incendio'].sort_values(ascending=False)
for var, corr in correlaciones.items():
    if var != 'Incendio':
        print(f"  • {var}: {corr:+.3f}")

# ============================================================
# PARTE 3: ENTRENAMIENTO DEL MODELO
# ============================================================

print("\n🧠 PARTE 3: ENTRENAMIENTO DEL MODELO")
print("-" * 70)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Preparar datos
print("\n1. Preparando datos para entrenamiento...")
X = ejemplo_data.drop(['Incendio', 'date'], axis=1)
y = ejemplo_data['Incendio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train set: {len(X_train)} muestras")
print(f"✓ Test set: {len(X_test)} muestras")

# Entrenar modelo
print("\n2. Entrenando Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✓ Modelo entrenado exitosamente")

# ============================================================
# PARTE 4: EVALUACIÓN DEL MODELO
# ============================================================

print("\n📊 PARTE 4: EVALUACIÓN DEL MODELO")
print("-" * 70)

# Predicciones
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Métricas
print("\n1. Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Fire', 'Fire']))

print("\n2. Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n3. Métricas Adicionales:")
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"✓ AUC-ROC: {auc_score:.3f}")

# Feature Importance
print("\n4. Top 5 Features Más Importantes:")
feature_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_imp.head(5).iterrows():
    print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")

# ============================================================
# PARTE 5: HACER PREDICCIONES
# ============================================================

print("\n🔮 PARTE 5: HACER PREDICCIONES")
print("-" * 70)

# Ejemplo de nuevos datos
print("\n1. Simulando observación meteorológica del día:")
nuevo_dato = pd.DataFrame({
    'Temperature': [32.5],
    'Relative_Humidity': [25.0],
    'Solar_Radiation': [5200],
    'Soil_Moisture': [0.12],
    'Cloud_Cover': [10],
    'Precipitation': [0],
    'Evapotranspiration': [7.5],
    'Vapor_Pressure_Deficit': [4.2]
})

print(nuevo_dato.T)

# Predicción
print("\n2. Realizando predicción...")
prediccion = model.predict(nuevo_dato)[0]
probabilidad = model.predict_proba(nuevo_dato)[0, 1]

print(f"\n✓ Predicción: {'FUEGO' if prediccion == 1 else 'NO FUEGO'}")
print(f"✓ Probabilidad de incendio: {probabilidad:.2%}")

# Clasificar riesgo
if probabilidad < 0.30:
    nivel_riesgo = "BAJO"
    emoji = "🟢"
elif probabilidad < 0.60:
    nivel_riesgo = "MEDIO"
    emoji = "🟡"
else:
    nivel_riesgo = "ALTO"
    emoji = "🔴"

print(f"✓ Nivel de Riesgo: {emoji} {nivel_riesgo}")

# ============================================================
# PARTE 6: GUARDAR MODELO
# ============================================================

print("\n💾 PARTE 6: GUARDAR MODELO")
print("-" * 70)

import joblib

model_path = 'models/ejemplo_random_forest.pkl'
joblib.dump(model, model_path)
print(f"✓ Modelo guardado en: {model_path}")

# Verificar tamaño del modelo
import os
size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"✓ Tamaño del modelo: {size_mb:.2f} MB")

# ============================================================
# RESUMEN FINAL
# ============================================================

print("\n" + "="*70)
print("✅ EJEMPLO COMPLETADO EXITOSAMENTE")
print("="*70)

print("\n📋 RESUMEN:")
print(f"  • Datos procesados: {len(ejemplo_data)} registros")
print(f"  • Modelo entrenado: Random Forest (100 árboles)")
print(f"  • AUC-ROC: {auc_score:.3f}")
print(f"  • Accuracy: {(y_pred == y_test).mean():.3f}")
print(f"  • Modelo guardado en: {model_path}")

print("\n🎯 PRÓXIMOS PASOS:")
print("  1. Ejecutar con datos reales de las APIs")
print("  2. Optimizar hiperparámetros con RandomizedSearchCV")
print("  3. Entrenar con Azure AutoML para mejores resultados")
print("  4. Desplegar como API REST en Azure")

print("\n📚 DOCUMENTACIÓN:")
print("  • README.md - Documentación completa del proyecto")
print("  • QUICK_START.md - Guía de inicio rápido")
print("  • notebooks/ - Análisis detallados en Jupyter")

print("\n" + "="*70)
print("¡Gracias por usar este proyecto! 🚀")
print("="*70)
