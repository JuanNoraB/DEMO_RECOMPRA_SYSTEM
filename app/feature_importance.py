"""
feature_importance.py — Análisis de importancia de features con LightGBM.

Qué hace:
  1. Carga parquet de features.
  2. Usa por defecto todas las columnas numéricas reales del parquet.
  3. Convierte la columna categórica de ciclo a one-hot si existe.
  4. Entrena LightGBM.
  5. Extrae importancia por GAIN y SPLIT.
  6. Imprime únicamente las features que acumulan al menos 95% del gain.
  7. Guarda CSV en docs/feature_analysis/feature_summary.csv.
  8. Guarda log JSONL en data/logs/feature_importance_runs.jsonl.
  9. Guarda