"""
Configuración centralizada del proyecto de predicción de compras.
Todas las constantes, paths, dtypes y parámetros compartidos van aquí.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# ── Directorios ──────────────────────────────────────────────────────────────
# En local:  APP_DIR=.../Final/app, DATA_DIR=.../Final/data
# En Docker: APP_DIR=/app,          DATA_DIR=/data  (env DATA_DIR=/data)
import os
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", APP_DIR.parent / "data"))
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features_store"
MODELS_DIR = DATA_DIR / "models"

# Crear directorios si no existen
for _d in (DATA_DIR, RAW_DIR, FEATURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Ventanas temporales ─────────────────────────────────────────────────────
PREDICTION_WINDOW_DAYS = 21          # ventana de predicción / target
BATCH_SIZE_FAMILIES = 300            # familias por lote de procesamiento
NEW_TX_THRESHOLD = 300               # umbral de nuevas transacciones para disparar reentrenamiento

# ── Ventanas de features ─────────────────────────────────────────────────────
RECENT_WINDOW_DAYS = 60
FREQUENCY_WINDOW_DAYS = 180
SOW_MONTHS_24 = 24
SOW_MONTHS_12 = 12

# ── Dtypes para lectura del CSV crudo ────────────────────────────────────────
RAW_DTYPES = {
    "COD_SUBCATEGORIA": "Int64",
    "COD_CATEGORIA": "Int64",
    "COD_UNIDAD_COMERCIAL": "Int64",
    "COD_ITEM": "Int64",
    "DIM_FACTURA": "Int64",
    "COD_LOCAL": "Int64",
    "CODIGO_FAMILIA": "Int64",
}

NUMERIC_COLUMNS = ["CANTIDAD_SUELTA", "PVP", "VENTA_NETA", "DESCUENTO"]
PARSE_DATES = ["DIM_PERIODO"]

# ── Columnas de features que usa el modelo FNN ───────────────────────────────
FEATURE_COLUMNS = [
    "recencia_hl",
    "freq_baja",
    "freq_media",
    "freq_alta",
    "cv_invertido",
    "sow_24m",
    "season_ratio",
    "score_final",
    "ciclo_dias_mu",
]

# ── Archivos de salida por defecto ───────────────────────────────────────────
FEATURES_TRAIN_FILE = FEATURES_DIR / "features_train.parquet"
FEATURES_INFERENCE_FILE = FEATURES_DIR / "features_inference.parquet"
MODEL_FILE = MODELS_DIR / "fnn_model.pth"
MODEL_META_FILE = MODELS_DIR / "fnn_meta.json"
