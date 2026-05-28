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
    # ── Top features por gain >= 1% (LightGBM) ─────────────────────────
    "recencia_hl",                # gain 1.77%
    "sow_24m",                    # gain 4.72%
    "score_final",                # gain 6.10%
    "ciclo_dias_mu",              # gain 12.62%
    "dias_desde_ultima_compra",   # gain 15.78% (NUEVA)
    "ticket_promedio",            # gain  6.49% (NUEVA)
    "n_subcats_familia",          # gain  5.92% (NUEVA)
    # NOTA: Debug_ciclos_tipo_ciclo_b genera 4 one-hots (tipo_corto_medio,
    # tipo_largo, tipo_mediano, tipo_no_ciclico) que se añaden automáticamente
    # → tipo_no_ciclico (40.34%) y tipo_largo (3.17%) son top features.
    # Total input al modelo: 7 numéricas + 4 one-hots = 11 features.
    #
    # ── Comentadas (gain < 1%, contribución marginal) ──────────────────
    # "freq_baja",                 # gain 0.009%
    # "freq_media",                # gain 0.008%
    # "freq_alta",                 # gain 0.019%
    # "cv_invertido",              # gain 0.562%
    # "season_ratio",              # gain 0.188%
    # "Ciclos_ciclo_binario_c",    # gain 0.001%
    # "l_compra_sobre_ciclo",      # gain 0.727% (NUEVA)
    # "compras_reales",            # gain 0.490% (NUEVA)
    # "ratio_temporal",            # gain 0.058% (NUEVA)
]

TIPO_CICLO_COL = "Debug_ciclos_tipo_ciclo_b"
TIPO_CICLO_CATEGORIES = ["corto", "corto_medio", "largo", "mediano", "no_ciclico"]

# ── Datos crudos ─────────────────────────────────────────────────────────────
# Sobreescribir con variable de entorno HISTORICO_PATH si se necesita
HISTORICO_FILE = Path(os.environ.get("HISTORICO_PATH", APP_DIR.parent / "Historico_08122025.csv"))

# ── Archivos de salida por defecto ───────────────────────────────────────────
FEATURES_TRAIN_FILE = FEATURES_DIR / "features_train.parquet"
FEATURES_EVAL_FILE = FEATURES_DIR / "features_eval.parquet"
FEATURES_INFERENCE_FILE = FEATURES_DIR / "features_inference.parquet"
FEATURES_HPT_TRAIN_FILE = FEATURES_DIR / "features_hpt_train.parquet"
MODEL_FILE = MODELS_DIR / "fnn_model.pth"
MODEL_META_FILE = MODELS_DIR / "fnn_meta.json"
BEST_HPARAMS_FILE = MODELS_DIR / "best_hparams.json"
