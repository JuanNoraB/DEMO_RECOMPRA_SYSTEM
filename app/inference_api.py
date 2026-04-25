"""
inference_api.py — API REST de inferencia con FastAPI.

Endpoints:
  GET /predict/{cedula}  → Top 3 subcategorías con mayor probabilidad de compra.
  GET /health            → Estado del servicio.

Carga el modelo FNN y las features pre-calculadas al iniciar.
Hot-reload: cada 2 minutos chequea si el modelo cambió en disco y recarga.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_client import Counter, Histogram, make_asgi_app

from config import (
    FEATURES_TRAIN_FILE,
    FEATURES_INFERENCE_FILE,
    MODEL_FILE,
    MODEL_META_FILE,
    FEATURE_COLUMNS,
)
from train_fnn import _build_features

RELOAD_INTERVAL = int(os.environ.get("RELOAD_INTERVAL", 120))
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "kafka-svc:9092")

# ── Métricas Prometheus ───────────────────────────────────────────────────────
REQUEST_COUNT   = Counter("requests_total", "Total requests", ["status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency")

# ── Kafka producer (simple) ───────────────────────────────────────────────────
_kafka_producer = None

def _get_kafka():
    global _kafka_producer
    if _kafka_producer is None:
        try:
            from kafka import KafkaProducer
            _kafka_producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER)
            print(f"[Kafka] Producer conectado a {KAFKA_BROKER}")
        except Exception as e:
            print(f"[Kafka] No disponible: {e}")
            _kafka_producer = False
    return _kafka_producer if _kafka_producer else None

# ── Estado global (se carga al iniciar) ──────────────────────────────────────
MODEL = None
META = None
FEATURES_DF = None
_model_mtime = 0.0
_features_mtime = 0.0
_reload_lock = threading.Lock()


def _get_features_file() -> Path:
    """Retorna el archivo de features a usar: inference si existe, sino train."""
    if FEATURES_INFERENCE_FILE.exists():
        return FEATURES_INFERENCE_FILE
    return FEATURES_TRAIN_FILE


def _load_model():
    """Carga modelo + meta + features al arrancar o recargar."""
    global MODEL, META, FEATURES_DF, _model_mtime, _features_mtime

    if not MODEL_FILE.exists() or not MODEL_META_FILE.exists():
        print("[API] Modelo no disponible aún. Esperando...")
        return False

    features_file = _get_features_file()
    if not features_file.exists():
        print("[API] Features no disponibles aún. Esperando...")
        return False

    with open(MODEL_META_FILE) as f:
        meta = json.load(f)

    from train_fnn import PurchaseFNN
    device = "cpu"
    model = PurchaseFNN(input_dim=meta["input_dim"]).to(device)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device, weights_only=True))
    model.eval()

    features_df = pd.read_parquet(features_file)

    with _reload_lock:
        MODEL = model
        META = meta
        FEATURES_DF = features_df
        _model_mtime = MODEL_FILE.stat().st_mtime
        _features_mtime = features_file.stat().st_mtime

    print(f"[API] Modelo cargado: {meta['input_dim']} features, "
          f"{len(features_df)} filas, {features_df['nucleo'].nunique()} familias "
          f"(source: {features_file.name})")
    return True


def _hot_reload_loop():
    """Background thread: chequea cada RELOAD_INTERVAL si el modelo cambió."""
    while True:
        time.sleep(RELOAD_INTERVAL)
        try:
            if not MODEL_FILE.exists():
                continue
            new_mtime = MODEL_FILE.stat().st_mtime
            features_file = _get_features_file()
            new_feat_mtime = features_file.stat().st_mtime if features_file.exists() else 0

            if new_mtime != _model_mtime or new_feat_mtime != _features_mtime:
                print(f"[API] Cambio detectado. Recargando modelo...")
                _load_model()
        except Exception as e:
            print(f"[API] Error en hot-reload: {e}")


@asynccontextmanager
async def lifespan(app):
    _load_model()
    t = threading.Thread(target=_hot_reload_loop, daemon=True)
    t.start()
    print(f"[API] Hot-reload activo cada {RELOAD_INTERVAL}s")
    yield


app = FastAPI(title="Predicción de Compras", version="1.0", lifespan=lifespan)

app.mount("/metrics", make_asgi_app())

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "API activa. Usa /health o /predict/{cedula}"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "features_loaded": FEATURES_DF is not None,
        "total_familias": int(FEATURES_DF["nucleo"].nunique()) if FEATURES_DF is not None else 0,
        "total_series": len(FEATURES_DF) if FEATURES_DF is not None else 0,
        "model_timestamp": META.get("timestamp", "") if META else "",
        "precision@3": META.get("precision@3", 0) if META else 0,
        "hit_rate@3": META.get("hit_rate@3", 0) if META else 0,
        "reload_interval_s": RELOAD_INTERVAL,
    }


@app.get("/predict/{cedula}")
def predict(cedula: int, top_n: int = 3):
    """
    Dado un núcleo familiar (cédula), retorna las top_n subcategorías
    con mayor probabilidad de compra en los próximos 21 días.
    """
    t_start = time.time()
    if MODEL is None or FEATURES_DF is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    df_family = FEATURES_DF[FEATURES_DF["nucleo"] == cedula]
    if df_family.empty:
        raise HTTPException(status_code=404, detail=f"Cédula {cedula} no encontrada")

    feat_cols_base = [c for c in FEATURE_COLUMNS if c in df_family.columns]
    X, _ = _build_features(df_family, feat_cols_base)

    with torch.no_grad():
        probas = torch.sigmoid(MODEL(torch.tensor(X))).numpy()

    df_result = df_family[["nucleo", "COD_SUBCATEGORIA"]].copy()
    df_result["proba_compra"] = probas
    df_result = df_result.sort_values("proba_compra", ascending=False).head(top_n)

    preds = [
        {
            "subcategoria": int(row["COD_SUBCATEGORIA"]),
            "proba_compra": round(float(row["proba_compra"]), 6),
        }
        for _, row in df_result.iterrows()
    ]

    latencia = time.time() - t_start
    REQUEST_COUNT.labels(status="200").inc()
    REQUEST_LATENCY.observe(latencia)

    # Log a Kafka (simple, no bloquea si falla)
    kafka = _get_kafka()
    if kafka:
        try:
            import json
            from datetime import datetime
            msg = json.dumps({
                "timestamp": datetime.now().isoformat(),
                "cedula": cedula,
                "subcategorias": [p["subcategoria"] for p in preds],
                "latencia_ms": round(latencia * 1000, 1)
            })
            kafka.send("predictions.log", msg.encode('utf-8'))
            kafka.flush()
            print(f"[Kafka] Mensaje enviado: cedula={cedula}")
        except Exception as e:
            print(f"[Kafka] Error al enviar: {e}")

    return {
        "cedula": cedula,
        "total_series": len(df_family),
        "predictions": preds,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
