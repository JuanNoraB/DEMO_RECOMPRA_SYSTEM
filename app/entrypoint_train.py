"""
entrypoint_train.py — Punto de entrada del container/CronJob de entrenamiento.

Modos:
  1. Manual (fuerza entrenamiento):
     python entrypoint_train.py --historico data/raw/historico_1000_base.csv --force

  2. CronJob (chequea si hay nuevas transacciones >= threshold):
     python entrypoint_train.py --historico data/raw/historico_1000_base.csv

  En modo CronJob:
    - Lee tx_counter.json → si total_new < threshold, sale sin hacer nada
    - Lee nuevas_series.csv → usa como --filtro (solo recalcula series modificadas)
    - Después de entrenar, resetea el contador y borra nuevas_series.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config import DATA_DIR, NEW_TX_THRESHOLD, PREDICTION_WINDOW_DAYS


COUNTER_FILE = DATA_DIR / "tx_counter.json"
SERIES_FILE = DATA_DIR / "nuevas_series.csv"


def _load_counter() -> dict:
    if COUNTER_FILE.exists():
        with open(COUNTER_FILE) as f:
            return json.load(f)
    return {"total_new": 0, "last_ingest": None}


def _reset_counter():
    """Resetea el contador y borra las series pendientes."""
    with open(COUNTER_FILE, "w") as f:
        json.dump({"total_new": 0, "last_ingest": None}, f, indent=2)
    if SERIES_FILE.exists():
        SERIES_FILE.unlink()


def main():
    parser = argparse.ArgumentParser(description="Training Pipeline (features + FNN)")
    parser.add_argument("--historico", type=str, required=True,
                        help="Path al CSV de transacciones históricas")
    parser.add_argument("--filtro", type=str, default=None,
                        help="Path a CSV con CODIGO_FAMILIA;COD_SUBCATEGORIA para filtrar series")
    parser.add_argument("--prediction-window", type=int, default=PREDICTION_WINDOW_DAYS)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=int, default=NEW_TX_THRESHOLD,
                        help="Mínimo de nuevas transacciones para disparar entrenamiento")
    parser.add_argument("--force", action="store_true",
                        help="Forzar entrenamiento sin chequear counter")
    args = parser.parse_args()

    # ── Chequeo de nuevas transacciones ──────────────────────────────────
    filtro_path = Path(args.filtro) if args.filtro else None

    if not args.force:
        counter = _load_counter()
        total_new = counter.get("total_new", 0)
        print(f"[Check] Nuevas transacciones acumuladas: {total_new} (threshold: {args.threshold})")

        if total_new < args.threshold:
            print(f"[Check] No hay suficientes transacciones nuevas. Saliendo.")
            return

        print(f"[Check] Threshold alcanzado. Iniciando entrenamiento...")

        # Usar nuevas_series.csv como filtro si existe y no se pasó uno explícito
        if filtro_path is None and SERIES_FILE.exists():
            filtro_path = SERIES_FILE
            print(f"[Check] Usando filtro automático: {SERIES_FILE}")

    t_total = time.time()

    # ── Paso 1: Feature Engineering (train) ──────────────────────────────
    print()
    print("=" * 60)
    print("PASO 1: FEATURE ENGINEERING (TRAIN)")
    print("=" * 60)

    from feature_engineering import run_pipeline

    t0 = time.time()
    result = run_pipeline(
        historico_path=Path(args.historico),
        filtro_path=filtro_path,
        prediction_window=args.prediction_window,
        n_workers=args.workers,
    )
    print(f"Features train: {time.time()-t0:.1f}s")

    if result.empty:
        print("[ERROR] No se generaron features. Abortando.")
        return

    # ── Paso 2: Feature Engineering (inference) ──────────────────────────
    print()
    print("=" * 60)
    print("PASO 2: FEATURE ENGINEERING (INFERENCE)")
    print("=" * 60)

    from config import FEATURES_INFERENCE_FILE

    t0 = time.time()
    result_inf = run_pipeline(
        historico_path=Path(args.historico),
        filtro_path=filtro_path,
        prediction_window=0,
        n_workers=args.workers,
        output_path=FEATURES_INFERENCE_FILE,
    )
    print(f"Features inference: {time.time()-t0:.1f}s")

    # ── Paso 3: Entrenamiento FNN ────────────────────────────────────────
    print()
    print("=" * 60)
    print("PASO 3: ENTRENAMIENTO FNN")
    print("=" * 60)

    from train_fnn import run_training

    t0 = time.time()
    model, meta = run_training(epochs=args.epochs, lr=args.lr)
    print(f"Training: {time.time()-t0:.1f}s")

    # ── Publicar a Kafka ──────────────────────────────────────────────────
    from kafka_helper import publish
    publish("training.completed", {
        "precision@3": meta.get("precision@3", 0),
        "hit_rate@3": meta.get("hit_rate@3", 0),
        "recall@3": meta.get("recall@3", 0),
        "val_loss": meta.get("best_val_loss", 0),
        "epochs": meta.get("epochs", 0),
        "n_families_eval": meta.get("n_families_eval", 0),
        "training_duration_s": round(time.time() - t_total, 1),
        "model_timestamp": meta.get("timestamp", ""),
    })

    # ── Resetear counter si estamos en modo CronJob ──────────────────────
    if not args.force:
        _reset_counter()
        print("[Check] Counter reseteado. Nuevas series limpiadas.")

    print()
    print("=" * 60)
    print(f"PIPELINE COMPLETO: {time.time()-t_total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
