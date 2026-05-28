"""
hptuning.py — Búsqueda automática de hiperparámetros con Optuna.

Maximiza Precision@3 evaluada sobre features_eval.parquet (ventana temporal distinta al training).

Flujo:
  1. Preparar features y correr búsqueda (todo junto):
       python hptuning.py --prepare --trials 100 --epochs 30

  2. Correr búsqueda sobre features ya generadas:
       python hptuning.py --trials 100 --epochs 30

  El path al CSV se lee de config.py (HISTORICO_FILE / env HISTORICO_PATH).
  El mejor config queda en BEST_HPARAMS_FILE (config.py).

Espacio de búsqueda:
  - lr            : 1e-4 → 1e-2 (log)
  - dropout       : 0.1 → 0.5
  - activation    : relu | tanh | leaky_relu
  - n_layers      : 2 → 4
  - layer_i       : 32 | 64 | 128 | 256
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEST_HPARAMS_FILE,
    FEATURES_HPT_TRAIN_FILE,
    FEATURES_TRAIN_FILE,
    HISTORICO_FILE,
)
from train_fnn import run_training


def objective(trial: optuna.Trial, epochs: int) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "leaky_relu"])
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_dims = [
        trial.suggest_categorical(f"layer_{i}", [32, 64, 128, 256])
        for i in range(n_layers)
    ]

    _, meta = run_training(
        features_path=FEATURES_HPT_TRAIN_FILE,
        eval_features_path=FEATURES_TRAIN_FILE,
        epochs=epochs,
        lr=lr,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        save=False,
    )
    return meta["precision@3"]


def main():
    parser = argparse.ArgumentParser(description="HPT — Búsqueda de hiperparámetros con Optuna")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30,
                        help="Epochs por trial (menor = más rápido pero menos preciso)")
    parser.add_argument("--prepare", action="store_true",
                        help="Regenerar features HPT antes de optimizar")
    parser.add_argument("--historico", type=str, default=str(HISTORICO_FILE),
                        help="Path al CSV (default: HISTORICO_FILE en config.py)")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.prepare:
        from feature_engineering import run_pipeline
        from simulation import prepare_hpt_split
        # Solo genera el CSV truncado — features_train.parquet (CSV completo) ya existe
        train_csv, _ = prepare_hpt_split(Path(args.historico), trim_days=21)

        print("[HPT] Calculando features_hpt_train (CSV - 21d, entrena el modelo)...")
        run_pipeline(historico_path=train_csv, filtro_path=None,
                     prediction_window=21, n_workers=args.workers,
                     output_path=FEATURES_HPT_TRAIN_FILE)

    if not FEATURES_HPT_TRAIN_FILE.exists() or not FEATURES_TRAIN_FILE.exists():
        print("[ERROR] Faltan features. Ejecuta con --prepare:")
        print("  python hptuning.py --prepare --trials 100")
        return

    import pandas as pd
    df_tr = pd.read_parquet(FEATURES_HPT_TRAIN_FILE, columns=["nucleo", "COD_SUBCATEGORIA", "target"])
    df_ev = pd.read_parquet(FEATURES_TRAIN_FILE, columns=["nucleo", "COD_SUBCATEGORIA", "target"])
    print(f"\n[HPT] Verificación de parquets:")
    print(f"  features_hpt_train  filas={len(df_tr):,} | familias={df_tr['nucleo'].nunique()} | positivos={int(df_tr['target'].sum())} ({df_tr['target'].mean()*100:.1f}%)")
    print(f"  features_train      filas={len(df_ev):,} | familias={df_ev['nucleo'].nunique()} | positivos={int(df_ev['target'].sum())} ({df_ev['target'].mean()*100:.1f}%)")
    print(f"  → train tiene MENOS filas/positivos que eval (ventana temporal anterior) — si es igual algo está mal")
    print()
    print(f"[HPT] Iniciando: {args.trials} trials | {args.epochs} epochs/trial")
    print(f"[HPT] Train (truncado): {FEATURES_HPT_TRAIN_FILE}")
    print(f"[HPT] Eval  (completo): {FEATURES_TRAIN_FILE}")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, args.epochs),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    n_layers = best.params["n_layers"]
    hidden_dims = [best.params[f"layer_{i}"] for i in range(n_layers)]

    best_hparams = {
        "lr": best.params["lr"],
        "dropout": best.params["dropout"],
        "activation": best.params["activation"],
        "hidden_dims": hidden_dims,
        "precision@3": best.value,
    }

    BEST_HPARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_HPARAMS_FILE, "w") as f:
        json.dump(best_hparams, f, indent=2)

    print(f"\n[HPT] ══════════════════════════════════")
    print(f"[HPT] Mejor trial #{best.number}")
    print(f"[HPT] Precision@3: {best.value:.4f} ({best.value*100:.1f}%)")
    print(f"[HPT] Parámetros:")
    for k, v in best_hparams.items():
        print(f"       {k}: {v}")
    print(f"[HPT] Guardado: {BEST_HPARAMS_FILE}")

    top5 = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True,
    )[:5]
    print(f"\n[HPT] Top 5 trials:")
    for t in top5:
        n = t.params["n_layers"]
        hdims = [t.params[f"layer_{i}"] for i in range(n)]
        print(f"  #{t.number:3d} | Precision@3={t.value:.4f} | "
              f"lr={t.params['lr']:.5f} | dropout={t.params['dropout']:.2f} | "
              f"act={t.params['activation']} | layers={hdims}")


if __name__ == "__main__":
    main()
