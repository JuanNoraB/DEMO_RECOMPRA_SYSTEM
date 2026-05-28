"""
hptuning_lgbm.py — Búsqueda de hiperparámetros con LightGBM + Optuna.
Usa los mismos parquets que hptuning.py:
  - features_hpt_train.parquet  → entrena el modelo
  - features_train.parquet      → evalúa Precision@3
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEST_HPARAMS_FILE,
    FEATURE_COLUMNS,
    FEATURES_DIR,
    FEATURES_HPT_TRAIN_FILE,
    FEATURES_TRAIN_FILE,
    TIPO_CICLO_CATEGORIES,
    TIPO_CICLO_COL,
)

BEST_LGBM_HPARAMS_FILE = FEATURES_DIR.parent / "models" / "best_hparams_lgbm.json"


def _build_X(df: pd.DataFrame) -> np.ndarray:
    feat_cols_base = [c for c in FEATURE_COLUMNS if c in df.columns]
    X_base = df[feat_cols_base].copy()
    if TIPO_CICLO_COL in df.columns:
        dummies = pd.get_dummies(
            pd.Categorical(df[TIPO_CICLO_COL], categories=TIPO_CICLO_CATEGORIES),
            prefix="tipo",
            drop_first=True,
        )
        dummies.index = X_base.index
        X_base = pd.concat([X_base, dummies], axis=1)
    return np.nan_to_num(X_base.values.astype(np.float32), nan=0.0)


def evaluate_topk(df_eval: pd.DataFrame, probas: np.ndarray, top_k: int = 3) -> dict:
    df_eval = df_eval.copy()
    df_eval["proba"] = probas
    precisions, recalls, hit_rates = [], [], []
    for _, grp in df_eval.groupby("nucleo"):
        if len(grp) < top_k:
            continue
        top_items = grp.nlargest(top_k, "proba")["COD_SUBCATEGORIA"].values
        reales = grp.loc[grp["target"] == 1, "COD_SUBCATEGORIA"].values
        if len(reales) == 0:
            continue
        n_hit = len(set(top_items) & set(reales))
        precisions.append(n_hit / top_k)
        recalls.append(n_hit / len(reales))
        hit_rates.append(1.0 if n_hit > 0 else 0.0)
    n = len(precisions)
    if n == 0:
        return {"precision@k": 0.0, "recall@k": 0.0, "hit_rate@k": 0.0, "n_families": 0}
    return {
        "precision@k": float(np.mean(precisions)),
        "recall@k": float(np.mean(recalls)),
        "hit_rate@k": float(np.mean(hit_rates)),
        "n_families": n,
    }


def objective(trial: optuna.Trial, df_train: pd.DataFrame, df_eval: pd.DataFrame) -> float:
    n_estimators = trial.suggest_int("n_estimators", 100, 800)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
    }

    X_tr = _build_X(df_train)
    y_tr = df_train["target"].values.astype(np.float32)

    n_pos = y_tr.sum()
    params["scale_pos_weight"] = (len(y_tr) - n_pos) / max(n_pos, 1)

    ds_train = lgb.Dataset(X_tr, label=y_tr)
    booster = lgb.train(params, ds_train, num_boost_round=n_estimators, callbacks=[lgb.log_evaluation(-1)])

    X_ev = _build_X(df_eval)
    probas = booster.predict(X_ev)

    metrics = evaluate_topk(df_eval, probas, top_k=3)
    return metrics["precision@k"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="HPT LightGBM — Búsqueda de hiperparámetros")
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    if not FEATURES_HPT_TRAIN_FILE.exists() or not FEATURES_TRAIN_FILE.exists():
        print("[ERROR] Faltan parquets. Ejecuta primero:")
        print("  python hptuning.py --prepare --workers 16")
        return

    df_train = pd.read_parquet(FEATURES_HPT_TRAIN_FILE)
    df_eval = pd.read_parquet(FEATURES_TRAIN_FILE)

    print(f"\n[LGBM-HPT] Verificación de parquets:")
    print(f"  features_hpt_train  filas={len(df_train):,} | familias={df_train['nucleo'].nunique()} | positivos={int(df_train['target'].sum())} ({df_train['target'].mean()*100:.1f}%)")
    print(f"  features_train      filas={len(df_eval):,}  | familias={df_eval['nucleo'].nunique()}  | positivos={int(df_eval['target'].sum())} ({df_eval['target'].mean()*100:.1f}%)")
    print(f"\n[LGBM-HPT] Iniciando: {args.trials} trials")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, df_train, df_eval),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n[LGBM-HPT] Mejor trial #{best.number} | Precision@3={best.value:.4f} ({best.value*100:.1f}%)")
    print(f"  Params: {best.params}")

    BEST_LGBM_HPARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_LGBM_HPARAMS_FILE, "w") as f:
        json.dump({"precision@3": best.value, **best.params}, f, indent=2)
    print(f"[LGBM-HPT] Guardado: {BEST_LGBM_HPARAMS_FILE}")

    print(f"\n[LGBM-HPT] Top 5 trials:")
    top5 = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:5]
    for t in top5:
        print(f"  #{t.number:3d} | Precision@3={t.value:.4f} | {t.params}")


if __name__ == "__main__":
    main()
