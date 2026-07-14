"""Entrenamiento final LightGBM con hiperparámetros congelados y test W4."""
from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import math
import os
import shutil
import socket
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import MODELS_DIR
from hptuning import matrix, resolve_features, save_json, seed_all

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_FINAL_DIR = ROOT_DIR / "docs" / "final"
LOCK_FILE = ROOT_DIR / "data" / "locks" / "lgbm.lock"


def acquire_lock(label: str):
    """Impide que tuning y entrenamiento LightGBM se solapen."""
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = open(LOCK_FILE, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "proceso desconocido"
        handle.close()
        raise RuntimeError(f"Ya existe una ejecución LightGBM activa: {owner}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "label": label,
        "started_at": datetime.now().isoformat(),
    }, ensure_ascii=False))
    handle.flush()

    def release() -> None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        except Exception:
            pass

    atexit.register(release)
    return handle


def load_hpt(hpt_dir: Path) -> tuple[dict, dict, float]:
    params_path = hpt_dir / "best_hparams.json"
    metrics_path = hpt_dir / "best_metrics.json"
    if not params_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Faltan best_hparams.json o best_metrics.json en {hpt_dir}")
    params = json.load(open(params_path, encoding="utf-8"))
    summary = json.load(open(metrics_path, encoding="utf-8"))
    threshold = float(summary.get("best_metrics_validation", {}).get("threshold_f1", 0.5))
    return params, summary, threshold


def classification_metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int8)
    base_rate = float(y.mean())
    n_top = max(1, int(math.ceil(len(probabilities) * 0.10)))
    top_idx = np.argpartition(probabilities, -n_top)[-n_top:]
    top_rate = float(y[top_idx].mean())
    return {
        "pr_auc": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "threshold_frozen_from_validation": float(threshold),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "lift_at_10": top_rate / base_rate if base_rate else 0.0,
        "base_rate": base_rate,
        "top10_positive_rate": top_rate,
    }


def topk_metrics(df: pd.DataFrame, probabilities: np.ndarray, k: int = 3) -> dict:
    required = {"nucleo", "COD_SUBCATEGORIA", "target"}
    if not required.issubset(df.columns):
        return {
            "precision_at_3": 0.0,
            "recall_at_3": 0.0,
            "hit_rate_at_3": 0.0,
            "n_groups_evaluated": 0,
            "note": f"Columnas faltantes: {sorted(required - set(df.columns))}",
        }
    work = df[["nucleo", "COD_SUBCATEGORIA", "target"]].copy()
    work["proba"] = probabilities
    precisions, recalls, hits = [], [], []
    for _, group in work.groupby("nucleo", sort=False):
        if len(group) < k:
            continue
        positives = set(group.loc[group.target == 1, "COD_SUBCATEGORIA"].tolist())
        if not positives:
            continue
        predicted = set(group.nlargest(k, "proba")["COD_SUBCATEGORIA"].tolist())
        n_hits = len(predicted & positives)
        precisions.append(n_hits / k)
        recalls.append(n_hits / len(positives))
        hits.append(float(n_hits > 0))
    if not precisions:
        return {
            "precision_at_3": 0.0,
            "recall_at_3": 0.0,
            "hit_rate_at_3": 0.0,
            "n_groups_evaluated": 0,
        }
    return {
        "precision_at_3": float(np.mean(precisions)),
        "recall_at_3": float(np.mean(recalls)),
        "hit_rate_at_3": float(np.mean(hits)),
        "n_groups_evaluated": len(precisions),
    }


def plot_outputs(
    y: np.ndarray,
    probabilities: np.ndarray,
    feature_importance: pd.DataFrame,
    plots: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y, probabilities)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label="LightGBM")
    ax.axhline(float(y.mean()), linestyle="--", label="Prevalencia")
    ax.set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall — evaluación W4")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "external_pr_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    top = feature_importance.sort_values("gain", ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.32)))
    ax.barh(top["feature"], top["gain"])
    ax.set(xlabel="Importancia por gain", ylabel="Feature", title="LightGBM — importancia final")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(plots / "feature_importance_gain.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena LightGBM en W0-W3 y evalúa una vez en W4")
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--eval-parquet", required=True)
    parser.add_argument("--feature-set", choices=["gain95", "all"], required=True)
    parser.add_argument("--hpt-dir", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.threads <= 0:
        raise ValueError("--threads debe ser positivo")
    train_path = Path(args.train_parquet)
    eval_path = Path(args.eval_parquet)
    hpt_dir = Path(args.hpt_dir)
    for path in (train_path, eval_path, hpt_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    out = MODELS_DIR / "final" / args.experiment_name
    plots = DOCS_FINAL_DIR / args.experiment_name
    if out.exists() and not args.force:
        raise FileExistsError(f"Ya existe {out}. Use otro nombre o --force")
    if args.force and out.exists():
        shutil.rmtree(out)
    if args.force and plots.exists():
        shutil.rmtree(plots)
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    acquire_lock(f"final:{args.experiment_name}")

    params, hpt_summary, threshold = load_hpt(hpt_dir)
    if hpt_summary.get("feature_set") not in (None, args.feature_set):
        raise ValueError(
            f"El HPT corresponde a {hpt_summary.get('feature_set')}, no a {args.feature_set}"
        )
    num_boost_round = int(params.pop("num_boost_round"))
    seed = int(args.seed if args.seed is not None else params.get("seed", 42))
    params["num_threads"] = args.threads
    params["seed"] = seed
    params["feature_fraction_seed"] = seed
    params["bagging_seed"] = seed
    params["data_random_seed"] = seed
    seed_all(seed)

    started = time.time()
    train_df = pd.read_parquet(train_path)
    eval_df = pd.read_parquet(eval_path)
    for label, frame in (("train", train_df), ("eval", eval_df)):
        if "target" not in frame or frame["target"].nunique() != 2:
            raise ValueError(f"{label}: target binario requerido")

    base_features, onehot, excluded = resolve_features(train_df, args.feature_set)
    x_train, feature_names = matrix(train_df, base_features, onehot)
    x_eval, eval_feature_names = matrix(eval_df, base_features, onehot)
    if feature_names != eval_feature_names:
        raise RuntimeError("Las columnas de train y eval no coinciden")
    hpt_features = hpt_summary.get("model_feature_columns")
    if hpt_features and feature_names != hpt_features:
        raise RuntimeError("Las features actuales no coinciden con las utilizadas en el tuning")
    y_train = train_df["target"].to_numpy(np.float32)
    y_eval = eval_df["target"].to_numpy(np.float32)

    print(
        f"[LGBM-FINAL] Experimento={args.experiment_name} | feature_set={args.feature_set} "
        f"| train={len(train_df):,} | eval={len(eval_df):,} | features={len(feature_names)}"
    )
    print(
        f"[LGBM-FINAL] Hiperparámetros congelados | iteraciones={num_boost_round} "
        f"| threshold={threshold:.6f} | threads={args.threads}"
    )

    train_set = lgb.Dataset(x_train, label=y_train, feature_name=feature_names, free_raw_data=False)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        callbacks=[lgb.log_evaluation(max(1, num_boost_round // 20))],
    )
    probabilities = booster.predict(x_eval, num_iteration=num_boost_round)
    eval_metrics = classification_metrics(y_eval, probabilities, threshold)
    ranking = topk_metrics(eval_df, probabilities, 3)

    importance = pd.DataFrame({
        "feature": feature_names,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    total_gain = float(importance["gain"].sum())
    importance["gain_pct"] = importance["gain"] / total_gain if total_gain else 0.0
    importance.to_csv(out / "feature_importance.csv", index=False)
    booster.save_model(str(out / "model.txt"), num_iteration=num_boost_round)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": args.experiment_name,
        "model": "LightGBM",
        "feature_set": args.feature_set,
        "training_protocol": "fit_all_W0_W3_then_single_external_W4_evaluation",
        "train_parquet": str(train_path),
        "eval_parquet": str(eval_path),
        "hpt_dir": str(hpt_dir),
        "hyperparameters": {**params, "num_boost_round": num_boost_round},
        "seed": seed,
        "feature_columns": feature_names,
        "base_feature_columns": base_features,
        "excluded_columns": excluded,
        "cycle_onehot": onehot,
        "n_train": len(train_df),
        "n_eval": len(eval_df),
        "train_positive_rate": float(y_train.mean()),
        "eval_positive_rate": float(y_eval.mean()),
        "threshold_source": "best_validation_trial_threshold_f1",
        "frozen_threshold": threshold,
        "external_eval_metrics": eval_metrics,
        "top3_metrics": ranking,
        "duration_seconds": round(time.time() - started, 2),
        "threads": args.threads,
        "hpt_validation_summary": {
            "best_trial_number": hpt_summary.get("best_trial_number"),
            "best_metrics_validation": hpt_summary.get("best_metrics_validation"),
        },
    }
    save_json(out / "run_summary.json", summary)
    save_json(plots / "run_summary.json", summary)
    save_json(plots / "eval_metrics.json", {**eval_metrics, **ranking})
    importance.to_csv(plots / "feature_importance.csv", index=False)
    plot_outputs(y_eval, probabilities, importance, plots)

    if args.save_predictions:
        pred_df = eval_df.copy()
        pred_df["proba_compra"] = probabilities
        pred_df.to_parquet(out / "eval_predictions.parquet", index=False)

    print("\n[LGBM-FINAL] ================================================================")
    print(
        f"[LGBM-FINAL] PR-AUC={eval_metrics['pr_auc']:.6f} "
        f"| ROC-AUC={eval_metrics['roc_auc']:.6f} "
        f"| F1={eval_metrics['f1']:.6f}"
    )
    print(
        f"[LGBM-FINAL] Precision={eval_metrics['precision']:.6f} "
        f"| Recall={eval_metrics['recall']:.6f} "
        f"| Lift@10={eval_metrics['lift_at_10']:.4f}"
    )
    print(
        f"[LGBM-FINAL] Precision@3={ranking['precision_at_3']:.6f} "
        f"| Recall@3={ranking['recall_at_3']:.6f} "
        f"| HitRate@3={ranking['hit_rate_at_3']:.6f}"
    )
    print(f"[LGBM-FINAL] Artefactos: {out}")
    print(f"[LGBM-FINAL] Resumen rastreable: {plots / 'run_summary.json'}")
    print("[LGBM-FINAL] ================================================================")


if __name__ == "__main__":
    main()
