"""Entrenamiento final FNN con hiperparámetros congelados y evaluación externa."""
from __future__ import annotations

import argparse
import atexit
import fcntl
import gc
import json
import math
import os
import shutil
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

from config import MODELS_DIR
from hptuning import device_from, duration, matrix, resolve_features, save_json, seed_all
from train_fnn import FeatureScaler, PurchaseFNN

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_FINAL_DIR = ROOT_DIR / "docs" / "final"
LOCK_FILE = ROOT_DIR / "data" / "locks" / "fnn_gpu.lock"


def acquire_lock(label: str):
    """Impide que dos ejecuciones FNN compartan la GPU accidentalmente."""
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = open(LOCK_FILE, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "proceso desconocido"
        handle.close()
        raise RuntimeError(f"Ya existe una ejecución FNN activa: {owner}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({
        "pid": os.getpid(), "host": socket.gethostname(), "label": label,
        "started_at": datetime.now().isoformat(),
    }, ensure_ascii=False))
    handle.flush()

    def release():
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        except Exception:
            pass

    atexit.register(release)
    return handle


def load_hpt(hpt_dir: Path) -> tuple[dict[str, Any], dict[str, Any], int, float]:
    params_path = hpt_dir / "best_hparams.json"
    metrics_path = hpt_dir / "best_metrics.json"
    if not params_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Faltan best_hparams.json o best_metrics.json en {hpt_dir}")
    params = json.load(open(params_path, encoding="utf-8"))
    summary = json.load(open(metrics_path, encoding="utf-8"))
    seed = int(summary.get("best_seed", summary.get("seed", 42)))
    winner_runs = summary.get("confirmation", {}).get("winner_seed_metrics", [])
    thresholds = [float(r["threshold_f1"]) for r in winner_runs if r.get("threshold_f1") is not None]
    threshold = float(np.median(thresholds)) if thresholds else float(
        summary.get("best_single_seed_metrics_validation", {}).get("threshold_f1", 0.5)
    )
    return params, summary, seed, threshold


def train_full(model, optimizer, criterion, x, y, epochs, batch_size, seed):
    rows = int(x.shape[0])
    history = []
    for epoch in range(1, epochs + 1):
        epoch_seed = seed + epoch
        torch.manual_seed(epoch_seed)
        if x.device.type == "cuda":
            torch.cuda.manual_seed_all(epoch_seed)
            torch.cuda.synchronize()
        started = time.time()
        order = torch.randperm(rows, device=x.device)
        model.train()
        total = 0.0
        updates = 0
        for start in range(0, rows, batch_size):
            idx = order[start:start + batch_size]
            xb, yb = x.index_select(0, idx), y.index_select(0, idx)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().item()) * int(idx.numel())
            updates += 1
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - started
        train_loss = total / rows
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "duration_seconds": elapsed, "optimizer_updates": updates})
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 20) == 0:
            print(f"[FINAL-TRAIN] época {epoch}/{epochs} | loss={train_loss:.6f} | updates={updates} | {duration(elapsed)}", flush=True)
    return history


@torch.inference_mode()
def predict(model, x, batch_size):
    model.eval()
    chunks = []
    eval_batch = max(batch_size, 262_144 if x.device.type == "cuda" else 65_536)
    for start in range(0, int(x.shape[0]), eval_batch):
        chunks.append(torch.sigmoid(model(x[start:start + eval_batch])).float().cpu().numpy())
    return np.concatenate(chunks).ravel()


def classification_metrics(y, p, threshold):
    pred = (p >= threshold).astype(np.int8)
    base = float(y.mean())
    n = max(1, int(math.ceil(len(p) * 0.10)))
    top = np.argpartition(p, -n)[-n:]
    top_rate = float(y[top].mean())
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "threshold_frozen_from_validation": float(threshold),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "lift_at_10": top_rate / base if base else 0.0,
        "base_rate": base,
        "top10_positive_rate": top_rate,
    }


def topk_metrics(df, p, k=3):
    required = {"nucleo", "COD_SUBCATEGORIA", "target"}
    if not required.issubset(df.columns):
        return {"precision_at_3": 0.0, "recall_at_3": 0.0,
                "hit_rate_at_3": 0.0, "n_groups_evaluated": 0,
                "note": f"Columnas faltantes: {sorted(required - set(df.columns))}"}
    work = df[["nucleo", "COD_SUBCATEGORIA", "target"]].copy()
    work["proba"] = p
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
        return {"precision_at_3": 0.0, "recall_at_3": 0.0,
                "hit_rate_at_3": 0.0, "n_groups_evaluated": 0}
    return {"precision_at_3": float(np.mean(precisions)),
            "recall_at_3": float(np.mean(recalls)),
            "hit_rate_at_3": float(np.mean(hits)),
            "n_groups_evaluated": len(precisions)}


def plot_outputs(y, p, history, plots):
    precision, recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label="FNN")
    ax.axhline(float(y.mean()), linestyle="--", label="Prevalencia")
    ax.set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall — evaluación externa")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(plots / "external_pr_curve.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    hist = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist.epoch, hist.train_loss)
    ax.set(xlabel="Época", ylabel="BCE ponderada", title="Entrenamiento final sobre W0-W3 completo")
    ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(plots / "training_loss.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Entrena FNN final sobre W0-W3 y evalúa una vez en W4/test")
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--eval-parquet", required=True)
    p.add_argument("--feature-set", choices=["gain95", "all"], required=True)
    p.add_argument("--hpt-dir", required=True)
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--torch-threads", type=int, default=min(32, os.cpu_count() or 1))
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    device = device_from(args.device)
    torch.set_num_threads(max(1, args.torch_threads))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats()

    train_path, eval_path, hpt_dir = Path(args.train_parquet), Path(args.eval_parquet), Path(args.hpt_dir)
    for path in (train_path, eval_path, hpt_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    out = MODELS_DIR / "final" / args.experiment_name
    plots = DOCS_FINAL_DIR / args.experiment_name
    if out.exists() and not args.force:
        raise FileExistsError(f"Ya existe {out}. Use otro nombre o --force")
    if args.force and out.exists(): shutil.rmtree(out)
    if args.force and plots.exists(): shutil.rmtree(plots)
    out.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)

    acquire_lock(f"final:{args.experiment_name}")
    params, hpt_summary, hpt_seed, threshold = load_hpt(hpt_dir)
    hpt_feature_set = hpt_summary.get("feature_set")
    if hpt_feature_set and hpt_feature_set != args.feature_set:
        raise ValueError(f"El HPT es {hpt_feature_set}, no {args.feature_set}")
    epochs = int(args.epochs or hpt_summary.get("final_epochs", 120))
    seed = int(args.seed if args.seed is not None else hpt_seed)
    seed_all(seed)

    print(f"[FINAL] Experimento={args.experiment_name} | feature_set={args.feature_set} | device={device}")
    print(f"[FINAL] Hiperparámetros congelados | epochs={epochs} | seed={seed} | threshold={threshold:.6f}")
    started = time.time()
    train_df, eval_df = pd.read_parquet(train_path), pd.read_parquet(eval_path)
    for label, frame in (("train", train_df), ("eval", eval_df)):
        if "target" not in frame or frame.target.nunique() != 2:
            raise ValueError(f"{label}: target binario requerido")

    base, onehot, excluded = resolve_features(train_df, args.feature_set)
    xtr0, names = matrix(train_df, base, onehot)
    xev0, eval_names = matrix(eval_df, base, onehot)
    if names != eval_names:
        raise RuntimeError("Las columnas de train y eval no coinciden")
    scaler = FeatureScaler()
    xtr = scaler.fit_transform(xtr0, names).astype(np.float32, copy=False)
    xev = scaler.transform(xev0, names).astype(np.float32, copy=False)
    ytr, yev = train_df.target.to_numpy(np.float32), eval_df.target.to_numpy(np.float32)
    del xtr0, xev0; gc.collect()
    xtr_t = torch.from_numpy(np.ascontiguousarray(xtr)).to(device)
    ytr_t = torch.from_numpy(np.ascontiguousarray(ytr)).to(device)
    xev_t = torch.from_numpy(np.ascontiguousarray(xev)).to(device)

    print(f"[FINAL] Train={len(train_df):,} | Eval={len(eval_df):,} | features={len(names)} | batch={int(params['batch_size']):,}")
    if device.type == "cuda":
        print(f"[FINAL] GPU={torch.cuda.get_device_name(0)} | VRAM inicial={torch.cuda.memory_allocated()/1024**2:,.1f} MiB")

    model = PurchaseFNN(len(names), params["hidden_dims"], float(params["dropout"]), params["activation"]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(params["pos_weight"])], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
    history = train_full(model, optimizer, criterion, xtr_t, ytr_t, epochs, int(params["batch_size"]), seed)
    proba = predict(model, xev_t, int(params["batch_size"]))
    eval_metrics = classification_metrics(yev, proba, threshold)
    ranking = topk_metrics(eval_df, proba, 3)

    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "model_state.pth")
    save_json(out / "scaler.json", {"means": scaler.means, "stds": scaler.stds})
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    summary = {
        "timestamp": datetime.now().isoformat(), "experiment_name": args.experiment_name,
        "model": "FNN", "feature_set": args.feature_set,
        "training_protocol": "fit_all_W0_W3_then_single_external_W4_evaluation",
        "train_parquet": str(train_path), "eval_parquet": str(eval_path), "hpt_dir": str(hpt_dir),
        "hyperparameters": params, "epochs": epochs, "seed": seed,
        "feature_columns": names, "base_feature_columns": base, "excluded_columns": excluded,
        "cycle_onehot": onehot, "n_train": len(train_df), "n_eval": len(eval_df),
        "train_positive_rate": float(ytr.mean()), "eval_positive_rate": float(yev.mean()),
        "threshold_source": "median_threshold_f1_of_winner_validation_seeds",
        "frozen_threshold": threshold, "external_eval_metrics": eval_metrics,
        "top3_metrics": ranking, "duration_seconds": round(time.time() - started, 2),
        "device": str(device), "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "max_gpu_memory_mib": round(torch.cuda.max_memory_allocated()/1024**2, 2) if device.type == "cuda" else None,
        "hpt_validation_summary": {"best_candidate_id": hpt_summary.get("best_candidate_id"),
                                   "confirmation": hpt_summary.get("confirmation")},
    }
    save_json(out / "run_summary.json", summary)
    save_json(plots / "run_summary.json", summary)
    save_json(plots / "eval_metrics.json", {**eval_metrics, **ranking})
    plot_outputs(yev, proba, history, plots)
    if args.save_predictions:
        pred_df = eval_df.copy(); pred_df["proba_compra"] = proba
        pred_df.to_parquet(out / "eval_predictions.parquet", index=False)

    print("\n[FINAL] ================================================================")
    print(f"[FINAL] PR-AUC={eval_metrics['pr_auc']:.6f} | ROC-AUC={eval_metrics['roc_auc']:.6f} | F1={eval_metrics['f1']:.6f}")
    print(f"[FINAL] Precision={eval_metrics['precision']:.6f} | Recall={eval_metrics['recall']:.6f} | Lift@10={eval_metrics['lift_at_10']:.4f}")
    print(f"[FINAL] Precision@3={ranking['precision_at_3']:.6f} | Recall@3={ranking['recall_at_3']:.6f} | HitRate@3={ranking['hit_rate_at_3']:.6f}")
    print(f"[FINAL] Artefactos: {out}")
    print(f"[FINAL] Resumen rastreable: {plots / 'run_summary.json'}")
    print("[FINAL] ================================================================")


if __name__ == "__main__":
    main()
