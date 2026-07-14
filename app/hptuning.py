"""Successive Halving para FNN de recompra binaria.

Por defecto: 45 candidatos -> 3 épocas -> 15; luego 9 -> 5; luego 25 -> 1.
Mantiene train/validación en GPU, muestra progreso por época y candidato,
guarda checkpoints por candidato y permite reanudar el mismo experimento.
W4/test no participa en el tuning.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
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
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config import FNN_GAIN95_FEATURES, MODELS_DIR, TIPO_CICLO_CATEGORIES, TIPO_CICLO_COL
from train_fnn import FeatureScaler, PurchaseFNN

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_HPT_DIR = ROOT_DIR / "docs" / "hpt"
GLOBAL_LOG_FILE = ROOT_DIR / "data" / "logs" / "fnn_hpt_runs.jsonl"
ID_COLUMNS = {"target", "nucleo", "COD_SUBCATEGORIA", "CODIGO_FAMILIA", "Ciclos_CODIGO_FAMILIA"}
GPU_BATCHES = [16384, 32768, 65536, 131072, 262144]
CPU_BATCHES = [4096, 8192, 16384, 32768, 65536]


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def device_from(value: str) -> torch.device:
    if value == "auto": value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero torch.cuda.is_available() es False")
    return torch.device(value)


def int_list(value: str) -> list[int]:
    try: result = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc: raise argparse.ArgumentTypeError("Use enteros separados por comas") from exc
    if not result or any(x <= 0 for x in result): raise argparse.ArgumentTypeError("Los valores deben ser positivos")
    return result


def duration(seconds: float) -> str:
    seconds = int(max(0, round(seconds))); h, r = divmod(seconds, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def save_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f: json.dump(value, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def resolve_features(df: pd.DataFrame, mode: str) -> tuple[list[str], bool, list[str]]:
    if mode == "gain95":
        missing = [c for c in FNN_GAIN95_FEATURES if c not in df.columns]
        if missing: raise ValueError(f"Faltan features GAIN95: {missing}")
        return list(FNN_GAIN95_FEATURES), False, []
    selected, excluded = [], []
    for c in df.columns:
        if c in ID_COLUMNS or c == TIPO_CICLO_COL or not is_numeric_dtype(df[c]) or df[c].nunique(dropna=False) <= 1:
            excluded.append(c)
        else: selected.append(c)
    if not selected: raise ValueError("No se encontraron features numéricas válidas")
    return selected, True, excluded


def matrix(df: pd.DataFrame, columns: list[str], onehot: bool) -> tuple[np.ndarray, list[str]]:
    x = df[columns].copy()
    if onehot and TIPO_CICLO_COL in df.columns:
        d = pd.get_dummies(pd.Categorical(df[TIPO_CICLO_COL], categories=TIPO_CICLO_CATEGORIES),
                           prefix="tipo", drop_first=True, dtype=np.float32)
        d.index = x.index; x = pd.concat([x, d], axis=1)
    names = list(x.columns)
    return np.nan_to_num(x.to_numpy(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0), names


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pr = float(average_precision_score(y, p)); roc = float(roc_auc_score(y, p))
    pc, rc, th = precision_recall_curve(y, p)
    if len(th):
        den = pc[:-1] + rc[:-1]
        fs = np.divide(2 * pc[:-1] * rc[:-1], den, out=np.zeros_like(den), where=den > 0)
        threshold = float(th[int(np.argmax(fs))])
    else: threshold = 0.5
    pred = (p >= threshold).astype(np.int8)
    base = float(y.mean()); n = max(1, int(math.ceil(len(p) * .10)))
    top = np.argpartition(p, -n)[-n:]; top_rate = float(y[top].mean())
    return {"pr_auc": pr, "roc_auc": roc, "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "lift_at_10": top_rate / base if base else 0.0, "threshold_f1": threshold,
            "base_rate": base, "top10_positive_rate": top_rate}


def spread(count: int, low: float, high: float, rng: np.random.Generator, log: bool = False) -> np.ndarray:
    q = (np.arange(count) + rng.random(count)) / count; rng.shuffle(q)
    if log: return 10 ** (np.log10(low) + q * (np.log10(high) - np.log10(low)))
    return low + q * (high - low)


def balanced(values: list[Any], count: int, rng: np.random.Generator) -> list[Any]:
    result = [values[i % len(values)] for i in range(count)]; rng.shuffle(result); return result


def candidates(count: int, seed: int, device: torch.device) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed); batches = GPU_BATCHES if device.type == "cuda" else CPU_BATCHES
    lr = spread(count, 1e-4, 1e-2, rng, True); drop = spread(count, .05, .50, rng)
    wd = spread(count, 1e-7, 1e-3, rng, True); acts = balanced(["relu", "tanh", "leaky_relu"], count, rng)
    layers = balanced([2, 3, 4], count, rng); bs = balanced(batches, count, rng); pw = balanced([1., 2., 4.], count, rng)
    widths = [balanced([32, 64, 128, 256], count, rng) for _ in range(4)]
    result = []
    for i in range(count):
        result.append({"candidate_id": i, "lr": float(lr[i]), "dropout": float(drop[i]),
                       "activation": acts[i], "hidden_dims": [int(widths[j][i]) for j in range(int(layers[i]))],
                       "batch_size": int(bs[i]), "weight_decay": float(wd[i]), "pos_weight": float(pw[i])})
    return result


def cpu_state(value: Any) -> Any:
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value, dict): return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_state(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_state(v) for v in value)
    return value


def checkpoint(path: Path, cfg: dict, model: nn.Module, optimizer, epochs: int, history: list, round_metrics: dict) -> None:
    payload = {"config": cfg, "model_state": cpu_state(model.state_dict()), "optimizer_state": cpu_state(optimizer.state_dict()),
               "epochs_trained": epochs, "history": history, "round_metrics": round_metrics}
    tmp = path.with_suffix(".tmp"); torch.save(payload, tmp); tmp.replace(path)


def load_candidate(cfg: dict, input_dim: int, path: Path, device: torch.device, seed: int):
    seed_all(seed + cfg["candidate_id"] * 1009)
    model = PurchaseFNN(input_dim, cfg["hidden_dims"], cfg["dropout"], cfg["activation"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    epochs, history, round_metrics = 0, [], {}
    if path.exists():
        data = torch.load(path, map_location=device, weights_only=False)
        if data["config"] != cfg: raise RuntimeError(f"Checkpoint incompatible: {path}")
        model.load_state_dict(data["model_state"]); optimizer.load_state_dict(data["optimizer_state"])
        epochs, history, round_metrics = data["epochs_trained"], data.get("history", []), data.get("round_metrics", {})
    return model, optimizer, epochs, history, round_metrics


def train_to(model, optimizer, criterion, x, y, start_epoch: int, target_epoch: int,
             batch: int, seed: int, cid: int, round_no: int) -> list[dict]:
    result, rows = [], int(x.shape[0])
    for epoch in range(start_epoch + 1, target_epoch + 1):
        t0 = time.time(); epoch_seed = seed + cid * 100003 + epoch; torch.manual_seed(epoch_seed)
        if x.device.type == "cuda": torch.cuda.manual_seed_all(epoch_seed)
        order = torch.randperm(rows, device=x.device); model.train(); total = 0.0
        for start in range(0, rows, batch):
            idx = order[start:start + batch]; xb = x.index_select(0, idx); yb = y.index_select(0, idx)
            optimizer.zero_grad(set_to_none=True); loss = criterion(model(xb), yb); loss.backward(); optimizer.step()
            total += float(loss.detach().item()) * int(idx.numel())
        elapsed = time.time() - t0; train_loss = total / rows
        result.append({"round": round_no, "epoch": epoch, "train_loss": train_loss, "duration_seconds": elapsed})
        print(f"\r[HALVING][R{round_no}][C{cid:03d}] época {epoch}/{target_epoch} | loss={train_loss:.6f} | {duration(elapsed)}",
              end="", flush=True)
    if target_epoch > start_epoch: print(flush=True)
    return result


@torch.inference_mode()
def evaluate(model, criterion, x, y, y_cpu: np.ndarray, batch: int):
    model.eval(); parts, total, rows = [], 0.0, 0; eval_batch = max(batch, 262144 if x.device.type == "cuda" else 65536)
    for start in range(0, int(x.shape[0]), eval_batch):
        xb, yb = x[start:start + eval_batch], y[start:start + eval_batch]; logits = model(xb); loss = criterion(logits, yb)
        total += float(loss.item()) * len(xb); rows += len(xb); parts.append(torch.sigmoid(logits).float().cpu().numpy())
    p = np.concatenate(parts).ravel(); result = metrics(y_cpu, p); result["val_loss"] = total / rows
    return result, p


def plot_progress(records: list[dict], path: Path) -> None:
    if not records: return
    df = pd.DataFrame(records); fig, ax = plt.subplots(figsize=(10, 6))
    for _, group in df.groupby("candidate_id"):
        group = group.sort_values("target_epoch"); ax.plot(group["target_epoch"], group["pr_auc"], marker="o", alpha=.45)
    ax.set(xlabel="Épocas acumuladas", ylabel="PR-AUC validación", title="Successive Halving — evolución de candidatos")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def validate_plan(count: int, epochs: list[int], survivors: list[int]) -> None:
    if len(epochs) != len(survivors): raise ValueError("--round-epochs y --survivors deben tener igual longitud")
    if any(b <= a for a, b in zip(epochs, epochs[1:])): raise ValueError("Las épocas deben crecer estrictamente")
    active = count
    for keep in survivors:
        if keep > active: raise ValueError("Un valor de supervivientes supera el grupo activo")
        active = keep
    if survivors[-1] != 1: raise ValueError("El último valor de --survivors debe ser 1")


def main() -> None:
    parser = argparse.ArgumentParser(description="HPT FNN con Successive Halving y tensores residentes en GPU")
    parser.add_argument("--train-parquet", required=True); parser.add_argument("--feature-set", choices=["gain95", "all"], required=True)
    parser.add_argument("--experiment-name", default=None); parser.add_argument("--candidates", "--trials", dest="candidates", type=int, default=45)
    parser.add_argument("--round-epochs", type=int_list, default=[3, 9, 25]); parser.add_argument("--survivors", type=int_list, default=[15, 5, 1])
    parser.add_argument("--val-size", type=float, default=.20); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=min(32, os.cpu_count() or 1)); parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-rows", type=int, default=None); parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS); parser.add_argument("--patience", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--loader-workers", type=int, default=0, help=argparse.SUPPRESS); args = parser.parse_args()
    if not 0 < args.val_size < 1 or args.candidates <= 0: raise ValueError("Parámetros inválidos")
    if args.epochs is not None: args.round_epochs[-1] = args.epochs
    validate_plan(args.candidates, args.round_epochs, args.survivors)
    seed_all(args.seed); torch.set_num_threads(max(1, args.torch_threads)); device = device_from(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.set_float32_matmul_precision("high")
    train_path = Path(args.train_parquet)
    if not train_path.exists(): raise FileNotFoundError(train_path)
    name = args.experiment_name or f"fnn_{args.feature_set}_halving"; out = MODELS_DIR / "hpt" / name; plots = DOCS_HPT_DIR / name; ckpts = out / "checkpoints"
    if args.force and out.exists(): shutil.rmtree(out)
    if args.force and plots.exists(): shutil.rmtree(plots)
    out.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True); ckpts.mkdir(parents=True, exist_ok=True)
    print(f"[HALVING] Experimento={name} | device={device} | plan={args.candidates}->{args.survivors} | épocas={args.round_epochs}")
    if device.type == "cuda": print(f"[HALVING] GPU: {torch.cuda.get_device_name(0)}")
    started = time.time(); df = pd.read_parquet(train_path)
    if "target" not in df or df["target"].nunique() != 2: raise ValueError("target binario requerido")
    if args.max_rows and len(df) > args.max_rows:
        _, df = train_test_split(df, test_size=args.max_rows, random_state=args.seed, stratify=df["target"]); df = df.reset_index(drop=True)
    base, onehot, excluded = resolve_features(df, args.feature_set); raw, feature_names = matrix(df, base, onehot); y = df["target"].to_numpy(np.float32)
    idx_tr, idx_va = train_test_split(np.arange(len(df)), test_size=args.val_size, random_state=args.seed, stratify=y)
    xtr0, xva0, ytr0, yva0 = np.ascontiguousarray(raw[idx_tr]), np.ascontiguousarray(raw[idx_va]), np.ascontiguousarray(y[idx_tr]), np.ascontiguousarray(y[idx_va])
    scaler = FeatureScaler(); xtr0 = scaler.fit_transform(xtr0, feature_names).astype(np.float32, copy=False); xva0 = scaler.transform(xva0, feature_names).astype(np.float32, copy=False)
    del raw, idx_tr, idx_va; gc.collect(); print(f"[HALVING] Moviendo tensores completos a {device}...")
    xtr, xva, ytr, yva = torch.from_numpy(xtr0).to(device), torch.from_numpy(xva0).to(device), torch.from_numpy(ytr0).to(device), torch.from_numpy(yva0).to(device)
    print(f"[HALVING] Filas={len(df):,} | train={len(ytr0):,} | val={len(yva0):,} | features={len(feature_names)} | positivos={y.mean()*100:.2f}%")
    if device.type == "cuda": print(f"[HALVING] VRAM inicial: {torch.cuda.memory_allocated()/1024**2:,.1f} MiB")
    configs = candidates(args.candidates, args.seed, device); definition = {"train_parquet": str(train_path.resolve()), "feature_set": args.feature_set,
        "candidate_count": args.candidates, "round_epochs": args.round_epochs, "survivors": args.survivors, "val_size": args.val_size,
        "seed": args.seed, "feature_columns": feature_names, "candidates": configs}
    definition_path = out / "run_definition.json"
    if definition_path.exists() and not args.force:
        if json.load(open(definition_path, encoding="utf-8")) != definition: raise RuntimeError("Experimento existente incompatible; use otro nombre o --force")
        print("[HALVING] Reanudación detectada: usando checkpoints existentes")
    else: save_json(definition_path, definition)
    save_json(out / "candidate_configs.json", configs); save_json(out / "scaler.json", {"means": scaler.means, "stds": scaler.stds})
    by_id = {c["candidate_id"]: c for c in configs}; active = list(by_id); all_records = []
    for round_no, (target_epoch, keep) in enumerate(zip(args.round_epochs, args.survivors), 1):
        round_started = time.time(); records = []; print(f"\n[HALVING] RONDA {round_no}/{len(args.round_epochs)} | candidatos={len(active)} | épocas={target_epoch} | pasan={keep}")
        for pos, cid in enumerate(active, 1):
            t0 = time.time(); cfg = by_id[cid]; path = ckpts / f"candidate_{cid:04d}.pt"
            model, optimizer, trained, history, rmetrics = load_candidate(cfg, len(feature_names), path, device, args.seed)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg["pos_weight"]], dtype=torch.float32, device=device)); key = str(target_epoch)
            if trained < target_epoch:
                history += train_to(model, optimizer, criterion, xtr, ytr, trained, target_epoch, cfg["batch_size"], args.seed, cid, round_no); trained = target_epoch
                result, _ = evaluate(model, criterion, xva, yva, yva0, cfg["batch_size"]); rmetrics[key] = result
                checkpoint(path, cfg, model, optimizer, trained, history, rmetrics)
            else: result = rmetrics[key] if key in rmetrics else evaluate(model, criterion, xva, yva, yva0, cfg["batch_size"])[0]
            elapsed = time.time() - t0; eta = (time.time() - round_started) / pos * (len(active) - pos)
            rec = {"round": round_no, "target_epoch": target_epoch, **cfg, **result, "duration_seconds": elapsed}; records.append(rec); all_records.append(rec)
            pd.DataFrame(records).sort_values("pr_auc", ascending=False).to_csv(out / f"round_{round_no}_results.csv", index=False)
            pd.DataFrame(all_records).to_csv(out / "all_round_results.csv", index=False)
            print(f"[HALVING][R{round_no} {pos:02d}/{len(active):02d}] C{cid:03d} | PR-AUC={result['pr_auc']:.5f} | F1={result['f1']:.5f} | Lift@10={result['lift_at_10']:.3f} | batch={cfg['batch_size']:,} | tiempo={duration(elapsed)} | ETA={duration(eta)}")
            del model, optimizer, criterion
        ranked = sorted(records, key=lambda r: r["pr_auc"], reverse=True); active = [r["candidate_id"] for r in ranked[:keep]]; save_json(out / f"round_{round_no}_survivors.json", active)
        print(f"[HALVING] Supervivientes R{round_no}: {active} | duración={duration(time.time()-round_started)}")
    best_id = active[0]; cfg = by_id[best_id]; data = torch.load(ckpts / f"candidate_{best_id:04d}.pt", map_location=device, weights_only=False)
    model = PurchaseFNN(len(feature_names), cfg["hidden_dims"], cfg["dropout"], cfg["activation"]).to(device); model.load_state_dict(data["model_state"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg["pos_weight"]], dtype=torch.float32, device=device)); best, probabilities = evaluate(model, criterion, xva, yva, yva0, cfg["batch_size"])
    torch.save(cpu_state(model.state_dict()), out / "best_model_state.pth"); save_json(out / "best_hparams.json", cfg); save_json(out / "best_history.json", data.get("history", []))
    summary = {"timestamp": datetime.now().isoformat(), "experiment_name": name, "model": "FNN", "search_method": "successive_halving",
        "objective": "maximize_pr_auc_validation", "feature_set": args.feature_set, "train_parquet": str(train_path), "n_rows_total": len(df),
        "n_rows_train": len(ytr0), "n_rows_validation": len(yva0), "target_positive_rate_total": float(y.mean()),
        "split": {"type": "row_stratified", "validation_size": args.val_size, "seed": args.seed}, "base_feature_columns": base,
        "model_feature_columns": feature_names, "excluded_columns": excluded, "cycle_onehot": onehot, "candidate_count": args.candidates,
        "round_epochs": args.round_epochs, "survivors": args.survivors, "best_candidate_id": best_id, "best_params": cfg,
        "best_metrics_validation": best, "duration_seconds": round(time.time()-started, 2), "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch_threads": torch.get_num_threads(),
        "max_gpu_memory_mib": round(torch.cuda.max_memory_allocated()/1024**2, 2) if device.type == "cuda" else None}
    save_json(out / "best_metrics.json", summary); GLOBAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_LOG_FILE, "a", encoding="utf-8") as f: f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    plot_progress(all_records, plots / "halving_progress.png")
    pc, rc, _ = precision_recall_curve(yva0, probabilities); fig, ax = plt.subplots(figsize=(7, 6)); ax.plot(rc, pc); ax.axhline(yva0.mean(), linestyle="--"); ax.set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall — mejor candidato"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(plots / "best_pr_curve.png", dpi=160); plt.close(fig)
    print("\n[HALVING] ================================================================")
    print(f"[HALVING] Mejor candidato: C{best_id:03d} | PR-AUC={best['pr_auc']:.6f} | F1={best['f1']:.6f} | Lift@10={best['lift_at_10']:.4f}")
    print(f"[HALVING] Parámetros: {cfg}"); print(f"[HALVING] Duración total: {duration(time.time()-started)}")
    if device.type == "cuda": print(f"[HALVING] Pico VRAM: {summary['max_gpu_memory_mib']:,.1f} MiB")
    print(f"[HALVING] Artefactos: {out}"); print("[HALVING] W4/test NO se utilizó durante el tuning.")
    print("[HALVING] ================================================================")


if __name__ == "__main__": main()
