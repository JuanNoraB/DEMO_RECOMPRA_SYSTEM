"""
hptuning.py — Búsqueda de hiperparámetros para FNN de recompra binaria.

Trabaja directamente con un parquet de features ya construido.
No vuelve a leer el histórico ni recalcula ventanas temporales.

Experimentos:
  - gain95: usa las ocho features que acumularon 95% del gain.
  - all: usa todas las features numéricas válidas, excluyendo IDs, target,
         columnas constantes y texto; añade one-hot del tipo de ciclo.

Objetivo de Optuna:
  - Maximizar PR-AUC / Average Precision sobre validación estratificada.

Registra por trial:
  - PR-AUC, ROC-AUC, F1, precision, recall, Lift@10 y threshold óptimo.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

optuna.logging.set_verbosity(optuna.logging.WARNING)

from config import FNN_GAIN95_FEATURES, MODELS_DIR, TIPO_CICLO_CATEGORIES, TIPO_CICLO_COL
from train_fnn import FeatureScaler, PurchaseFNN

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_HPT_DIR = ROOT_DIR / "docs" / "hpt"
GLOBAL_LOG_FILE = ROOT_DIR / "data" / "logs" / "fnn_hpt_runs.jsonl"

ID_OR_TARGET_COLUMNS = {
    "target",
    "nucleo",
    "COD_SUBCATEGORIA",
    "CODIGO_FAMILIA",
    "Ciclos_CODIGO_FAMILIA",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero torch.cuda.is_available() es False")
    return torch.device(requested)


def stratified_limit(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    _, sampled = train_test_split(
        df,
        test_size=max_rows,
        random_state=seed,
        stratify=df["target"],
    )
    return sampled.reset_index(drop=True)


def resolve_feature_set(df: pd.DataFrame, feature_set: str) -> tuple[list[str], bool, list[str]]:
    excluded: list[str] = []

    if feature_set == "gain95":
        missing = [c for c in FNN_GAIN95_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan features GAIN95 en el parquet: {missing}")
        return list(FNN_GAIN95_FEATURES), False, excluded

    selected: list[str] = []
    for col in df.columns:
        if col in ID_OR_TARGET_COLUMNS or col == TIPO_CICLO_COL:
            excluded.append(col)
            continue
        if not is_numeric_dtype(df[col]):
            excluded.append(col)
            continue
        if df[col].nunique(dropna=False) <= 1:
            excluded.append(col)
            continue
        selected.append(col)

    if not selected:
        raise ValueError("No se encontraron features numéricas válidas para feature-set=all")

    return selected, True, excluded


def build_feature_matrix(
    df: pd.DataFrame,
    base_columns: list[str],
    include_cycle_onehot: bool,
) -> tuple[np.ndarray, list[str]]:
    x_df = df[base_columns].copy()

    if include_cycle_onehot and TIPO_CICLO_COL in df.columns:
        dummies = pd.get_dummies(
            pd.Categorical(df[TIPO_CICLO_COL], categories=TIPO_CICLO_CATEGORIES),
            prefix="tipo",
            drop_first=True,
            dtype=np.float32,
        )
        dummies.index = x_df.index
        x_df = pd.concat([x_df, dummies], axis=1)

    columns = list(x_df.columns)
    x = np.nan_to_num(
        x_df.to_numpy(dtype=np.float32, copy=True),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return x, columns


def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            chunks.append(torch.sigmoid(model(xb)).detach().cpu().numpy().ravel())
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pr_auc = float(average_precision_score(y_true, proba))
    roc_auc = float(roc_auc_score(y_true, proba))

    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        best_threshold = 0.5
    else:
        denom = precision_curve[:-1] + recall_curve[:-1]
        f1_curve = np.divide(
            2 * precision_curve[:-1] * recall_curve[:-1],
            denom,
            out=np.zeros_like(denom),
            where=denom > 0,
        )
        best_threshold = float(thresholds[int(np.argmax(f1_curve))])

    pred = (proba >= best_threshold).astype(np.int8)
    precision = float(precision_score(y_true, pred, zero_division=0))
    recall = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))

    base_rate = float(np.mean(y_true))
    top_n = max(1, int(math.ceil(len(proba) * 0.10)))
    top_idx = np.argpartition(proba, -top_n)[-top_n:]
    top_rate = float(np.mean(y_true[top_idx]))
    lift_at_10 = float(top_rate / base_rate) if base_rate > 0 else 0.0

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "lift_at_10": lift_at_10,
        "threshold_f1": best_threshold,
        "base_rate": base_rate,
        "top10_positive_rate": top_rate,
    }


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = criterion(model(xb), yb)
            rows = len(xb)
            total_loss += float(loss.item()) * rows
            total_rows += rows
    return total_loss / max(total_rows, 1)


def train_one_trial(
    trial: optuna.Trial,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    y_val: np.ndarray,
    input_dim: int,
    epochs: int,
    patience: int,
    min_delta: float,
    loader_workers: int,
    device: torch.device,
    seed: int,
) -> tuple[dict, dict, list[dict], dict]:
    set_seed(seed + trial.number)

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "leaky_relu"])
    n_layers = trial.suggest_int("n_layers", 2, 4)
    hidden_dims = [
        trial.suggest_categorical(f"layer_{i}", [32, 64, 128, 256])
        for i in range(n_layers)
    ]
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    pos_weight = trial.suggest_categorical("pos_weight", [1.0, 2.0, 4.0])

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=loader_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(batch_size, 4096),
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=pin_memory,
    )

    model = PurchaseFNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_rows = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            rows = len(xb)
            total_train_loss += float(loss.item()) * rows
            total_rows += rows

        train_loss = total_train_loss / max(total_rows, 1)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        trial.report(-val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("El trial no produjo un estado válido")

    model.load_state_dict(best_state)
    model.to(device)
    proba = predict_probabilities(model, val_loader, device)
    metrics = classification_metrics(y_val, proba)
    metrics["best_val_loss"] = float(best_val_loss)
    metrics["best_epoch"] = int(best_epoch)
    metrics["epochs_executed"] = int(len(history))

    for key, value in metrics.items():
        trial.set_user_attr(key, float(value))

    params = {
        "lr": lr,
        "dropout": dropout,
        "activation": activation,
        "hidden_dims": hidden_dims,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "pos_weight": pos_weight,
    }
    return metrics, best_state, history, params


def save_scaler(scaler: FeatureScaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"means": scaler.means, "stds": scaler.stds}, f, indent=2)


def save_trial_tables(study: optuna.Study, output_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for trial in study.trials:
        rows.append({
            "trial": trial.number,
            "state": trial.state.name,
            "objective_pr_auc": trial.value,
            **{f"param_{k}": v for k, v in trial.params.items()},
            **trial.user_attrs,
        })

    df_trials = pd.DataFrame(rows)
    df_trials.to_csv(output_dir / "trials.csv", index=False)
    with open(output_dir / "trials.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return df_trials


def plot_optimization_history(df_trials: pd.DataFrame, path: Path) -> None:
    completed = df_trials[
        (df_trials["state"] == "COMPLETE") & df_trials["objective_pr_auc"].notna()
    ].copy()
    if completed.empty:
        return
    completed = completed.sort_values("trial")
    completed["best_so_far"] = completed["objective_pr_auc"].cummax()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(completed["trial"], completed["objective_pr_auc"], marker="o", label="PR-AUC por trial")
    ax.plot(completed["trial"], completed["best_so_far"], linewidth=2, label="Mejor acumulado")
    ax.set_xlabel("Trial")
    ax.set_ylabel("PR-AUC validación")
    ax.set_title("Historial de optimización FNN")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, proba: np.ndarray, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label="FNN")
    ax.axhline(baseline, linestyle="--", label=f"Prevalencia={baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall — mejor trial")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_loss_history(history: list[dict], path: Path) -> None:
    hist = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["epoch"], hist["train_loss"], label="Train loss")
    ax.plot(hist["epoch"], hist["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE ponderada")
    ax.set_title("Curva de aprendizaje — mejor trial")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_metrics(y_true: np.ndarray, proba: np.ndarray, path: Path) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return
    precision = precision[:-1]
    recall = recall[:-1]
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precision, label="Precision")
    ax.plot(thresholds, recall, label="Recall")
    ax.plot(thresholds, f1, label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Métrica")
    ax.set_title("Selección de threshold en validación")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="HPT FNN — optimiza PR-AUC sobre validación estratificada")
    parser.add_argument("--train-parquet", type=str, required=True)
    parser.add_argument("--feature-set", choices=["gain95", "all"], required=True)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Muestra estratificada opcional para pruebas rápidas")
    args = parser.parse_args()

    if not 0 < args.val_size < 1:
        raise ValueError("--val-size debe estar entre 0 y 1")
    if args.trials <= 0 or args.epochs <= 0:
        raise ValueError("--trials y --epochs deben ser mayores que 0")

    set_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    device = resolve_device(args.device)

    train_path = Path(args.train_parquet)
    if not train_path.exists():
        raise FileNotFoundError(f"No existe: {train_path}")

    experiment_name = args.experiment_name or f"fnn_{args.feature_set}"
    output_dir = MODELS_DIR / "hpt" / experiment_name
    plots_dir = DOCS_HPT_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[HPT] Experimento: {experiment_name}")
    print(f"[HPT] Parquet: {train_path}")
    print(f"[HPT] Feature set: {args.feature_set}")
    print(f"[HPT] Device: {device} | torch_threads={torch.get_num_threads()}")

    start_time = time.time()
    df = pd.read_parquet(train_path)
    if "target" not in df.columns:
        raise ValueError("El parquet no contiene columna target")
    if df["target"].nunique() != 2:
        raise ValueError("target debe contener exactamente dos clases")

    df = stratified_limit(df, args.max_rows, args.seed)
    base_columns, include_cycle_onehot, excluded_columns = resolve_feature_set(df, args.feature_set)
    x_raw, feature_columns = build_feature_matrix(df, base_columns, include_cycle_onehot)
    y = df["target"].to_numpy(dtype=np.float32)

    indices = np.arange(len(df))
    idx_train, idx_val = train_test_split(
        indices,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y,
    )

    x_train_raw = np.ascontiguousarray(x_raw[idx_train])
    x_val_raw = np.ascontiguousarray(x_raw[idx_val])
    y_train = np.ascontiguousarray(y[idx_train])
    y_val = np.ascontiguousarray(y[idx_val])

    scaler = FeatureScaler()
    x_train = scaler.fit_transform(x_train_raw, feature_columns).astype(np.float32, copy=False)
    x_val = scaler.transform(x_val_raw, feature_columns).astype(np.float32, copy=False)

    del x_raw, x_train_raw, x_val_raw, indices, idx_train, idx_val
    gc.collect()

    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    print(
        f"[HPT] Filas total={len(df):,} | train={len(train_dataset):,} | "
        f"val={len(val_dataset):,} | positivos={int(y.sum()):,} ({y.mean()*100:.2f}%)"
    )
    print(f"[HPT] Features modelo={len(feature_columns)}")
    print(f"[HPT] Columnas base: {base_columns}")
    if include_cycle_onehot:
        print(f"[HPT] One-hot ciclo: activado desde {TIPO_CICLO_COL}")
    if excluded_columns:
        print(f"[HPT] Excluidas en modo all: {excluded_columns}")

    best_artifacts = {
        "score": -float("inf"),
        "trial_number": None,
        "metrics": None,
        "state_dict": None,
        "history": None,
        "params": None,
        "proba": None,
    }

    def objective(trial: optuna.Trial) -> float:
        metrics, state_dict, history, params = train_one_trial(
            trial=trial,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            y_val=y_val,
            input_dim=len(feature_columns),
            epochs=args.epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            loader_workers=args.loader_workers,
            device=device,
            seed=args.seed,
        )
        score = metrics["pr_auc"]
        if score > best_artifacts["score"]:
            best_artifacts.update({
                "score": score,
                "trial_number": trial.number,
                "metrics": metrics,
                "state_dict": state_dict,
                "history": history,
                "params": params,
            })
            model = PurchaseFNN(
                input_dim=len(feature_columns),
                hidden_dims=params["hidden_dims"],
                dropout=params["dropout"],
                activation=params["activation"],
            ).to(device)
            model.load_state_dict(state_dict)
            val_loader = DataLoader(
                val_dataset,
                batch_size=max(params["batch_size"], 4096),
                shuffle=False,
                num_workers=args.loader_workers,
                pin_memory=device.type == "cuda",
            )
            best_artifacts["proba"] = predict_probabilities(model, val_loader, device)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print(
            f"[HPT][Trial {trial.number:03d}] PR-AUC={metrics['pr_auc']:.5f} | "
            f"F1={metrics['f1']:.5f} | Lift@10={metrics['lift_at_10']:.3f} | "
            f"epoch={metrics['best_epoch']}"
        )
        return score

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True, gc_after_trial=True)

    if best_artifacts["state_dict"] is None:
        raise RuntimeError("No hubo trials completados")

    df_trials = save_trial_tables(study, output_dir)
    joblib.dump(study, output_dir / "study.pkl")
    torch.save(best_artifacts["state_dict"], output_dir / "best_model_state.pth")
    save_scaler(scaler, output_dir / "scaler.json")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "model": "FNN",
        "objective": "maximize_pr_auc_validation",
        "feature_set": args.feature_set,
        "train_parquet": str(train_path),
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(train_dataset)),
        "n_rows_validation": int(len(val_dataset)),
        "target_positive_rate_total": float(y.mean()),
        "split": {
            "type": "row_stratified",
            "validation_size": args.val_size,
            "seed": args.seed,
        },
        "base_feature_columns": base_columns,
        "model_feature_columns": feature_columns,
        "excluded_columns": excluded_columns,
        "cycle_onehot": include_cycle_onehot,
        "best_trial": int(best_artifacts["trial_number"]),
        "best_params": best_artifacts["params"],
        "best_metrics_validation": best_artifacts["metrics"],
        "trials_requested": args.trials,
        "trials_completed": int(sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)),
        "trials_pruned": int(sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)),
        "epochs_max": args.epochs,
        "duration_seconds": round(time.time() - start_time, 2),
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
    }

    with open(output_dir / "best_hparams.json", "w", encoding="utf-8") as f:
        json.dump(summary["best_params"], f, ensure_ascii=False, indent=2)
    with open(output_dir / "best_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    GLOBAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    plot_optimization_history(df_trials, plots_dir / "optimization_history.png")
    plot_pr_curve(y_val, best_artifacts["proba"], plots_dir / "best_pr_curve.png")
    plot_loss_history(best_artifacts["history"], plots_dir / "best_loss_curve.png")
    plot_threshold_metrics(y_val, best_artifacts["proba"], plots_dir / "threshold_metrics.png")

    print("\n[HPT] ================================================================")
    print(f"[HPT] Mejor trial: {best_artifacts['trial_number']}")
    print(f"[HPT] PR-AUC validación: {best_artifacts['metrics']['pr_auc']:.6f}")
    print(f"[HPT] F1 validación: {best_artifacts['metrics']['f1']:.6f}")
    print(f"[HPT] Precision validación: {best_artifacts['metrics']['precision']:.6f}")
    print(f"[HPT] Recall validación: {best_artifacts['metrics']['recall']:.6f}")
    print(f"[HPT] Lift@10 validación: {best_artifacts['metrics']['lift_at_10']:.4f}")
    print(f"[HPT] Threshold F1: {best_artifacts['metrics']['threshold_f1']:.6f}")
    print(f"[HPT] Parámetros: {best_artifacts['params']}")
    print(f"[HPT] Artefactos: {output_dir}")
    print(f"[HPT] Gráficas: {plots_dir}")
    print("[HPT] El parquet W4/test NO se utilizó durante el tuning.")
    print("[HPT] ================================================================")


if __name__ == "__main__":
    main()
