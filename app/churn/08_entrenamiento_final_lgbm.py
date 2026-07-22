"""Entrenamiento final y evaluación única del modelo LightGBM de churn.

Protocolo experimental:
1. Carga los splits persistidos por 07_hptuning_lgbm.py.
2. Carga los hiperparámetros, el número de árboles y el umbral seleccionados
   exclusivamente mediante TRAIN y VALIDATION.
3. Une TRAIN + VALIDATION (90 % del conjunto total).
4. Entrena desde cero un único LightGBM sin early stopping.
5. Evalúa una sola vez sobre TEST (10 %), que permaneció aislado del tuning.

Las métricas principales son PR-AUC, ROC-AUC, precision, recall, F1 y Lift@10.
No se calculan métricas Top-K de recomendación porque la unidad de análisis es el
cliente y la salida es binaria: churn / no churn.

Ejemplo:
    python app/churn/08_entrenamiento_final_lgbm.py --threads 16

Para reemplazar una ejecución final existente:
    python app/churn/08_entrenamiento_final_lgbm.py --threads 16 --force
"""
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
from typing import Any, Callable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = (
    REPO_ROOT / "data" / "churn" / "splits" / "split_80_10_10_seed_42"
)
DEFAULT_HPT_DIR = REPO_ROOT / "data" / "models" / "churn" / "hpt" / "lgbm_churn_main"
DEFAULT_LOG = REPO_ROOT / "data" / "logs" / "08_entrenamiento_final_lgbm_churn.log"
GLOBAL_LOG = REPO_ROOT / "data" / "logs" / "churn_lgbm_final_runs.jsonl"
LOCK_FILE = REPO_ROOT / "data" / "locks" / "churn_lgbm_final.lock"

ID_COL = "IDENTIFICACION"
TARGET_COL = "target"
CATEGORICAL_COLUMNS = ["SEXO"]
SEX_CATEGORIES = ["F", "M", "DESCONOCIDO"]
FEATURE_COLUMNS = [
    "dias_desde_ultima_compra",
    "total_compras_24m",
    "gasto_total_24m",
    "ticket_promedio_24m",
    "longitud_relacion_dias",
    "intervalo_promedio",
    "intervalo_maximo",
    "intervalo_cv",
    "recencia_relativa",
    "compras_ultimos_180d",
    "delta_frecuencia_180d",
    "subcategorias_distintas_24m",
    "EDAD",
    "SEXO",
    "EDAD_IMPUTADA",
]


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
    temporary.replace(path)


def create_logger(path: Path) -> Callable[[str], None]:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n=== 08_entrenamiento_final_lgbm.py | "
            f"{datetime.now().isoformat(timespec='seconds')} ===\n"
        )

    def log(message: str = "") -> None:
        print(message, flush=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    return log


def acquire_lock(label: str):
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = open(LOCK_FILE, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "proceso desconocido"
        handle.close()
        raise RuntimeError(f"Ya existe una ejecución LightGBM final activa: {owner}") from exc

    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "label": label,
                "started_at": datetime.now().isoformat(),
            },
            ensure_ascii=False,
        )
    )
    handle.flush()

    def release() -> None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        except Exception:
            pass

    atexit.register(release)
    return handle


def parse_args() -> argparse.Namespace:
    cpus = available_cpus()
    parser = argparse.ArgumentParser(
        description="Entrena LightGBM churn con TRAIN+VALIDATION y evalúa TEST una sola vez"
    )
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--hpt-dir", type=Path, default=DEFAULT_HPT_DIR)
    parser.add_argument("--experiment-name", default="lgbm_churn_final")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--docs-dir", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--threads", type=int, default=min(16, cpus))
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reemplaza los artefactos finales existentes; no modifica HPT ni splits.",
    )
    return parser.parse_args()


def validate_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = [ID_COL, TARGET_COL] + FEATURE_COLUMNS
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label}: faltan columnas requeridas: {missing}")

    frame = frame.copy()
    frame[ID_COL] = frame[ID_COL].astype("string").str.strip()
    frame[TARGET_COL] = pd.to_numeric(frame[TARGET_COL], errors="raise").astype("int8")

    if frame.empty:
        raise ValueError(f"{label}: conjunto vacío")
    if frame[ID_COL].isna().any() or (frame[ID_COL] == "").any():
        raise ValueError(f"{label}: {ID_COL} contiene valores vacíos")
    if frame[ID_COL].duplicated().any():
        raise ValueError(f"{label}: debe existir una sola fila por {ID_COL}")

    target_values = set(frame[TARGET_COL].unique().tolist())
    if target_values != {0, 1}:
        raise ValueError(f"{label}: target debe contener exactamente 0 y 1; valores={target_values}")
    return frame


def validate_disjoint(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    train_ids = set(train[ID_COL].astype(str))
    validation_ids = set(validation[ID_COL].astype(str))
    test_ids = set(test[ID_COL].astype(str))

    if train_ids & validation_ids:
        raise ValueError("Existe solapamiento de clientes entre TRAIN y VALIDATION")
    if train_ids & test_ids:
        raise ValueError("Existe solapamiento de clientes entre TRAIN y TEST")
    if validation_ids & test_ids:
        raise ValueError("Existe solapamiento de clientes entre VALIDATION y TEST")


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        if column == "SEXO":
            sexo = features[column].astype("string").fillna("DESCONOCIDO")
            sexo = sexo.where(sexo.isin(SEX_CATEGORIES), "DESCONOCIDO")
            features[column] = pd.Categorical(sexo, categories=SEX_CATEGORIES)
        else:
            values = pd.to_numeric(features[column], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).astype("float32")
            if values.isna().any():
                count = int(values.isna().sum())
                raise ValueError(f"La característica {column} contiene {count} valores no válidos")
            features[column] = values
    return features


def load_hpt_artifacts(hpt_dir: Path) -> tuple[dict[str, Any], dict[str, Any], int, float]:
    params_path = hpt_dir / "best_hparams.json"
    metrics_path = hpt_dir / "best_metrics.json"
    if not params_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"Faltan artefactos del HPT. Esperados: {params_path} y {metrics_path}"
        )

    with params_path.open(encoding="utf-8") as handle:
        params = json.load(handle)
    with metrics_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    if summary.get("test_used_in_tuning") is not False:
        raise RuntimeError("El resumen HPT no confirma que TEST permaneció aislado")

    hpt_features = summary.get("feature_columns")
    if hpt_features is not None and list(hpt_features) != FEATURE_COLUMNS:
        raise RuntimeError("Las características del HPT no coinciden con las 15 características esperadas")

    if "num_boost_round" not in params:
        raise ValueError("best_hparams.json no contiene num_boost_round")
    rounds = int(params.pop("num_boost_round"))
    if rounds <= 0:
        raise ValueError("num_boost_round debe ser positivo")

    validation_metrics = summary.get("best_metrics_validation", {})
    if "threshold_f1" not in validation_metrics:
        raise ValueError("best_metrics.json no contiene el threshold_f1 de validación")
    threshold = float(validation_metrics["threshold_f1"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("El umbral congelado debe estar entre 0 y 1")

    params.pop("num_threads", None)
    return params, summary, rounds, threshold


def split_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    positives = int((frame[TARGET_COL] == 1).sum())
    negatives = int((frame[TARGET_COL] == 0).sum())
    return {
        "rows": int(len(frame)),
        "target_0": negatives,
        "target_1": positives,
        "positive_rate": float(positives / len(frame)),
    }


def training_progress(total_rounds: int, every: int, rows: list[dict[str, Any]]):
    started = time.time()

    def callback(environment) -> None:
        iteration = environment.iteration + 1
        if iteration != 1 and iteration % every != 0 and iteration != total_rounds:
            return
        elapsed = time.time() - started
        eta = elapsed / max(1, iteration) * (total_rounds - iteration)
        rows.append(
            {
                "iteration": iteration,
                "elapsed_seconds": round(elapsed, 3),
                "eta_seconds": round(eta, 3),
            }
        )
        print(
            f"[LGBM-FINAL][FIT] {iteration}/{total_rounds} | "
            f"{elapsed:.1f}s | ETA={eta:.1f}s",
            flush=True,
        )

    callback.order = 10
    callback.before_iteration = False
    return callback


def evaluate_binary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    base_rate = float(y_true.mean())
    top_count = max(1, int(math.ceil(len(probabilities) * 0.10)))
    top_indices = np.argpartition(probabilities, -top_count)[-top_count:]
    top_positive_count = int(y_true[top_indices].sum())
    top_positive_rate = float(y_true[top_indices].mean())
    total_positives = int(y_true.sum())

    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    capture_at_10 = float(top_positive_count / total_positives) if total_positives else 0.0

    metrics = {
        "n_test": int(len(y_true)),
        "target_0": int((y_true == 0).sum()),
        "target_1": total_positives,
        "positive_rate": base_rate,
        "threshold_frozen_from_validation": float(threshold),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "specificity": specificity,
        "predicted_positive_rate": float(predictions.mean()),
        "lift_at_10": float(top_positive_rate / base_rate) if base_rate else 0.0,
        "top10_rows": top_count,
        "top10_positives": top_positive_count,
        "top10_positive_rate": top_positive_rate,
        "capture_at_10": capture_at_10,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return metrics, predictions


def build_decile_table(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-probabilities, kind="stable")
    sorted_y = y_true[order].astype(np.int8)
    sorted_p = probabilities[order]
    n = len(order)

    ranking = pd.DataFrame(
        {
            "target": sorted_y,
            "probability": sorted_p,
            "position": np.arange(1, n + 1),
        }
    )
    ranking["decile"] = np.minimum(10, ((ranking["position"] - 1) * 10 // n) + 1)

    base_rate = float(sorted_y.mean())
    total_positives = int(sorted_y.sum())
    table = (
        ranking.groupby("decile", as_index=False)
        .agg(
            rows=("target", "size"),
            positives=("target", "sum"),
            positive_rate=("target", "mean"),
            score_min=("probability", "min"),
            score_max=("probability", "max"),
            score_mean=("probability", "mean"),
        )
        .sort_values("decile")
    )
    table["lift"] = table["positive_rate"] / base_rate if base_rate else 0.0
    table["cumulative_rows"] = table["rows"].cumsum()
    table["cumulative_population_pct"] = table["cumulative_rows"] / n
    table["cumulative_positives"] = table["positives"].cumsum()
    table["cumulative_recall"] = (
        table["cumulative_positives"] / total_positives if total_positives else 0.0
    )
    return table


def metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    principal = [
        ("PR-AUC", metrics["pr_auc"]),
        ("ROC-AUC", metrics["roc_auc"]),
        ("Precisión", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1", metrics["f1"]),
        ("Lift@10", metrics["lift_at_10"]),
        ("Tasa positiva TEST", metrics["positive_rate"]),
        ("Umbral congelado", metrics["threshold_frozen_from_validation"]),
    ]
    return pd.DataFrame(principal, columns=["Métrica", "Resultado"])


def hyperparameters_table(
    params: dict[str, Any],
    rounds: int,
    threshold: float,
    candidate_id: Any,
) -> pd.DataFrame:
    rows: list[tuple[str, Any]] = [("Candidato seleccionado", candidate_id)]
    preferred = [
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_data_in_leaf",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "lambda_l1",
        "lambda_l2",
        "min_gain_to_split",
        "max_bin",
        "scale_pos_weight",
    ]
    labels = {
        "learning_rate": "Tasa de aprendizaje",
        "num_leaves": "Número de hojas",
        "max_depth": "Profundidad máxima",
        "min_data_in_leaf": "Mínimo de registros por hoja",
        "feature_fraction": "Fracción de características",
        "bagging_fraction": "Fracción de observaciones",
        "bagging_freq": "Frecuencia de bagging",
        "lambda_l1": "Regularización L1",
        "lambda_l2": "Regularización L2",
        "min_gain_to_split": "Ganancia mínima de división",
        "max_bin": "Número máximo de bins",
        "scale_pos_weight": "Peso de la clase positiva",
    }
    for key in preferred:
        if key in params:
            rows.append((labels[key], params[key]))
    rows.extend(
        [
            ("Número final de árboles", rounds),
            ("Umbral de clasificación", threshold),
        ]
    )
    return pd.DataFrame(rows, columns=["Hiperparámetro", "Valor"])


def save_confusion_table(metrics: dict[str, Any], path: Path) -> pd.DataFrame:
    table = pd.DataFrame(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ],
        index=["Real 0", "Real 1"],
        columns=["Predicho 0", "Predicho 1"],
    )
    table.to_csv(path)
    return table


def plot_precision_recall(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metrics: dict[str, Any],
    output: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig, axis = plt.subplots(figsize=(7.2, 5.8))
    axis.plot(recall, precision, label=f"LightGBM (PR-AUC={metrics['pr_auc']:.4f})")
    axis.axhline(
        metrics["positive_rate"],
        linestyle="--",
        label=f"Prevalencia TEST ({metrics['positive_rate']:.4f})",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precisión")
    axis.set_title("Curva Precision–Recall — LightGBM churn sobre TEST")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metrics: dict[str, Any],
    output: Path,
) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    fig, axis = plt.subplots(figsize=(7.2, 5.8))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"LightGBM (ROC-AUC={metrics['roc_auc']:.4f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", label="Clasificador aleatorio")
    axis.set_xlabel("Tasa de falsos positivos")
    axis.set_ylabel("Tasa de verdaderos positivos")
    axis.set_title("Curva ROC — LightGBM churn sobre TEST")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    confusion: pd.DataFrame,
    output: Path,
) -> None:
    values = confusion.to_numpy(dtype=float)
    row_totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, row_totals, out=np.zeros_like(values), where=row_totals > 0)

    fig, axis = plt.subplots(figsize=(6.4, 5.5))
    image = axis.imshow(normalized, vmin=0, vmax=1)
    axis.set_xticks([0, 1], labels=["Predicho 0", "Predicho 1"])
    axis.set_yticks([0, 1], labels=["Real 0", "Real 1"])
    axis.set_xlabel("Clase predicha")
    axis.set_ylabel("Clase real")
    axis.set_title("Matriz de confusión — umbral congelado")

    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{int(values[row, column]):,}\n{normalized[row, column] * 100:.1f}%",
                ha="center",
                va="center",
            )
    fig.colorbar(image, ax=axis, label="Proporción dentro de la clase real")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_gain(deciles: pd.DataFrame, output: Path) -> None:
    x = np.concatenate([[0.0], deciles["cumulative_population_pct"].to_numpy(float)])
    y = np.concatenate([[0.0], deciles["cumulative_recall"].to_numpy(float)])

    fig, axis = plt.subplots(figsize=(7.2, 5.8))
    axis.plot(x, y, marker="o", label="LightGBM")
    axis.plot([0, 1], [0, 1], linestyle="--", label="Selección aleatoria")
    axis.axvline(0.10, linestyle=":", label="Top 10 %")
    axis.set_xlabel("Proporción acumulada de clientes")
    axis.set_ylabel("Proporción acumulada de churners capturados")
    axis.set_title("Curva de ganancia acumulada — LightGBM churn")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 5.8))
    axis.hist(probabilities[y_true == 0], bins=50, density=True, alpha=0.55, label="No churn")
    axis.hist(probabilities[y_true == 1], bins=50, density=True, alpha=0.55, label="Churn")
    axis.axvline(threshold, linestyle="--", label=f"Umbral={threshold:.4f}")
    axis.set_xlabel("Probabilidad estimada de churn")
    axis.set_ylabel("Densidad")
    axis.set_title("Distribución de puntuaciones por clase — TEST")
    axis.grid(alpha=0.20)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame, output: Path) -> None:
    ordered = importance.sort_values("gain_pct", ascending=True)
    fig, axis = plt.subplots(figsize=(9.2, max(5.5, len(ordered) * 0.38)))
    axis.barh(ordered["feature"], ordered["gain_pct"] * 100)
    axis.set_xlabel("Importancia por gain (%)")
    axis.set_title("Importancia de características del LightGBM final")
    axis.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def copy_table_to_both(table: pd.DataFrame, name: str, output_dir: Path, docs_dir: Path) -> None:
    table.to_csv(output_dir / name, index=False)
    table.to_csv(docs_dir / name, index=False)


def main() -> None:
    args = parse_args()
    log = create_logger(args.log)
    total_started = time.time()
    stage_times: dict[str, float] = {}

    if args.threads <= 0:
        raise ValueError("--threads debe ser mayor que 0")
    if args.log_every <= 0:
        raise ValueError("--log-every debe ser mayor que 0")
    cpus = available_cpus()
    if args.threads > cpus:
        raise ValueError(f"Solicitas {args.threads} threads, pero la reserva permite {cpus}")

    splits_dir = args.splits_dir.expanduser().resolve()
    hpt_dir = args.hpt_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT / "data" / "models" / "churn" / "final" / args.experiment_name
    )
    docs_dir = (
        args.docs_dir.expanduser().resolve()
        if args.docs_dir is not None
        else REPO_ROOT / "docs" / "final" / "churn" / args.experiment_name
    )

    split_paths = {
        "train": splits_dir / "train.parquet",
        "validation": splits_dir / "validation.parquet",
        "test": splits_dir / "test.parquet",
        "summary": splits_dir / "split_summary.json",
    }
    for name, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Falta el artefacto de split {name}: {path}")
    if not hpt_dir.exists():
        raise FileNotFoundError(hpt_dir)

    if output_dir.exists() and not args.force:
        raise FileExistsError(
            f"Ya existe una evaluación final en {output_dir}. "
            "No se volverá a tocar TEST salvo que ejecutes explícitamente con --force."
        )
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    if args.force and docs_dir.exists():
        shutil.rmtree(docs_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    acquire_lock(f"final:churn:{args.experiment_name}")

    log("\n=== CONFIGURACIÓN FINAL LIGHTGBM CHURN ===")
    log(f"splits_dir: {splits_dir}")
    log(f"hpt_dir: {hpt_dir}")
    log(f"output_dir: {output_dir}")
    log(f"docs_dir: {docs_dir}")
    log(f"threads: {args.threads}/{cpus}")
    log("protocolo: TRAIN + VALIDATION para entrenamiento final; TEST para evaluación única")

    stage_started = time.time()
    log("\n[1/7] Cargando y verificando splits persistidos ...")
    train_df = validate_frame(pd.read_parquet(split_paths["train"]), "TRAIN")
    validation_df = validate_frame(pd.read_parquet(split_paths["validation"]), "VALIDATION")
    test_df = validate_frame(pd.read_parquet(split_paths["test"]), "TEST")
    validate_disjoint(train_df, validation_df, test_df)
    with split_paths["summary"].open(encoding="utf-8") as handle:
        split_summary = json.load(handle)

    for name, frame in (
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df),
    ):
        distribution = split_distribution(frame)
        log(
            f"   {name:<10}: filas={distribution['rows']:,} | "
            f"target0={distribution['target_0']:,} | target1={distribution['target_1']:,} | "
            f"churn={distribution['positive_rate'] * 100:.2f}%"
        )
    stage_times["load_and_validate_splits"] = round(time.time() - stage_started, 3)

    stage_started = time.time()
    log("\n[2/7] Cargando configuración congelada del HPT ...")
    params, hpt_summary, rounds, threshold = load_hpt_artifacts(hpt_dir)
    selected_candidate = hpt_summary.get("best_candidate_id")
    validation_metrics = hpt_summary.get("best_metrics_validation", {})
    scale_pos_weight = float(params.get("scale_pos_weight", 1.0))
    log(f"   candidato: C{int(selected_candidate):04d}" if selected_candidate is not None else "   candidato: no registrado")
    log(f"   num_boost_round: {rounds}")
    log(f"   scale_pos_weight: {scale_pos_weight:.6f}")
    log(f"   threshold F1 congelado: {threshold:.6f}")
    log(f"   PR-AUC de validación del ganador: {float(validation_metrics.get('pr_auc', float('nan'))):.6f}")
    log("   TEST no se usa para cambiar hiperparámetros, árboles ni umbral")
    stage_times["load_hpt"] = round(time.time() - stage_started, 3)

    stage_started = time.time()
    log("\n[3/7] Construyendo conjunto final TRAIN + VALIDATION ...")
    development_df = pd.concat([train_df, validation_df], ignore_index=True)
    if development_df[ID_COL].duplicated().any():
        raise RuntimeError("TRAIN + VALIDATION contiene clientes duplicados")

    x_development = prepare_features(development_df)
    x_test = prepare_features(test_df)
    y_development = development_df[TARGET_COL].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.float32)
    log(f"   entrenamiento final: {len(development_df):,} clientes (90%)")
    log(f"   evaluación final: {len(test_df):,} clientes (10%)")
    log(f"   características: {len(FEATURE_COLUMNS)}")
    stage_times["prepare_final_matrices"] = round(time.time() - stage_started, 3)

    stage_started = time.time()
    log("\n[4/7] Entrenando LightGBM final desde cero ...")
    final_params = dict(params)
    final_params["num_threads"] = args.threads
    progress_rows: list[dict[str, Any]] = []
    train_set = lgb.Dataset(
        x_development,
        label=y_development,
        feature_name=FEATURE_COLUMNS,
        categorical_feature=CATEGORICAL_COLUMNS,
        free_raw_data=True,
    )
    booster = lgb.train(
        final_params,
        train_set,
        num_boost_round=rounds,
        callbacks=[training_progress(rounds, args.log_every, progress_rows)],
    )
    stage_times["final_training"] = round(time.time() - stage_started, 3)
    log(f"   entrenamiento completado con {rounds} árboles; no se aplicó early stopping")

    stage_started = time.time()
    log("\n[5/7] Generando predicciones y métricas sobre TEST ...")
    probabilities = booster.predict(x_test, num_iteration=rounds, num_threads=args.threads)
    test_metrics, predictions = evaluate_binary(y_test, probabilities, threshold)
    deciles = build_decile_table(y_test, probabilities)
    stage_times["test_prediction_and_metrics"] = round(time.time() - stage_started, 3)

    log(f"   PR-AUC: {test_metrics['pr_auc']:.6f}")
    log(f"   ROC-AUC: {test_metrics['roc_auc']:.6f}")
    log(f"   Precisión: {test_metrics['precision']:.6f}")
    log(f"   Recall: {test_metrics['recall']:.6f}")
    log(f"   F1: {test_metrics['f1']:.6f}")
    log(f"   Lift@10: {test_metrics['lift_at_10']:.4f}")
    log(
        f"   Matriz: TN={test_metrics['true_negative']:,} | FP={test_metrics['false_positive']:,} | "
        f"FN={test_metrics['false_negative']:,} | TP={test_metrics['true_positive']:,}"
    )

    stage_started = time.time()
    log("\n[6/7] Guardando modelo, tablas, predicciones y gráficas ...")
    booster.save_model(str(output_dir / "model.txt"), num_iteration=rounds)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    total_gain = float(importance["gain"].sum())
    importance["gain_pct"] = importance["gain"] / total_gain if total_gain else 0.0
    importance["gain_cumulative_pct"] = importance["gain_pct"].cumsum()

    main_metrics_table = metrics_table(test_metrics)
    final_hparams_table = hyperparameters_table(
        params,
        rounds,
        threshold,
        f"C{int(selected_candidate):04d}" if selected_candidate is not None else "No registrado",
    )
    confusion = save_confusion_table(test_metrics, output_dir / "confusion_matrix.csv")
    confusion.to_csv(docs_dir / "confusion_matrix.csv")

    copy_table_to_both(main_metrics_table, "test_metrics_table.csv", output_dir, docs_dir)
    copy_table_to_both(final_hparams_table, "final_hyperparameters_table.csv", output_dir, docs_dir)
    copy_table_to_both(importance, "feature_importance.csv", output_dir, docs_dir)
    copy_table_to_both(deciles, "score_deciles.csv", output_dir, docs_dir)
    copy_table_to_both(pd.DataFrame(progress_rows), "training_progress.csv", output_dir, docs_dir)

    prediction_frame = test_df[[ID_COL, TARGET_COL]].copy()
    prediction_frame["probability_churn"] = probabilities
    prediction_frame["prediction_churn"] = predictions
    prediction_frame["risk_rank"] = pd.Series(probabilities).rank(method="first", ascending=False).astype(int)
    prediction_frame["risk_decile"] = np.minimum(
        10,
        ((prediction_frame["risk_rank"] - 1) * 10 // len(prediction_frame)) + 1,
    ).astype(int)
    prediction_frame.to_parquet(output_dir / "test_predictions.parquet", index=False)

    plot_precision_recall(
        y_test,
        probabilities,
        test_metrics,
        docs_dir / "precision_recall_test.png",
    )
    plot_roc(y_test, probabilities, test_metrics, docs_dir / "roc_test.png")
    plot_confusion_matrix(confusion, docs_dir / "confusion_matrix_test.png")
    plot_cumulative_gain(deciles, docs_dir / "cumulative_gain_test.png")
    plot_score_distribution(
        y_test,
        probabilities,
        threshold,
        docs_dir / "score_distribution_test.png",
    )
    plot_feature_importance(importance, docs_dir / "feature_importance_gain_final.png")
    stage_times["save_artifacts"] = round(time.time() - stage_started, 3)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": args.experiment_name,
        "model": "LightGBM",
        "problem": "churn_cliente",
        "training_protocol": "fit_train_plus_validation_90pct_then_single_test_10pct_evaluation",
        "splits_dir": str(splits_dir),
        "split_summary": split_summary,
        "hpt_dir": str(hpt_dir),
        "hpt_best_candidate_id": selected_candidate,
        "hpt_validation_metrics": validation_metrics,
        "hyperparameters": {**final_params, "num_boost_round": rounds},
        "frozen_threshold": threshold,
        "threshold_source": "best_validation_trial_threshold_f1",
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "sex_categories": SEX_CATEGORIES,
        "n_train_original": int(len(train_df)),
        "n_validation_original": int(len(validation_df)),
        "n_final_train_development": int(len(development_df)),
        "n_test": int(len(test_df)),
        "final_train_positive_rate": float(y_development.mean()),
        "test_positive_rate": float(y_test.mean()),
        "test_metrics": test_metrics,
        "test_used_for_hyperparameters": False,
        "test_used_for_num_boost_round": False,
        "test_used_for_threshold": False,
        "test_evaluated_after_final_fit": True,
        "threads": args.threads,
        "stage_times_seconds": stage_times,
        "duration_seconds": round(time.time() - total_started, 2),
        "artifacts": {
            "model": str(output_dir / "model.txt"),
            "predictions": str(output_dir / "test_predictions.parquet"),
            "metrics_table": str(docs_dir / "test_metrics_table.csv"),
            "hyperparameters_table": str(docs_dir / "final_hyperparameters_table.csv"),
            "confusion_matrix": str(docs_dir / "confusion_matrix.csv"),
            "deciles": str(docs_dir / "score_deciles.csv"),
            "precision_recall_plot": str(docs_dir / "precision_recall_test.png"),
            "roc_plot": str(docs_dir / "roc_test.png"),
            "confusion_plot": str(docs_dir / "confusion_matrix_test.png"),
            "cumulative_gain_plot": str(docs_dir / "cumulative_gain_test.png"),
            "score_distribution_plot": str(docs_dir / "score_distribution_test.png"),
            "feature_importance_plot": str(docs_dir / "feature_importance_gain_final.png"),
        },
    }

    save_json(output_dir / "run_summary.json", summary)
    save_json(docs_dir / "run_summary.json", summary)
    save_json(output_dir / "test_metrics.json", test_metrics)
    save_json(docs_dir / "test_metrics.json", test_metrics)

    GLOBAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with GLOBAL_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    log("\n[7/7] Experimento LightGBM finalizado")
    log("[LGBM-FINAL] ================================================================")
    log(
        f"[LGBM-FINAL] PR-AUC={test_metrics['pr_auc']:.6f} | "
        f"ROC-AUC={test_metrics['roc_auc']:.6f} | F1={test_metrics['f1']:.6f}"
    )
    log(
        f"[LGBM-FINAL] Precision={test_metrics['precision']:.6f} | "
        f"Recall={test_metrics['recall']:.6f} | Lift@10={test_metrics['lift_at_10']:.4f}"
    )
    log(
        f"[LGBM-FINAL] Desarrollo={len(development_df):,} | TEST={len(test_df):,} | "
        f"árboles={rounds} | threshold={threshold:.6f} | weight={scale_pos_weight:.4f}"
    )
    log(f"[LGBM-FINAL] Modelo: {output_dir / 'model.txt'}")
    log(f"[LGBM-FINAL] Resultados para tesis: {docs_dir}")
    log("[LGBM-FINAL] TEST no modificó ninguna decisión del modelo.")
    log(f"[LGBM-FINAL] Tiempo total: {(time.time() - total_started) / 60:.2f} minutos")
    log("[LGBM-FINAL] ================================================================")


if __name__ == "__main__":
    main()
