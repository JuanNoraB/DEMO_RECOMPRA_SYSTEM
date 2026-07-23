"""Entrenamiento final y evaluación única de Balanced Random Forest para churn.

Protocolo experimental:
1. Reutiliza los splits persistidos 80/10/10.
2. Carga los hiperparámetros, el número de árboles y el umbral seleccionados
   exclusivamente con TRAIN y VALIDATION por 09_hptuning_brf.py.
3. Une TRAIN + VALIDATION (90 %).
4. Entrena desde cero un único Balanced Random Forest.
5. Evalúa una sola vez sobre TEST (10 %), aislado durante el tuning.
6. Genera métricas, tablas y figuras equivalentes a las del modelo LightGBM.
7. Si existen las predicciones finales de LightGBM, produce comparaciones directas
   entre ambos modelos sobre exactamente los mismos clientes de TEST.

Ejemplo:
    python app/churn/10_entrenamiento_final_brf.py --threads 16

Para reemplazar una ejecución final existente:
    python app/churn/10_entrenamiento_final_brf.py --threads 16 --force
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
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
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

try:
    import imblearn
    from imblearn.ensemble import BalancedRandomForestClassifier
except ImportError as exc:
    raise SystemExit(
        "Falta imbalanced-learn en el entorno activo. Instálelo con: "
        "python -m pip install imbalanced-learn"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = (
    REPO_ROOT / "data" / "churn" / "splits" / "split_80_10_10_seed_42"
)
DEFAULT_HPT_DIR = REPO_ROOT / "data" / "models" / "churn" / "hpt" / "brf_churn_main"
DEFAULT_LGBM_FINAL_DIR = (
    REPO_ROOT / "data" / "models" / "churn" / "final" / "lgbm_churn_final"
)
DEFAULT_LOG = REPO_ROOT / "data" / "logs" / "10_entrenamiento_final_brf_churn.log"
GLOBAL_LOG = REPO_ROOT / "data" / "logs" / "churn_brf_final_runs.jsonl"
LOCK_FILE = REPO_ROOT / "data" / "locks" / "churn_brf_final.lock"

ID_COL = "IDENTIFICACION"
TARGET_COL = "target"
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
NUMERIC_FEATURE_COLUMNS = [column for column in FEATURE_COLUMNS if column != "SEXO"]
MODEL_FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + [
    "SEXO_F",
    "SEXO_M",
    "SEXO_DESCONOCIDO",
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
            "\n=== 10_entrenamiento_final_brf.py | "
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
        raise RuntimeError(f"Ya existe una ejecución BRF final activa: {owner}") from exc

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
        description=(
            "Entrena Balanced Random Forest churn con TRAIN+VALIDATION "
            "y evalúa TEST una sola vez"
        )
    )
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--hpt-dir", type=Path, default=DEFAULT_HPT_DIR)
    parser.add_argument("--lgbm-final-dir", type=Path, default=DEFAULT_LGBM_FINAL_DIR)
    parser.add_argument("--experiment-name", default="brf_churn_final")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--docs-dir", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--threads", type=int, default=min(16, cpus))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reemplaza artefactos finales existentes; no modifica HPT ni splits.",
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

    values = set(frame[TARGET_COL].unique().tolist())
    if values != {0, 1}:
        raise ValueError(f"{label}: target debe contener exactamente 0 y 1; valores={values}")
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
        raise ValueError("Existe solapamiento entre TRAIN y VALIDATION")
    if train_ids & test_ids:
        raise ValueError("Existe solapamiento entre TRAIN y TEST")
    if validation_ids & test_ids:
        raise ValueError("Existe solapamiento entre VALIDATION y TEST")


def split_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    positives = int((frame[TARGET_COL] == 1).sum())
    negatives = int((frame[TARGET_COL] == 0).sum())
    return {
        "rows": int(len(frame)),
        "target_0": negatives,
        "target_1": positives,
        "positive_rate": float(positives / len(frame)),
    }


def prepare_features(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame[NUMERIC_FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURE_COLUMNS:
        values = pd.to_numeric(numeric[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)
        if values.isna().any():
            raise ValueError(
                f"La característica {column} contiene "
                f"{int(values.isna().sum())} valores no válidos"
            )
        numeric[column] = values.astype("float32")

    sexo = frame["SEXO"].astype("string").fillna("DESCONOCIDO")
    sexo = sexo.where(sexo.isin(SEX_CATEGORIES), "DESCONOCIDO")
    categorical = pd.Categorical(sexo, categories=SEX_CATEGORIES)
    one_hot = pd.get_dummies(categorical, prefix="SEXO", dtype="float32")
    one_hot = one_hot.reindex(
        columns=["SEXO_F", "SEXO_M", "SEXO_DESCONOCIDO"],
        fill_value=0.0,
    )

    matrix = pd.concat(
        [numeric.reset_index(drop=True), one_hot.reset_index(drop=True)],
        axis=1,
    )
    if matrix.columns.tolist() != MODEL_FEATURE_COLUMNS:
        raise RuntimeError("La matriz BRF no conserva el orden esperado de características")
    return np.ascontiguousarray(matrix.to_numpy(dtype=np.float32))


def candidate_label(value: Any) -> str:
    if value is None:
        return "No registrado"
    text = str(value).strip()
    if text.upper().startswith("C"):
        return text.upper()
    return f"C{int(value):04d}"


def load_hpt_artifacts(
    hpt_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], int, float]:
    params_path = hpt_dir / "best_hparams.json"
    metrics_path = hpt_dir / "best_metrics.json"
    if not params_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"Faltan artefactos HPT. Esperados: {params_path} y {metrics_path}"
        )

    with params_path.open(encoding="utf-8") as handle:
        raw_params = json.load(handle)
    with metrics_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    if summary.get("test_used_in_tuning") is not False:
        raise RuntimeError("El resumen HPT no confirma que TEST permaneció aislado")

    conceptual = summary.get("feature_columns_conceptual")
    if conceptual is not None and list(conceptual) != FEATURE_COLUMNS:
        raise RuntimeError("Las características conceptuales del HPT no coinciden")

    model_columns = summary.get("model_feature_columns")
    if model_columns is not None and list(model_columns) != MODEL_FEATURE_COLUMNS:
        raise RuntimeError("Las columnas efectivas del HPT no coinciden")

    if "n_estimators" not in raw_params:
        raise ValueError("best_hparams.json no contiene n_estimators")
    n_estimators = int(raw_params["n_estimators"])
    if n_estimators <= 0:
        raise ValueError("n_estimators debe ser positivo")

    validation_metrics = summary.get("best_metrics_validation", {})
    if "threshold_f1" not in validation_metrics:
        raise ValueError("best_metrics.json no contiene threshold_f1 de validación")
    threshold = float(validation_metrics["threshold_f1"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("El umbral congelado debe estar entre 0 y 1")

    params = dict(raw_params)
    params.pop("n_estimators", None)
    params.pop("n_model_features", None)
    params.pop("n_jobs", None)
    params.pop("verbose", None)
    params.pop("warm_start", None)
    return params, summary, n_estimators, threshold


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
    capture_at_10 = (
        float(top_positive_count / total_positives) if total_positives else 0.0
    )

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


def build_decile_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
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
    rows = [
        ("PR-AUC", metrics["pr_auc"]),
        ("ROC-AUC", metrics["roc_auc"]),
        ("Precisión", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1", metrics["f1"]),
        ("Lift@10", metrics["lift_at_10"]),
        ("Capture@10", metrics["capture_at_10"]),
        ("Tasa positiva TEST", metrics["positive_rate"]),
        ("Umbral congelado", metrics["threshold_frozen_from_validation"]),
    ]
    return pd.DataFrame(rows, columns=["Métrica", "Resultado"])


def hyperparameters_table(
    params: dict[str, Any],
    n_estimators: int,
    threshold: float,
    candidate_id: Any,
) -> pd.DataFrame:
    labels = {
        "criterion": "Criterio",
        "max_depth": "Profundidad máxima",
        "min_samples_split": "Mínimo de registros para dividir",
        "min_samples_leaf": "Mínimo de registros por hoja",
        "max_features": "Características evaluadas por división",
        "max_leaf_nodes": "Máximo de nodos hoja",
        "min_impurity_decrease": "Reducción mínima de impureza",
        "ccp_alpha": "Poda CCP alpha",
        "bootstrap": "Bootstrap externo",
        "sampling_strategy": "Estrategia de balanceo",
        "replacement": "Muestreo con reemplazo",
        "class_weight": "Peso de clases",
        "random_state": "Semilla",
    }
    rows: list[tuple[str, Any]] = [
        ("Candidato seleccionado", candidate_label(candidate_id))
    ]
    for key in labels:
        if key in params:
            rows.append((labels[key], params[key]))
    rows.extend(
        [
            ("Número final de árboles", n_estimators),
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


def copy_table_to_both(
    table: pd.DataFrame,
    name: str,
    output_dir: Path,
    docs_dir: Path,
) -> None:
    table.to_csv(output_dir / name, index=False)
    table.to_csv(docs_dir / name, index=False)


def plot_precision_recall(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    metrics: dict[str, Any],
    output: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig, axis = plt.subplots(figsize=(7.2, 5.8))
    axis.plot(
        recall,
        precision,
        label=f"Balanced RF (PR-AUC={metrics['pr_auc']:.4f})",
    )
    axis.axhline(
        metrics["positive_rate"],
        linestyle="--",
        label=f"Prevalencia TEST ({metrics['positive_rate']:.4f})",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precisión")
    axis.set_title("Curva Precision–Recall — Balanced Random Forest churn sobre TEST")
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
        label=f"Balanced RF (ROC-AUC={metrics['roc_auc']:.4f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", label="Clasificador aleatorio")
    axis.set_xlabel("Tasa de falsos positivos")
    axis.set_ylabel("Tasa de verdaderos positivos")
    axis.set_title("Curva ROC — Balanced Random Forest churn sobre TEST")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(confusion: pd.DataFrame, output: Path) -> None:
    values = confusion.to_numpy(dtype=float)
    row_totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(
        values,
        row_totals,
        out=np.zeros_like(values),
        where=row_totals > 0,
    )

    fig, axis = plt.subplots(figsize=(6.4, 5.5))
    image = axis.imshow(normalized, vmin=0, vmax=1)
    axis.set_xticks([0, 1], labels=["Predicho 0", "Predicho 1"])
    axis.set_yticks([0, 1], labels=["Real 0", "Real 1"])
    axis.set_xlabel("Clase predicha")
    axis.set_ylabel("Clase real")
    axis.set_title("Matriz de confusión BRF — umbral congelado")

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
    x = np.concatenate(
        [[0.0], deciles["cumulative_population_pct"].to_numpy(float)]
    )
    y = np.concatenate([[0.0], deciles["cumulative_recall"].to_numpy(float)])

    fig, axis = plt.subplots(figsize=(7.2, 5.8))
    axis.plot(x, y, marker="o", label="Balanced Random Forest")
    axis.plot([0, 1], [0, 1], linestyle="--", label="Selección aleatoria")
    axis.axvline(0.10, linestyle=":", label="Top 10 %")
    axis.set_xlabel("Proporción acumulada de clientes")
    axis.set_ylabel("Proporción acumulada de churners capturados")
    axis.set_title("Curva de ganancia acumulada — Balanced Random Forest churn")
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
    axis.hist(
        probabilities[y_true == 0],
        bins=50,
        density=True,
        alpha=0.55,
        label="No churn",
    )
    axis.hist(
        probabilities[y_true == 1],
        bins=50,
        density=True,
        alpha=0.55,
        label="Churn",
    )
    axis.axvline(threshold, linestyle="--", label=f"Umbral={threshold:.4f}")
    axis.set_xlabel("Puntuación estimada de churn")
    axis.set_ylabel("Densidad")
    axis.set_title("Distribución de puntuaciones BRF por clase — TEST")
    axis.grid(alpha=0.20)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame, output: Path) -> None:
    ordered = importance.sort_values("importance_pct", ascending=True)
    fig, axis = plt.subplots(figsize=(9.2, max(5.5, len(ordered) * 0.38)))
    axis.barh(ordered["feature"], ordered["importance_pct"] * 100)
    axis.set_xlabel("Importancia de impureza (%)")
    axis.set_title("Importancia de características del Balanced Random Forest final")
    axis.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_lgbm_comparison(
    lgbm_dir: Path,
    brf_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    metrics_path = lgbm_dir / "test_metrics.json"
    predictions_path = lgbm_dir / "test_predictions.parquet"
    if not metrics_path.exists() or not predictions_path.exists():
        return None

    with metrics_path.open(encoding="utf-8") as handle:
        lgbm_metrics = json.load(handle)
    lgbm_predictions = pd.read_parquet(
        predictions_path,
        columns=[ID_COL, TARGET_COL, "probability_churn"],
    )
    lgbm_predictions[ID_COL] = (
        lgbm_predictions[ID_COL].astype("string").str.strip()
    )
    lgbm_predictions = lgbm_predictions.rename(
        columns={"probability_churn": "probability_lgbm"}
    )

    brf_frame = brf_predictions[
        [ID_COL, TARGET_COL, "probability_churn"]
    ].rename(
        columns={
            TARGET_COL: "target_brf",
            "probability_churn": "probability_brf",
        }
    )
    lgbm_frame = lgbm_predictions.rename(columns={TARGET_COL: "target_lgbm"})
    merged = brf_frame.merge(
        lgbm_frame,
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(brf_predictions):
        raise RuntimeError(
            "Las predicciones LightGBM y BRF no contienen exactamente los mismos clientes"
        )
    if not np.array_equal(
        merged["target_brf"].to_numpy(),
        merged["target_lgbm"].to_numpy(),
    ):
        raise RuntimeError("Los targets de TEST no coinciden entre LightGBM y BRF")

    merged = merged.rename(columns={"target_brf": TARGET_COL}).drop(
        columns=["target_lgbm"]
    )
    return merged, lgbm_metrics


def comparison_table(
    brf_metrics: dict[str, Any],
    lgbm_metrics: dict[str, Any],
) -> pd.DataFrame:
    metrics = [
        ("PR-AUC", "pr_auc"),
        ("ROC-AUC", "roc_auc"),
        ("Precisión", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("Lift@10", "lift_at_10"),
        ("Capture@10", "capture_at_10"),
    ]
    rows = []
    for label, key in metrics:
        brf_value = float(brf_metrics[key])
        lgbm_value = float(lgbm_metrics[key])
        rows.append(
            {
                "Métrica": label,
                "LightGBM": lgbm_value,
                "Balanced Random Forest": brf_value,
                "Diferencia BRF-LGBM": brf_value - lgbm_value,
            }
        )
    return pd.DataFrame(rows)


def plot_pr_comparison(
    comparison: pd.DataFrame,
    brf_metrics: dict[str, Any],
    lgbm_metrics: dict[str, Any],
    output: Path,
) -> None:
    y_true = comparison[TARGET_COL].to_numpy(dtype=np.int8)
    precision_brf, recall_brf, _ = precision_recall_curve(
        y_true,
        comparison["probability_brf"].to_numpy(float),
    )
    precision_lgbm, recall_lgbm, _ = precision_recall_curve(
        y_true,
        comparison["probability_lgbm"].to_numpy(float),
    )

    fig, axis = plt.subplots(figsize=(7.4, 5.9))
    axis.plot(
        recall_lgbm,
        precision_lgbm,
        label=f"LightGBM (PR-AUC={float(lgbm_metrics['pr_auc']):.4f})",
    )
    axis.plot(
        recall_brf,
        precision_brf,
        label=f"Balanced RF (PR-AUC={float(brf_metrics['pr_auc']):.4f})",
    )
    axis.axhline(
        float(brf_metrics["positive_rate"]),
        linestyle="--",
        label=f"Prevalencia TEST ({float(brf_metrics['positive_rate']):.4f})",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precisión")
    axis.set_title("Comparación Precision–Recall — modelos de churn sobre TEST")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gain_comparison(
    comparison: pd.DataFrame,
    output: Path,
) -> None:
    y_true = comparison[TARGET_COL].to_numpy(dtype=np.int8)
    brf_deciles = build_decile_table(
        y_true,
        comparison["probability_brf"].to_numpy(float),
    )
    lgbm_deciles = build_decile_table(
        y_true,
        comparison["probability_lgbm"].to_numpy(float),
    )

    x_brf = np.concatenate(
        [[0.0], brf_deciles["cumulative_population_pct"].to_numpy(float)]
    )
    y_brf = np.concatenate(
        [[0.0], brf_deciles["cumulative_recall"].to_numpy(float)]
    )
    x_lgbm = np.concatenate(
        [[0.0], lgbm_deciles["cumulative_population_pct"].to_numpy(float)]
    )
    y_lgbm = np.concatenate(
        [[0.0], lgbm_deciles["cumulative_recall"].to_numpy(float)]
    )

    fig, axis = plt.subplots(figsize=(7.4, 5.9))
    axis.plot(x_lgbm, y_lgbm, marker="o", label="LightGBM")
    axis.plot(x_brf, y_brf, marker="o", label="Balanced Random Forest")
    axis.plot([0, 1], [0, 1], linestyle="--", label="Selección aleatoria")
    axis.axvline(0.10, linestyle=":", label="Top 10 %")
    axis.set_xlabel("Proporción acumulada de clientes")
    axis.set_ylabel("Proporción acumulada de churners capturados")
    axis.set_title("Comparación de ganancia acumulada — modelos de churn")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log = create_logger(args.log)
    total_started = time.time()
    stage_times: dict[str, float] = {}

    if args.threads <= 0:
        raise ValueError("--threads debe ser mayor que 0")
    cpus = available_cpus()
    if args.threads > cpus:
        raise ValueError(
            f"Solicitas {args.threads} threads, pero la reserva permite {cpus}"
        )

    splits_dir = args.splits_dir.expanduser().resolve()
    hpt_dir = args.hpt_dir.expanduser().resolve()
    lgbm_final_dir = args.lgbm_final_dir.expanduser().resolve()
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
            "No se volverá a evaluar TEST salvo que ejecutes explícitamente con --force."
        )
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    if args.force and docs_dir.exists():
        shutil.rmtree(docs_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    acquire_lock(f"final:churn:{args.experiment_name}")

    log("\n=== CONFIGURACIÓN FINAL BALANCED RANDOM FOREST CHURN ===")
    log(f"splits_dir: {splits_dir}")
    log(f"hpt_dir: {hpt_dir}")
    log(f"output_dir: {output_dir}")
    log(f"docs_dir: {docs_dir}")
    log(f"threads: {args.threads}/{cpus}")
    log(f"imbalanced-learn: {imblearn.__version__}")
    log(
        "protocolo: TRAIN + VALIDATION para entrenamiento final; "
        "TEST para evaluación única"
    )

    stage_started = time.time()
    log("\n[1/8] Cargando y verificando splits persistidos ...")
    train_df = validate_frame(pd.read_parquet(split_paths["train"]), "TRAIN")
    validation_df = validate_frame(
        pd.read_parquet(split_paths["validation"]),
        "VALIDATION",
    )
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
            f"target0={distribution['target_0']:,} | "
            f"target1={distribution['target_1']:,} | "
            f"churn={distribution['positive_rate'] * 100:.2f}%"
        )
    stage_times["load_and_validate_splits"] = round(
        time.time() - stage_started,
        3,
    )

    stage_started = time.time()
    log("\n[2/8] Cargando configuración congelada del HPT ...")
    params, hpt_summary, n_estimators, threshold = load_hpt_artifacts(hpt_dir)
    selected_candidate = hpt_summary.get("best_candidate_id")
    validation_metrics = hpt_summary.get("best_metrics_validation", {})

    log(f"   candidato: {candidate_label(selected_candidate)}")
    log(f"   n_estimators: {n_estimators}")
    log(f"   threshold F1 congelado: {threshold:.6f}")
    log(
        "   PR-AUC validation: "
        f"{float(validation_metrics.get('pr_auc', float('nan'))):.6f}"
    )
    log(
        "   ROC-AUC validation: "
        f"{float(validation_metrics.get('roc_auc', float('nan'))):.6f}"
    )
    log(
        "   F1 validation: "
        f"{float(validation_metrics.get('f1', float('nan'))):.6f}"
    )
    log(
        "   Lift@10 validation: "
        f"{float(validation_metrics.get('lift_at_10', float('nan'))):.4f}"
    )
    log("   hiperparámetros congelados:")
    for key, value in params.items():
        log(f"      {key}: {value}")
    log("   TEST no modifica hiperparámetros, árboles ni umbral")
    stage_times["load_hpt"] = round(time.time() - stage_started, 3)

    stage_started = time.time()
    log("\n[3/8] Construyendo conjunto final TRAIN + VALIDATION ...")
    development_df = pd.concat([train_df, validation_df], ignore_index=True)
    if development_df[ID_COL].duplicated().any():
        raise RuntimeError("TRAIN + VALIDATION contiene clientes duplicados")

    x_development = prepare_features(development_df)
    x_test = prepare_features(test_df)
    y_development = development_df[TARGET_COL].to_numpy(dtype=np.int8)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.int8)

    log(f"   entrenamiento final: {len(development_df):,} clientes (90%)")
    log(f"   evaluación final: {len(test_df):,} clientes (10%)")
    log(f"   características conceptuales: {len(FEATURE_COLUMNS)}")
    log(f"   columnas efectivas tras one-hot: {len(MODEL_FEATURE_COLUMNS)}")
    log("   escalado: no aplicado")
    log("   balanceo: interno en cada árbol; class_weight=None")
    stage_times["prepare_final_matrices"] = round(
        time.time() - stage_started,
        3,
    )

    stage_started = time.time()
    log("\n[4/8] Entrenando Balanced Random Forest final desde cero ...")
    final_model = BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=args.threads,
        verbose=0,
        **params,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        final_model.fit(x_development, y_development)
    training_seconds = time.time() - stage_started
    stage_times["final_training"] = round(training_seconds, 3)
    log(
        f"   entrenamiento completado con {n_estimators} árboles "
        f"en {training_seconds:.1f}s"
    )
    log("   no se aplicó early stopping ni se recalculó el balanceo")

    stage_started = time.time()
    log("\n[5/8] Generando predicciones y métricas sobre TEST ...")
    probabilities = final_model.predict_proba(x_test)[:, 1]
    test_metrics, predictions = evaluate_binary(y_test, probabilities, threshold)
    deciles = build_decile_table(y_test, probabilities)
    stage_times["test_prediction_and_metrics"] = round(
        time.time() - stage_started,
        3,
    )

    log(f"   PR-AUC: {test_metrics['pr_auc']:.6f}")
    log(f"   ROC-AUC: {test_metrics['roc_auc']:.6f}")
    log(f"   Precisión: {test_metrics['precision']:.6f}")
    log(f"   Recall: {test_metrics['recall']:.6f}")
    log(f"   F1: {test_metrics['f1']:.6f}")
    log(f"   Lift@10: {test_metrics['lift_at_10']:.4f}")
    log(f"   Capture@10: {test_metrics['capture_at_10']:.4f}")
    log(
        f"   Top 10%: {test_metrics['top10_positives']:,} churners en "
        f"{test_metrics['top10_rows']:,} clientes | "
        f"tasa={test_metrics['top10_positive_rate'] * 100:.2f}%"
    )
    log(
        f"   Matriz: TN={test_metrics['true_negative']:,} | "
        f"FP={test_metrics['false_positive']:,} | "
        f"FN={test_metrics['false_negative']:,} | "
        f"TP={test_metrics['true_positive']:,}"
    )

    stage_started = time.time()
    log("\n[6/8] Guardando modelo, tablas, predicciones y gráficas ...")
    joblib.dump(final_model, output_dir / "model.joblib", compress=3)

    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURE_COLUMNS,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    total_importance = float(importance["importance"].sum())
    importance["importance_pct"] = (
        importance["importance"] / total_importance if total_importance else 0.0
    )
    importance["importance_cumulative_pct"] = (
        importance["importance_pct"].cumsum()
    )

    main_metrics_table = metrics_table(test_metrics)
    final_hparams_table = hyperparameters_table(
        params,
        n_estimators,
        threshold,
        selected_candidate,
    )
    confusion = save_confusion_table(
        test_metrics,
        output_dir / "confusion_matrix.csv",
    )
    confusion.to_csv(docs_dir / "confusion_matrix.csv")

    copy_table_to_both(
        main_metrics_table,
        "test_metrics_table.csv",
        output_dir,
        docs_dir,
    )
    copy_table_to_both(
        final_hparams_table,
        "final_hyperparameters_table.csv",
        output_dir,
        docs_dir,
    )
    copy_table_to_both(
        importance,
        "feature_importance.csv",
        output_dir,
        docs_dir,
    )
    copy_table_to_both(
        deciles,
        "score_deciles.csv",
        output_dir,
        docs_dir,
    )

    prediction_frame = test_df[[ID_COL, TARGET_COL]].copy().reset_index(drop=True)
    prediction_frame["probability_churn"] = probabilities
    prediction_frame["prediction_churn"] = predictions
    risk_rank = (
        pd.Series(probabilities)
        .rank(method="first", ascending=False)
        .astype(np.int64)
        .to_numpy()
    )
    prediction_frame["risk_rank"] = risk_rank
    prediction_frame["risk_decile"] = np.minimum(
        10,
        ((risk_rank - 1) * 10 // len(prediction_frame)) + 1,
    ).astype(np.int64)
    prediction_frame.to_parquet(
        output_dir / "test_predictions.parquet",
        index=False,
    )

    plot_precision_recall(
        y_test,
        probabilities,
        test_metrics,
        docs_dir / "precision_recall_test.png",
    )
    plot_roc(
        y_test,
        probabilities,
        test_metrics,
        docs_dir / "roc_test.png",
    )
    plot_confusion_matrix(
        confusion,
        docs_dir / "confusion_matrix_test.png",
    )
    plot_cumulative_gain(
        deciles,
        docs_dir / "cumulative_gain_test.png",
    )
    plot_score_distribution(
        y_test,
        probabilities,
        threshold,
        docs_dir / "score_distribution_test.png",
    )
    plot_feature_importance(
        importance,
        docs_dir / "feature_importance_final.png",
    )
    stage_times["save_artifacts"] = round(time.time() - stage_started, 3)

    stage_started = time.time()
    log("\n[7/8] Comparando con LightGBM final, cuando está disponible ...")
    comparison_summary: dict[str, Any] | None = None
    comparison_result = load_lgbm_comparison(
        lgbm_final_dir,
        prediction_frame,
    )
    if comparison_result is None:
        log(
            f"   comparación omitida: faltan métricas o predicciones en "
            f"{lgbm_final_dir}"
        )
    else:
        comparison_predictions, lgbm_metrics = comparison_result
        comparison = comparison_table(test_metrics, lgbm_metrics)
        copy_table_to_both(
            comparison,
            "comparison_lgbm_brf_test.csv",
            output_dir,
            docs_dir,
        )
        plot_pr_comparison(
            comparison_predictions,
            test_metrics,
            lgbm_metrics,
            docs_dir / "precision_recall_comparison_lgbm_brf.png",
        )
        plot_gain_comparison(
            comparison_predictions,
            docs_dir / "cumulative_gain_comparison_lgbm_brf.png",
        )

        log("   comparación sobre el mismo TEST:")
        for row in comparison.itertuples(index=False):
            log(
                f"      {row[0]} | LightGBM={float(row[1]):.6f} | "
                f"BRF={float(row[2]):.6f} | "
                f"BRF-LGBM={float(row[3]):+.6f}"
            )
        comparison_summary = {
            "lgbm_final_dir": str(lgbm_final_dir),
            "metrics": comparison.to_dict(orient="records"),
            "n_matched_test_clients": int(len(comparison_predictions)),
            "precision_recall_plot": str(
                docs_dir / "precision_recall_comparison_lgbm_brf.png"
            ),
            "cumulative_gain_plot": str(
                docs_dir / "cumulative_gain_comparison_lgbm_brf.png"
            ),
        }
    stage_times["comparison_with_lgbm"] = round(
        time.time() - stage_started,
        3,
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": args.experiment_name,
        "model": "BalancedRandomForestClassifier",
        "problem": "churn_cliente",
        "training_protocol": (
            "fit_train_plus_validation_90pct_then_single_test_10pct_evaluation"
        ),
        "splits_dir": str(splits_dir),
        "split_summary": split_summary,
        "hpt_dir": str(hpt_dir),
        "hpt_best_candidate_id": selected_candidate,
        "hpt_validation_metrics": validation_metrics,
        "hyperparameters": {
            **params,
            "n_estimators": n_estimators,
            "n_jobs_runtime": args.threads,
        },
        "frozen_threshold": threshold,
        "threshold_source": "best_validation_trial_threshold_f1",
        "feature_columns_conceptual": FEATURE_COLUMNS,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "categorical_encoding": {
            "SEXO": "one_hot_fixed_categories",
            "categories": SEX_CATEGORIES,
        },
        "imbalance_treatment": {
            "method": "BalancedRandomForest internal sampling per tree",
            "sampling_strategy": params.get("sampling_strategy"),
            "replacement": params.get("replacement"),
            "bootstrap": params.get("bootstrap"),
            "class_weight": params.get("class_weight"),
            "external_sampling": False,
        },
        "n_train_original": int(len(train_df)),
        "n_validation_original": int(len(validation_df)),
        "n_final_train_development": int(len(development_df)),
        "n_test": int(len(test_df)),
        "final_train_positive_rate": float(y_development.mean()),
        "test_positive_rate": float(y_test.mean()),
        "test_metrics": test_metrics,
        "test_used_for_hyperparameters": False,
        "test_used_for_n_estimators": False,
        "test_used_for_threshold": False,
        "test_evaluated_after_final_fit": True,
        "threads": args.threads,
        "imbalanced_learn_version": imblearn.__version__,
        "stage_times_seconds": stage_times,
        "duration_seconds": round(time.time() - total_started, 2),
        "comparison_with_lgbm": comparison_summary,
        "artifacts": {
            "model": str(output_dir / "model.joblib"),
            "predictions": str(output_dir / "test_predictions.parquet"),
            "metrics_table": str(docs_dir / "test_metrics_table.csv"),
            "hyperparameters_table": str(
                docs_dir / "final_hyperparameters_table.csv"
            ),
            "confusion_matrix": str(docs_dir / "confusion_matrix.csv"),
            "deciles": str(docs_dir / "score_deciles.csv"),
            "precision_recall_plot": str(
                docs_dir / "precision_recall_test.png"
            ),
            "roc_plot": str(docs_dir / "roc_test.png"),
            "confusion_plot": str(
                docs_dir / "confusion_matrix_test.png"
            ),
            "cumulative_gain_plot": str(
                docs_dir / "cumulative_gain_test.png"
            ),
            "score_distribution_plot": str(
                docs_dir / "score_distribution_test.png"
            ),
            "feature_importance_plot": str(
                docs_dir / "feature_importance_final.png"
            ),
        },
    }

    save_json(output_dir / "run_summary.json", summary)
    save_json(docs_dir / "run_summary.json", summary)
    save_json(output_dir / "test_metrics.json", test_metrics)
    save_json(docs_dir / "test_metrics.json", test_metrics)

    GLOBAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with GLOBAL_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    log("\n[8/8] Experimento Balanced Random Forest finalizado")
    log("[BRF-FINAL] ================================================================")
    log(
        f"[BRF-FINAL] PR-AUC={test_metrics['pr_auc']:.6f} | "
        f"ROC-AUC={test_metrics['roc_auc']:.6f} | "
        f"F1={test_metrics['f1']:.6f}"
    )
    log(
        f"[BRF-FINAL] Precision={test_metrics['precision']:.6f} | "
        f"Recall={test_metrics['recall']:.6f} | "
        f"Lift@10={test_metrics['lift_at_10']:.4f} | "
        f"Capture@10={test_metrics['capture_at_10']:.4f}"
    )
    log(
        f"[BRF-FINAL] Desarrollo={len(development_df):,} | "
        f"TEST={len(test_df):,} | árboles={n_estimators} | "
        f"threshold={threshold:.6f}"
    )
    log(f"[BRF-FINAL] Modelo: {output_dir / 'model.joblib'}")
    log(f"[BRF-FINAL] Resultados para tesis: {docs_dir}")
    log("[BRF-FINAL] TEST no modificó ninguna decisión del modelo.")
    log(f"[BRF-FINAL] Tiempo total: {(time.time() - total_started) / 60:.2f} minutos")
    log("[BRF-FINAL] ================================================================")


if __name__ == "__main__":
    main()
