"""Optimización de hiperparámetros de Balanced Random Forest para churn.

Este experimento utiliza exactamente los mismos splits persistidos por el tuning de
LightGBM:

    data/churn/splits/split_80_10_10_seed_42/
        train.parquet
        validation.parquet
        test.parquet
        split_summary.json

Protocolo:
- Las 15 características conceptuales de churn se conservan.
- SEXO se codifica mediante one-hot sin introducir orden artificial.
- TRAIN se utiliza para ajustar cada candidato.
- VALIDATION se utiliza para PR-AUC, selección de hiperparámetros y umbral F1.
- TEST solo se verifica por identidad/tamaño y nunca se usa para entrenar, seleccionar
  candidatos, elegir umbral ni calcular métricas durante el HPT.
- El desbalance se trata internamente con BalancedRandomForestClassifier mediante
  submuestreo balanceado por árbol. No se combina con class_weight, SMOTE ni otro
  remuestreo externo.
- La búsqueda reutiliza la estructura de Successive Halving del experimento LightGBM:
  subconjuntos estratificados anidados, paralelización de candidatos, persistencia por
  ronda y reanudación de ejecuciones interrumpidas.

Ejemplo para una reserva de 128 CPU:

    python app/churn/09_hptuning_brf.py \
        --workers 8 \
        --threads-per-worker 16

Para reiniciar únicamente los resultados del HPT BRF:

    python app/churn/09_hptuning_brf.py \
        --workers 8 \
        --threads-per-worker 16 \
        --force
"""
from __future__ import annotations

import argparse
import atexit
import fcntl
import gc
import json
import math
import multiprocessing as mp
import os
import shutil
import socket
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
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
import sklearn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import imblearn
    from imblearn.ensemble import BalancedRandomForestClassifier
except ImportError as exc:  # pragma: no cover - mensaje operativo para el servidor
    raise SystemExit(
        "Falta imbalanced-learn. Instálelo en el entorno activo con: "
        "python -m pip install imbalanced-learn"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = (
    REPO_ROOT / "data" / "churn" / "splits" / "split_80_10_10_seed_42"
)
DEFAULT_LOG = REPO_ROOT / "data" / "logs" / "09_hptuning_brf_churn.log"
GLOBAL_LOG_FILE = REPO_ROOT / "data" / "logs" / "churn_brf_hpt_runs.jsonl"
LOCK_FILE = REPO_ROOT / "data" / "locks" / "churn_brf.lock"

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

_XTR: np.ndarray | None = None
_YTR: np.ndarray | None = None
_XVA: np.ndarray | None = None
_YVA: np.ndarray | None = None


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def parse_csv(text: str, cast: Callable) -> list:
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


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
            "\n=== 09_hptuning_brf.py | "
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
        raise RuntimeError(f"Ya existe una ejecución BRF churn activa: {owner}") from exc

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
    default_threads = min(16, cpus)
    default_workers = max(1, min(8, cpus // default_threads))

    parser = argparse.ArgumentParser(
        description=(
            "Balanced Random Forest churn: Successive Halving por PR-AUC usando los "
            "mismos splits 80/10/10 del experimento LightGBM"
        )
    )
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--experiment-name", default="brf_churn_main")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--docs-dir", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--candidates", "--trials", dest="candidates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--threads-per-worker", type=int, default=default_threads)
    parser.add_argument("--round-fractions", default="0.20,0.40,0.70,1.0")
    parser.add_argument("--round-trees", default="64,160,400,800")
    parser.add_argument("--survivors", default="100,25,8,1")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinicia artefactos del HPT BRF; nunca elimina ni modifica los splits.",
    )
    return parser.parse_args()


def validate_frame(frame: pd.DataFrame, label: str, require_features: bool = True) -> pd.DataFrame:
    required = [ID_COL, TARGET_COL] + (FEATURE_COLUMNS if require_features else [])
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


def validate_disjoint_ids(
    train_ids: pd.Series,
    validation_ids: pd.Series,
    test_ids: pd.Series,
) -> None:
    train_set = set(train_ids.astype(str))
    validation_set = set(validation_ids.astype(str))
    test_set = set(test_ids.astype(str))
    if train_set & validation_set:
        raise ValueError("Existe solapamiento entre TRAIN y VALIDATION")
    if train_set & test_set:
        raise ValueError("Existe solapamiento entre TRAIN y TEST")
    if validation_set & test_set:
        raise ValueError("Existe solapamiento entre VALIDATION y TEST")


def split_distribution(frame: pd.DataFrame) -> dict[str, float | int]:
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
                f"La característica {column} contiene {int(values.isna().sum())} valores no válidos"
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


def nested_order(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    rng.shuffle(positive)
    rng.shuffle(negative)

    order = np.empty(len(y), dtype=np.int64)
    positive_position = 0
    negative_position = 0
    positive_rate = len(positive) / len(y)

    for index in range(len(y)):
        need_positive = (
            positive_position < len(positive)
            and (
                negative_position >= len(negative)
                or positive_position / max(1, index) < positive_rate
            )
        )
        if need_positive:
            order[index] = positive[positive_position]
            positive_position += 1
        else:
            order[index] = negative[negative_position]
            negative_position += 1
    return order


def subset_from_order(
    x: np.ndarray,
    y: np.ndarray,
    order: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if fraction >= 1.0:
        return x, y
    size = max(2, int(round(len(order) * fraction)))
    indices = order[:size]
    return (
        np.ascontiguousarray(x[indices], dtype=np.float32),
        np.ascontiguousarray(y[indices], dtype=np.int8),
    )


def make_candidates(count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = []

    depth_values: list[int | None] = [None, 6, 8, 10, 12, 16, 20, 30]
    split_values = [2, 5, 10, 20, 40, 80]
    leaf_values = [1, 2, 5, 10, 20, 40]
    feature_values: list[str | float] = ["sqrt", "log2", 0.35, 0.50, 0.70, 1.0]
    leaf_node_values: list[int | None] = [None, 31, 63, 127, 255]
    impurity_values = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    pruning_values = [0.0, 1e-6, 1e-5, 1e-4]

    for candidate_id in range(count):
        params = {
            "criterion": str(rng.choice(["gini", "entropy"])),
            "max_depth": depth_values[int(rng.integers(0, len(depth_values)))],
            "min_samples_split": int(rng.choice(split_values)),
            "min_samples_leaf": int(rng.choice(leaf_values)),
            "max_features": feature_values[int(rng.integers(0, len(feature_values)))],
            "max_leaf_nodes": leaf_node_values[
                int(rng.integers(0, len(leaf_node_values)))
            ],
            "min_impurity_decrease": float(rng.choice(impurity_values)),
            "ccp_alpha": float(rng.choice(pruning_values)),
            "bootstrap": False,
            "sampling_strategy": "all",
            "replacement": bool(rng.choice([False, True])),
            "class_weight": None,
            "random_state": seed + candidate_id,
        }
        candidates.append({"candidate_id": candidate_id, "params": params})
    return candidates


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    pr_auc = float(average_precision_score(y_true, probabilities))
    roc_auc = float(roc_auc_score(y_true, probabilities))
    precision_curve, recall_curve, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if len(thresholds):
        denominator = precision_curve[:-1] + recall_curve[:-1]
        f1_values = np.divide(
            2 * precision_curve[:-1] * recall_curve[:-1],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        threshold = float(thresholds[int(np.argmax(f1_values))])
    else:
        threshold = 0.5

    predictions = (probabilities >= threshold).astype(np.int8)
    base_rate = float(y_true.mean())
    top_count = max(1, int(math.ceil(len(probabilities) * 0.10)))
    top_indices = np.argpartition(probabilities, -top_count)[-top_count:]
    top_rate = float(y_true[top_indices].mean())

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "lift_at_10": float(top_rate / base_rate) if base_rate else 0.0,
        "threshold_f1": threshold,
        "base_rate": base_rate,
        "top10_positive_rate": top_rate,
    }


def build_model(
    params: dict[str, Any],
    n_estimators: int,
    threads: int,
) -> BalancedRandomForestClassifier:
    return BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=threads,
        verbose=0,
        **params,
    )


def evaluate(task: tuple[int, dict[str, Any], int, int]) -> dict[str, Any]:
    candidate_id, params, n_estimators, threads = task
    if any(value is None for value in (_XTR, _YTR, _XVA, _YVA)):
        raise RuntimeError("Datos del worker no inicializados")

    started = time.time()
    model = build_model(params, n_estimators, threads)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        model.fit(_XTR, _YTR)
    probabilities = model.predict_proba(_XVA)[:, 1]
    result = metrics(_YVA, probabilities)

    return {
        "candidate_id": candidate_id,
        "n_estimators": int(n_estimators),
        "duration_seconds": round(time.time() - started, 3),
        **{key: float(value) for key, value in result.items()},
        "params_json": json.dumps(params, sort_keys=True),
    }


def save_records(records: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(records).sort_values("candidate_id")
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return frame


def plot_halving(all_results: pd.DataFrame, output: Path) -> None:
    if all_results.empty:
        return

    rounds = sorted(all_results["round"].astype(int).unique())
    values = [
        all_results.loc[
            all_results["round"].astype(int) == round_id,
            "pr_auc",
        ].to_numpy()
        for round_id in rounds
    ]
    best = [float(np.max(value)) for value in values]

    fig, axis = plt.subplots(figsize=(9, 5.5))
    labels = [f"R{round_id}" for round_id in rounds]
    try:
        axis.boxplot(values, tick_labels=labels, showfliers=False)
    except TypeError:
        axis.boxplot(values, labels=labels, showfliers=False)
    axis.plot(range(1, len(rounds) + 1), best, marker="o", label="Mejor PR-AUC")
    axis.set_xlabel("Ronda de Successive Halving")
    axis.set_ylabel("PR-AUC de validación")
    axis.set_title("Balanced Random Forest — evolución del Successive Halving")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top_candidates(final_round: pd.DataFrame, output: Path) -> None:
    selected = final_round.sort_values("pr_auc", ascending=False).head(8).copy()
    selected["candidate"] = selected["candidate_id"].astype(int).map(lambda value: f"C{value:04d}")
    selected = selected.sort_values("pr_auc")

    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.barh(selected["candidate"], selected["pr_auc"])
    axis.set_xlabel("PR-AUC de validación")
    axis.set_title("Balanced Random Forest — candidatos finalistas")
    axis.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log = create_logger(args.log)
    started = time.time()

    fractions = parse_csv(args.round_fractions, float)
    tree_budgets = parse_csv(args.round_trees, int)
    survivors = parse_csv(args.survivors, int)

    if args.candidates <= 0:
        raise ValueError("--candidates debe ser mayor que 0")
    if args.workers <= 0 or args.threads_per_worker <= 0:
        raise ValueError("--workers y --threads-per-worker deben ser mayores que 0")
    if not (
        len(fractions) == len(tree_budgets) == len(survivors)
        and fractions == sorted(fractions)
        and fractions[-1] == 1.0
        and survivors[-1] == 1
    ):
        raise ValueError("Configuración de Successive Halving inválida")
    if any(value <= 0 for value in tree_budgets):
        raise ValueError("Todos los presupuestos de árboles deben ser positivos")

    active_count = args.candidates
    for keep in survivors:
        if keep <= 0 or keep > active_count:
            raise ValueError("Secuencia --survivors incompatible con --candidates")
        active_count = keep

    cpus = available_cpus()
    requested_threads = args.workers * args.threads_per_worker
    if requested_threads > cpus:
        raise ValueError(
            f"Solicitas {requested_threads} hilos "
            f"({args.workers}x{args.threads_per_worker}), pero la reserva permite {cpus}"
        )

    splits_dir = args.splits_dir.expanduser().resolve()
    split_paths = {
        "train": splits_dir / "train.parquet",
        "validation": splits_dir / "validation.parquet",
        "test": splits_dir / "test.parquet",
        "summary": splits_dir / "split_summary.json",
    }
    for name, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Falta el artefacto de split {name}: {path}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT / "data" / "models" / "churn" / "hpt" / args.experiment_name
    )
    docs_dir = (
        args.docs_dir.expanduser().resolve()
        if args.docs_dir is not None
        else REPO_ROOT / "docs" / "hpt" / "churn" / args.experiment_name
    )

    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    if args.force and docs_dir.exists():
        shutil.rmtree(docs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    acquire_lock(f"hpt:churn:{args.experiment_name}")

    log("\n=== CONFIGURACIÓN HPT BALANCED RANDOM FOREST ===")
    log(f"splits_dir: {splits_dir}")
    log(f"experiment_name: {args.experiment_name}")
    log(f"output_dir: {output_dir}")
    log(f"docs_dir: {docs_dir}")
    log(f"candidatos: {args.candidates}")
    log(f"round_fractions: {fractions}")
    log(f"round_trees: {tree_budgets}")
    log(f"survivors: {survivors}")
    log(f"CPU: workers={args.workers} x threads={args.threads_per_worker} = {requested_threads}/{cpus}")
    log(f"scikit-learn: {sklearn.__version__}")
    log(f"imbalanced-learn: {imblearn.__version__}")

    log("\n[1/7] Cargando los mismos splits usados por LightGBM ...")
    stage_started = time.time()
    train_df = validate_frame(pd.read_parquet(split_paths["train"]), "TRAIN")
    validation_df = validate_frame(
        pd.read_parquet(split_paths["validation"]),
        "VALIDATION",
    )
    test_identity = validate_frame(
        pd.read_parquet(split_paths["test"], columns=[ID_COL, TARGET_COL]),
        "TEST",
        require_features=False,
    )
    validate_disjoint_ids(
        train_df[ID_COL],
        validation_df[ID_COL],
        test_identity[ID_COL],
    )
    with split_paths["summary"].open(encoding="utf-8") as handle:
        split_summary = json.load(handle)

    for name, frame in (
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_identity),
    ):
        distribution = split_distribution(frame)
        log(
            f"   {name:<10}: filas={distribution['rows']:,} | "
            f"target0={distribution['target_0']:,} | target1={distribution['target_1']:,} | "
            f"churn={distribution['positive_rate'] * 100:.2f}%"
        )
    log("   TEST se verificó solo por ID/target y no se cargaron sus características")
    log(f"   tiempo: {time.time() - stage_started:.1f}s")

    log("\n[2/7] Preparando matrices numéricas ...")
    stage_started = time.time()
    x_train_full = prepare_features(train_df)
    x_validation_full = prepare_features(validation_df)
    y_train_full = train_df[TARGET_COL].to_numpy(dtype=np.int8)
    y_validation_full = validation_df[TARGET_COL].to_numpy(dtype=np.int8)
    log(f"   características conceptuales: {len(FEATURE_COLUMNS)}")
    log(f"   columnas del modelo tras one-hot de SEXO: {len(MODEL_FEATURE_COLUMNS)}")
    log(f"   columnas: {MODEL_FEATURE_COLUMNS}")
    log("   no se aplica escalado: los árboles no lo requieren")
    log("   balanceo: interno por árbol mediante BalancedRandomForestClassifier")
    log("   class_weight: None; no se duplica la corrección del desbalance")
    log(f"   tiempo: {time.time() - stage_started:.1f}s")

    del test_identity
    gc.collect()

    train_order = nested_order(y_train_full, args.seed + 100)
    validation_order = nested_order(y_validation_full, args.seed + 200)
    candidates = make_candidates(args.candidates, args.seed)
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}

    definition = {
        "schema_version": 1,
        "model": "BalancedRandomForestClassifier",
        "search_method": "parallel_successive_halving_random_nested",
        "experiment_name": args.experiment_name,
        "splits_dir": str(splits_dir),
        "split_summary": split_summary,
        "feature_columns_conceptual": FEATURE_COLUMNS,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "categorical_encoding": {
            "SEXO": "one_hot_fixed_categories",
            "categories": SEX_CATEGORIES,
        },
        "imbalance_treatment": {
            "method": "internal_balanced_bootstrap_per_tree",
            "sampling_strategy": "all",
            "class_weight": None,
            "external_sampling": False,
        },
        "candidates": args.candidates,
        "seed": args.seed,
        "round_fractions": fractions,
        "round_trees": tree_budgets,
        "survivors": survivors,
        "selection_metric": "PR-AUC validation",
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "test_used_in_tuning": False,
        "sklearn_version": sklearn.__version__,
        "imblearn_version": imblearn.__version__,
    }

    definition_path = output_dir / "run_definition.json"
    if definition_path.exists() and not args.force:
        with definition_path.open(encoding="utf-8") as handle:
            previous_definition = json.load(handle)
        ignored = {"workers", "threads_per_worker"}
        previous_comparable = {
            key: value for key, value in previous_definition.items() if key not in ignored
        }
        current_comparable = {
            key: value for key, value in definition.items() if key not in ignored
        }
        if previous_comparable != current_comparable:
            raise RuntimeError(
                "El experimento BRF existente tiene una configuración incompatible. "
                "Ejecute con --force para reiniciar únicamente el tuning BRF."
            )
    save_json(definition_path, definition)

    log("\n[3/7] Iniciando Successive Halving ...")
    log("   criterio único de supervivencia: PR-AUC de VALIDATION")
    log("   TEST permanece completamente fuera del proceso de selección")

    active = list(candidates_by_id)
    all_round_frames: list[pd.DataFrame] = []
    global _XTR, _YTR, _XVA, _YVA

    for round_id, (fraction, tree_count, keep) in enumerate(
        zip(fractions, tree_budgets, survivors),
        start=1,
    ):
        result_path = output_dir / f"round_{round_id}_results.csv"
        existing_frame = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
        records = existing_frame.to_dict("records") if not existing_frame.empty else []
        completed = (
            set(existing_frame["candidate_id"].astype(int))
            if not existing_frame.empty
            else set()
        )
        pending = [candidate_id for candidate_id in active if candidate_id not in completed]

        _XTR, _YTR = subset_from_order(
            x_train_full,
            y_train_full,
            train_order,
            fraction,
        )
        _XVA, _YVA = subset_from_order(
            x_validation_full,
            y_validation_full,
            validation_order,
            fraction,
        )

        log(
            f"\n[BRF-R{round_id}] candidatos={len(active)} | pendientes={len(pending)} | "
            f"datos={fraction:.0%} | train={len(_YTR):,} | val={len(_YVA):,} | "
            f"árboles={tree_count} | sobreviven={min(keep, len(active))}"
        )

        round_started = time.time()
        if pending:
            tasks = [
                (
                    candidate_id,
                    candidates_by_id[candidate_id]["params"],
                    tree_count,
                    args.threads_per_worker,
                )
                for candidate_id in pending
            ]
            completed_in_run = 0
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=mp.get_context("fork"),
            ) as executor:
                futures = {
                    executor.submit(evaluate, task): task[0]
                    for task in tasks
                }
                for future in as_completed(futures):
                    record = future.result()
                    record.update(
                        {
                            "round": round_id,
                            "data_fraction": fraction,
                        }
                    )
                    records.append(record)
                    completed_in_run += 1
                    save_records(records, result_path)

                    elapsed = time.time() - round_started
                    eta = (
                        elapsed / max(1, completed_in_run)
                        * (len(pending) - completed_in_run)
                    )
                    params = candidates_by_id[int(record["candidate_id"])]["params"]
                    log(
                        f"[BRF-R{round_id} {completed_in_run:03d}/{len(pending):03d}] "
                        f"C{int(record['candidate_id']):04d} | "
                        f"PR-AUC={record['pr_auc']:.6f} | "
                        f"F1={record['f1']:.6f} | "
                        f"Lift@10={record['lift_at_10']:.4f} | "
                        f"depth={params['max_depth']} | leaf={params['min_samples_leaf']} | "
                        f"{record['duration_seconds']:.1f}s | ETA={eta / 60:.1f}m"
                    )

        frame = save_records(records, result_path)
        frame = (
            frame[frame["candidate_id"].astype(int).isin(active)]
            .copy()
            .sort_values(["pr_auc", "candidate_id"], ascending=[False, True])
        )
        if frame.empty:
            raise RuntimeError(f"La ronda {round_id} no produjo resultados")

        active = frame.head(min(keep, len(frame)))["candidate_id"].astype(int).tolist()
        all_round_frames.append(frame)
        round_duration = time.time() - round_started
        best_row = frame.iloc[0]
        save_json(
            output_dir / f"round_{round_id}_survivors.json",
            {
                "round": round_id,
                "survivors": active,
                "best_candidate": int(active[0]),
                "best_pr_auc": float(best_row["pr_auc"]),
                "duration_seconds": round_duration,
            },
        )
        log(
            f"[BRF-R{round_id}] mejor=C{active[0]:04d} | "
            f"PR-AUC={float(best_row['pr_auc']):.6f} | "
            f"pasan={len(active)} | tiempo={round_duration / 60:.1f}m"
        )

        _XTR = _YTR = _XVA = _YVA = None
        gc.collect()

    log("\n[4/7] Confirmando candidato ganador sobre TRAIN y VALIDATION completos ...")
    winner = active[0]
    final_params = dict(candidates_by_id[winner]["params"])
    final_tree_count = tree_budgets[-1]
    final_model = build_model(final_params, final_tree_count, args.threads_per_worker)
    confirmation_started = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        final_model.fit(x_train_full, y_train_full)
    validation_probabilities = final_model.predict_proba(x_validation_full)[:, 1]
    best_metrics = metrics(y_validation_full, validation_probabilities)
    confirmation_duration = time.time() - confirmation_started

    final_hparams = {
        **final_params,
        "n_estimators": final_tree_count,
        "n_model_features": len(MODEL_FEATURE_COLUMNS),
    }
    save_json(output_dir / "best_hparams.json", final_hparams)
    joblib.dump(final_model, output_dir / "best_model_validation.joblib", compress=3)

    log(f"   ganador: C{winner:04d}")
    log(f"   n_estimators: {final_tree_count}")
    log(f"   PR-AUC validation: {best_metrics['pr_auc']:.6f}")
    log(f"   ROC-AUC validation: {best_metrics['roc_auc']:.6f}")
    log(f"   Precision validation: {best_metrics['precision']:.6f}")
    log(f"   Recall validation: {best_metrics['recall']:.6f}")
    log(f"   F1 validation: {best_metrics['f1']:.6f}")
    log(f"   Lift@10 validation: {best_metrics['lift_at_10']:.4f}")
    log(f"   threshold F1 validation: {best_metrics['threshold_f1']:.6f}")
    log(f"   tiempo confirmación: {confirmation_duration:.1f}s")

    log("\n[5/7] Consolidando resultados de las rondas ...")
    all_results = pd.concat(all_round_frames, ignore_index=True)
    all_results.to_csv(output_dir / "all_round_results.csv", index=False)
    final_round = all_round_frames[-1].sort_values("pr_auc", ascending=False)
    final_round.head(10).to_csv(output_dir / "top10_trials.csv", index=False)

    importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURE_COLUMNS,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance["importance_pct"] = (
        importance["importance"] / importance["importance"].sum()
        if importance["importance"].sum()
        else 0.0
    )
    importance["importance_cumulative_pct"] = importance["importance_pct"].cumsum()
    importance.to_csv(output_dir / "feature_importance_validation.csv", index=False)

    log("\n[6/7] Generando resumen y gráficas ...")
    halving_plot = docs_dir / "successive_halving_pr_auc.png"
    finalists_plot = docs_dir / "finalists_pr_auc.png"
    plot_halving(all_results, halving_plot)
    plot_top_candidates(final_round, finalists_plot)
    importance.to_csv(docs_dir / "feature_importance_validation.csv", index=False)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": args.experiment_name,
        "model": "BalancedRandomForestClassifier",
        "problem": "churn_cliente",
        "search_method": "parallel_successive_halving_random_nested",
        "objective": "maximize_pr_auc_validation",
        "splits_dir": str(splits_dir),
        "split_summary": split_summary,
        "feature_columns_conceptual": FEATURE_COLUMNS,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "n_features_conceptual": len(FEATURE_COLUMNS),
        "n_model_columns": len(MODEL_FEATURE_COLUMNS),
        "categorical_encoding": {
            "SEXO": "one_hot_fixed_categories",
            "categories": SEX_CATEGORIES,
        },
        "imbalance_treatment": {
            "method": "BalancedRandomForest internal sampling per tree",
            "sampling_strategy": final_params["sampling_strategy"],
            "replacement": final_params["replacement"],
            "bootstrap": final_params["bootstrap"],
            "class_weight": None,
            "external_sampling": False,
        },
        "candidate_count": args.candidates,
        "best_candidate_id": winner,
        "best_params": final_hparams,
        "best_metrics_validation": best_metrics,
        "round_fractions": fractions,
        "round_trees": tree_budgets,
        "survivors": survivors,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "test_used_in_tuning": False,
        "test_features_loaded": False,
        "sklearn_version": sklearn.__version__,
        "imblearn_version": imblearn.__version__,
        "plots": {
            "successive_halving": str(halving_plot),
            "finalists": str(finalists_plot),
        },
        "duration_seconds": round(time.time() - started, 2),
    }

    save_json(output_dir / "best_metrics.json", summary)
    save_json(docs_dir / "run_summary.json", summary)
    save_json(docs_dir / "best_hparams.json", final_hparams)

    GLOBAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with GLOBAL_LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    log(f"   gráfico halving: {halving_plot}")
    log(f"   gráfico finalistas: {finalists_plot}")
    log(f"   resumen: {docs_dir / 'run_summary.json'}")

    log("\n[7/7] Proceso finalizado")
    log("[BRF-HPT] ================================================================")
    log(
        f"[BRF-HPT] Mejor candidato C{winner:04d} | "
        f"PR-AUC={best_metrics['pr_auc']:.6f} | "
        f"ROC-AUC={best_metrics['roc_auc']:.6f} | "
        f"F1={best_metrics['f1']:.6f} | "
        f"Lift@10={best_metrics['lift_at_10']:.4f}"
    )
    log(
        f"[BRF-HPT] Árboles={final_tree_count} | "
        f"threshold={best_metrics['threshold_f1']:.6f}"
    )
    log(f"[BRF-HPT] Artefactos: {output_dir}")
    log(f"[BRF-HPT] Resumen: {docs_dir / 'run_summary.json'}")
    log("[BRF-HPT] TEST NO se utilizó para tuning, entrenamiento ni selección de umbral.")
    log(f"[BRF-HPT] Tiempo total: {(time.time() - started) / 60:.1f} minutos")
    log("[BRF-HPT] ================================================================")


if __name__ == "__main__":
    main()
