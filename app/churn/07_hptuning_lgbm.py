"""Optimización de hiperparámetros LightGBM para churn mediante Successive Halving.

Flujo:
    05_generar_target_final.py
        -> data/churn/dataset_churn_final.parquet
    06_importancia_caracteristicas.py
        -> análisis descriptivo de importancia
    07_hptuning_lgbm.py
        -> reutiliza o crea una única vez el split estratificado 80/10/10
        -> tuning exclusivamente con TRAIN + VALIDATION
        -> TEST persistido y completamente aislado

El script reutiliza la lógica robusta del tuning de recompra: candidatos aleatorios,
Successive Halving paralelo, resultados persistidos por ronda, reanudación de
corridas interrumpidas y control de oversubscription de CPU.

Diferencias para churn:
- Se utilizan las 15 características definidas para churn.
- SEXO se mantiene como categórica nativa de LightGBM.
- scale_pos_weight forma parte del espacio de hiperparámetros con la grilla
  interna {1, 2, 4, ratio TRAIN}.
- No se aplica undersampling, oversampling, SMOTE ni window slicing.
- El split final es 80 % TRAIN, 10 % VALIDATION y 10 % TEST.
- TEST nunca participa en tuning, early stopping ni selección de umbral.
- Early stopping de la ronda final utiliza PR-AUC de VALIDATION.

Ejemplo servidor con una reserva de 128 CPU:
    python app/churn/07_hptuning_lgbm.py \
        --workers 8 \
        --threads-per-worker 16 \
        --force

``--force`` reinicia únicamente los artefactos del tuning. Los splits existentes
no se eliminan ni se regeneran si su configuración es compatible.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "churn" / "dataset_churn_final.parquet"
DEFAULT_LOG = REPO_ROOT / "data" / "logs" / "07_hptuning_lgbm_churn.log"
GLOBAL_LOG_FILE = REPO_ROOT / "data" / "logs" / "churn_lgbm_hpt_runs.jsonl"
LOCK_FILE = REPO_ROOT / "data" / "locks" / "churn_lgbm.lock"

ID_COL = "IDENTIFICACION"
TARGET_COL = "target"
CATEGORICAL_COLUMNS = ["SEXO"]
SEX_CATEGORIES = ["F", "M", "DESCONOCIDO"]
CLASS_WEIGHT_GRID: tuple[float | str, ...] = (1.0, 2.0, 4.0, "auto")

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

_XTR: pd.DataFrame | None = None
_YTR: np.ndarray | None = None
_XVA: pd.DataFrame | None = None
_YVA: np.ndarray | None = None


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def parse_csv(text: str, cast: Callable) -> list:
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def crear_logger(path: Path) -> Callable[[str], None]:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n=== 07_hptuning_lgbm.py | "
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
        raise RuntimeError(f"Ya existe una ejecución LightGBM churn activa: {owner}") from exc

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

    p = argparse.ArgumentParser(
        description="LightGBM churn: split 80/10/10 + Successive Halving paralelo por PR-AUC"
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--experiment-name", type=str, default="lgbm_churn_main")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--docs-dir", type=Path, default=None)
    p.add_argument("--splits-dir", type=Path, default=None)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--candidates", "--trials", dest="candidates", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.10)
    p.add_argument("--val-size", type=float, default=0.10)
    p.add_argument("--workers", type=int, default=default_workers)
    p.add_argument("--threads-per-worker", type=int, default=default_threads)
    p.add_argument("--round-fractions", default="0.20,0.40,0.70,1.0")
    p.add_argument("--round-rounds", default="128,320,700,1600")
    p.add_argument("--survivors", default="100,25,8,1")
    p.add_argument("--final-early-stopping", type=int, default=100)
    p.add_argument(
        "--force",
        action="store_true",
        help="Reinicia artefactos del tuning; no elimina ni regenera splits compatibles.",
    )
    return p.parse_args()


def validar_dataset(df: pd.DataFrame, require_unique_ids: bool = True) -> None:
    requeridas = [ID_COL, TARGET_COL] + FEATURE_COLUMNS
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")
    if df[ID_COL].isna().any():
        raise ValueError(f"{ID_COL} contiene valores nulos")
    if require_unique_ids and df[ID_COL].duplicated().any():
        raise ValueError("El dataset debe contener una sola fila por IDENTIFICACION")
    if df[TARGET_COL].isna().any():
        raise ValueError("target contiene valores nulos")

    target = pd.to_numeric(df[TARGET_COL], errors="raise").astype(int)
    values = set(target.unique().tolist())
    if values != {0, 1}:
        raise ValueError(f"target debe contener exactamente las clases 0 y 1. Valores: {values}")


def normalizar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[ID_COL] = df[ID_COL].astype("string").str.strip()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="raise").astype("int8")
    return df


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        if col == "SEXO":
            sexo = x[col].astype("string").fillna("DESCONOCIDO")
            sexo = sexo.where(sexo.isin(SEX_CATEGORIES), "DESCONOCIDO")
            x[col] = pd.Categorical(sexo, categories=SEX_CATEGORIES)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce")
            x[col] = x[col].replace([np.inf, -np.inf], np.nan).astype("float32")
    return x


def split_estratificado(
    df: pd.DataFrame,
    seed: int,
    val_size: float,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("--val-size y --test-size deben ser > 0 y sumar menos de 1")

    idx = np.arange(len(df))
    y = df[TARGET_COL].to_numpy(dtype=np.int8)
    dev_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    val_relative = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        dev_idx,
        test_size=val_relative,
        random_state=seed + 1,
        stratify=y[dev_idx],
    )
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def distribucion_split(df: pd.DataFrame) -> dict[str, float | int]:
    y = pd.to_numeric(df[TARGET_COL], errors="raise").astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return {
        "rows": int(len(df)),
        "target_0": n_neg,
        "target_1": n_pos,
        "positive_rate": float(n_pos / len(df)) if len(df) else 0.0,
    }


def source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def split_paths(splits_dir: Path) -> dict[str, Path]:
    return {
        "train": splits_dir / "train.parquet",
        "validation": splits_dir / "validation.parquet",
        "test": splits_dir / "test.parquet",
        "summary": splits_dir / "split_summary.json",
    }


def validate_split_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    for name, frame in (
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ):
        validar_dataset(frame)
        if frame.empty:
            raise ValueError(f"El split {name} está vacío")

    train_ids = set(train_df[ID_COL].astype(str))
    val_ids = set(val_df[ID_COL].astype(str))
    test_ids = set(test_df[ID_COL].astype(str))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Los splits contienen IDENTIFICACION solapadas")


def persistir_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    splits_dir: Path,
    input_path: Path,
    seed: int,
    val_size: float,
    test_size: float,
) -> dict[str, Any]:
    splits_dir.mkdir(parents=True, exist_ok=True)
    paths = split_paths(splits_dir)
    train_df.to_parquet(paths["train"], index=False)
    val_df.to_parquet(paths["validation"], index=False)
    test_df.to_parquet(paths["test"], index=False)

    summary = {
        "schema_version": 2,
        "source": source_signature(input_path),
        "seed": seed,
        "split_type": "row_stratified",
        "train_fraction": 1.0 - val_size - test_size,
        "validation_fraction": val_size,
        "test_fraction": test_size,
        "train": {**distribucion_split(train_df), "path": str(paths["train"])},
        "validation": {**distribucion_split(val_df), "path": str(paths["validation"])},
        "test": {**distribucion_split(test_df), "path": str(paths["test"])},
    }
    save_json(paths["summary"], summary)
    return summary


def split_summary_compatible(
    summary: dict[str, Any],
    input_path: Path,
    seed: int,
    val_size: float,
    test_size: float,
) -> bool:
    source = summary.get("source")
    source_path = source.get("path") if isinstance(source, dict) else source
    expected = {
        "seed": seed,
        "train_fraction": 1.0 - val_size - test_size,
        "validation_fraction": val_size,
        "test_fraction": test_size,
    }
    if Path(str(source_path)).expanduser().resolve() != input_path.resolve():
        return False
    for key, value in expected.items():
        observed = summary.get(key)
        if key == "seed":
            if observed != value:
                return False
        elif observed is None or not math.isclose(float(observed), float(value), abs_tol=1e-12):
            return False
    return True


def cargar_o_crear_splits(
    input_path: Path,
    splits_dir: Path,
    seed: int,
    val_size: float,
    test_size: float,
    log: Callable[[str], None],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    paths = split_paths(splits_dir)
    complete = all(path.exists() for path in paths.values())

    if complete:
        with paths["summary"].open(encoding="utf-8") as handle:
            summary = json.load(handle)
        if not split_summary_compatible(summary, input_path, seed, val_size, test_size):
            raise RuntimeError(
                "Los splits existentes no corresponden al input/seed/proporciones solicitados. "
                "Use otra ruta --splits-dir o elimine explícitamente el directorio incompatible."
            )

        log("\n[1/7] Reutilizando split estratificado existente ...")
        t0 = time.time()
        train_df = normalizar_dataset(pd.read_parquet(paths["train"]))
        val_df = normalizar_dataset(pd.read_parquet(paths["validation"]))
        test_df = normalizar_dataset(pd.read_parquet(paths["test"]))
        validate_split_integrity(train_df, val_df, test_df)
        dist_total = {
            "rows": len(train_df) + len(val_df) + len(test_df),
            "target_0": sum(distribucion_split(x)["target_0"] for x in (train_df, val_df, test_df)),
            "target_1": sum(distribucion_split(x)["target_1"] for x in (train_df, val_df, test_df)),
        }
        dist_total["positive_rate"] = dist_total["target_1"] / dist_total["rows"]
        log(f"   splits cargados desde: {splits_dir}")
        log("   no se regeneraron ni sobrescribieron archivos")
        log(f"   tiempo: {time.time() - t0:.1f}s")
        return train_df, val_df, test_df, summary, dist_total

    if any(path.exists() for path in paths.values()):
        raise RuntimeError(
            f"El directorio de splits está incompleto: {splits_dir}. "
            "Elimínelo explícitamente antes de reconstruirlo."
        )

    log("\n[1/7] No existen splits; leyendo dataset y generando división 80/10/10 ...")
    t0 = time.time()
    df = normalizar_dataset(pd.read_parquet(input_path))
    validar_dataset(df)
    dist_total = distribucion_split(df)
    train_df, val_df, test_df = split_estratificado(df, seed, val_size, test_size)
    validate_split_integrity(train_df, val_df, test_df)
    summary = persistir_splits(
        train_df,
        val_df,
        test_df,
        splits_dir,
        input_path,
        seed,
        val_size,
        test_size,
    )
    del df
    gc.collect()
    log(f"   splits creados una sola vez en: {splits_dir}")
    log(f"   tiempo: {time.time() - t0:.1f}s")
    return train_df, val_df, test_df, summary, dist_total


def resolve_class_weights(train_ratio: float) -> list[float]:
    values: list[float] = []
    for item in CLASS_WEIGHT_GRID:
        value = train_ratio if item == "auto" else float(item)
        if not any(np.isclose(value, old, rtol=1e-12, atol=1e-12) for old in values):
            values.append(float(value))
    return values


def balanced_values(values: list[float], count: int, rng: np.random.Generator) -> list[float]:
    result = [values[i % len(values)] for i in range(count)]
    rng.shuffle(result)
    return result


def make_candidates(count: int, seed: int, class_weights: list[float]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    weights = balanced_values(class_weights, count, rng)
    out: list[dict[str, Any]] = []

    for cid in range(count):
        params = {
            "objective": "binary",
            "metric": "None",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
            "seed": seed + cid,
            "feature_fraction_seed": seed + cid,
            "bagging_seed": seed + cid,
            "data_random_seed": seed + cid,
            "learning_rate": log_uniform(rng, 0.01, 0.15),
            "num_leaves": int(round(log_uniform(rng, 15, 255))),
            "max_depth": int(rng.choice([-1, 6, 8, 10, 12, 16])),
            "min_data_in_leaf": int(round(log_uniform(rng, 50, 2000))),
            "feature_fraction": float(rng.uniform(0.55, 1.0)),
            "bagging_fraction": float(rng.uniform(0.55, 1.0)),
            "bagging_freq": 1,
            "lambda_l1": log_uniform(rng, 1e-8, 10.0),
            "lambda_l2": log_uniform(rng, 1e-8, 10.0),
            "min_gain_to_split": float(rng.uniform(0.0, 1.0)),
            "max_bin": int(rng.choice([63, 127])),
            "scale_pos_weight": float(weights[cid]),
        }
        out.append({"candidate_id": cid, "params": params})
    return out


def nested_order(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng.shuffle(pos)
    rng.shuffle(neg)

    order = np.empty(len(y), dtype=np.int64)
    p = n = 0
    pos_rate = len(pos) / len(y)
    for i in range(len(y)):
        need_pos = (p < len(pos)) and (n >= len(neg) or p / max(1, i) < pos_rate)
        if need_pos:
            order[i] = pos[p]
            p += 1
        else:
            order[i] = neg[n]
            n += 1
    return order


def subset_from_order(
    x: pd.DataFrame,
    y: np.ndarray,
    order: np.ndarray,
    fraction: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    if fraction >= 1.0:
        return x, y
    idx = order[: max(2, int(round(len(order) * fraction)))]
    return x.iloc[idx].reset_index(drop=True), np.ascontiguousarray(y[idx], dtype=np.float32)


def pr_auc_eval(preds: np.ndarray, dataset: lgb.Dataset):
    y_true = dataset.get_label()
    score = float(average_precision_score(y_true, preds))
    return "pr_auc", score, True


def metricas(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pr_auc = float(average_precision_score(y, p))
    roc_auc = float(roc_auc_score(y, p))
    precision_curve, recall_curve, thresholds = precision_recall_curve(y, p)
    if len(thresholds):
        den = precision_curve[:-1] + recall_curve[:-1]
        f1_values = np.divide(
            2 * precision_curve[:-1] * recall_curve[:-1],
            den,
            out=np.zeros_like(den),
            where=den > 0,
        )
        threshold = float(thresholds[int(np.argmax(f1_values))])
    else:
        threshold = 0.5

    pred = (p >= threshold).astype(np.int8)
    base_rate = float(y.mean())
    n_top = max(1, int(math.ceil(len(p) * 0.10)))
    top_idx = np.argpartition(p, -n_top)[-n_top:]
    top_rate = float(y[top_idx].mean())
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "lift_at_10": top_rate / base_rate if base_rate else 0.0,
        "threshold_f1": threshold,
        "base_rate": base_rate,
        "top10_positive_rate": top_rate,
    }


def train_booster(
    params: dict[str, Any],
    rounds: int,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    early_stopping: int,
):
    local = dict(params)
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        feature_name=FEATURE_COLUMNS,
        categorical_feature=CATEGORICAL_COLUMNS,
        free_raw_data=True,
    )
    callbacks = [lgb.log_evaluation(0)]

    if early_stopping > 0:
        local["metric"] = "None"
        val_set = lgb.Dataset(
            x_val,
            label=y_val,
            reference=train_set,
            feature_name=FEATURE_COLUMNS,
            categorical_feature=CATEGORICAL_COLUMNS,
            free_raw_data=True,
        )
        callbacks.append(
            lgb.early_stopping(
                early_stopping,
                first_metric_only=True,
                verbose=False,
            )
        )
        model = lgb.train(
            local,
            train_set,
            num_boost_round=rounds,
            valid_sets=[val_set],
            valid_names=["validation"],
            feval=pr_auc_eval,
            callbacks=callbacks,
        )
        best_iteration = int(model.best_iteration or rounds)
    else:
        local["metric"] = "None"
        model = lgb.train(
            local,
            train_set,
            num_boost_round=rounds,
            callbacks=callbacks,
        )
        best_iteration = rounds
    return model, best_iteration


def evaluate(task):
    cid, params, rounds, threads, early_stopping = task
    if any(v is None for v in (_XTR, _YTR, _XVA, _YVA)):
        raise RuntimeError("Datos del worker no inicializados")

    local = dict(params)
    local["num_threads"] = threads
    started = time.time()
    model, best_iteration = train_booster(
        local,
        rounds,
        _XTR,
        _YTR,
        _XVA,
        _YVA,
        early_stopping,
    )
    probabilities = model.predict(_XVA, num_iteration=best_iteration)
    result = metricas(_YVA, probabilities)
    return {
        "candidate_id": cid,
        "scale_pos_weight": float(params["scale_pos_weight"]),
        "best_iteration": best_iteration,
        "duration_seconds": round(time.time() - started, 3),
        **{k: float(v) for k, v in result.items()},
        "params_json": json.dumps(params, sort_keys=True),
    }


def save_records(records: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(records).sort_values("candidate_id")
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return frame


def plot_halving(all_results: pd.DataFrame, output: Path) -> None:
    if all_results.empty:
        return
    rounds = sorted(all_results["round"].astype(int).unique())
    values = [
        all_results.loc[all_results["round"].astype(int) == r, "pr_auc"].to_numpy()
        for r in rounds
    ]
    best = [float(np.max(v)) for v in values]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [f"R{r}" for r in rounds]
    try:
        ax.boxplot(values, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(values, labels=labels, showfliers=False)
    ax.plot(range(1, len(rounds) + 1), best, marker="o", label="Mejor PR-AUC")
    ax.set_xlabel("Ronda de Successive Halving")
    ax.set_ylabel("PR-AUC de validación")
    ax.set_title("LightGBM churn — evolución del Successive Halving")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def weight_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    return (
        all_results.groupby(["round", "scale_pos_weight"], as_index=False)
        .agg(
            candidates=("candidate_id", "count"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_median=("pr_auc", "median"),
            pr_auc_max=("pr_auc", "max"),
            f1_max=("f1", "max"),
            lift_at_10_max=("lift_at_10", "max"),
        )
        .sort_values(["round", "pr_auc_max"], ascending=[True, False])
    )


def main() -> None:
    args = parse_args()
    log = crear_logger(args.log)
    started = time.time()

    fractions = parse_csv(args.round_fractions, float)
    budgets = parse_csv(args.round_rounds, int)
    survivors = parse_csv(args.survivors, int)

    if args.candidates <= 0:
        raise ValueError("--candidates debe ser mayor que 0")
    if args.workers <= 0 or args.threads_per_worker <= 0:
        raise ValueError("--workers y --threads-per-worker deben ser mayores que 0")
    if args.final_early_stopping < 0:
        raise ValueError("--final-early-stopping no puede ser negativo")
    if not (
        len(fractions) == len(budgets) == len(survivors)
        and fractions == sorted(fractions)
        and fractions[-1] == 1.0
        and survivors[-1] == 1
    ):
        raise ValueError("Configuración de Successive Halving inválida")

    active_count = args.candidates
    for keep in survivors:
        if keep <= 0 or keep > active_count:
            raise ValueError("Secuencia --survivors incompatible con --candidates")
        active_count = keep

    cpus = available_cpus()
    requested = args.workers * args.threads_per_worker
    if requested > cpus:
        raise ValueError(
            f"Solicitas {requested} hilos ({args.workers}x{args.threads_per_worker}), "
            f"pero la reserva permite {cpus}"
        )

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

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

    train_fraction = 1.0 - args.val_size - args.test_size
    split_name = (
        f"split_{int(round(train_fraction * 100))}_"
        f"{int(round(args.val_size * 100))}_"
        f"{int(round(args.test_size * 100))}_seed_{args.seed}"
    )
    splits_dir = (
        args.splits_dir.expanduser().resolve()
        if args.splits_dir is not None
        else REPO_ROOT / "data" / "churn" / "splits" / split_name
    )

    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    if args.force and docs_dir.exists():
        shutil.rmtree(docs_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    acquire_lock(f"hpt:churn:{args.experiment_name}")

    log("\n=== CONFIGURACION ===")
    log(f"input: {input_path}")
    log(f"experiment_name: {args.experiment_name}")
    log(f"output_dir: {output_dir}")
    log(f"docs_dir: {docs_dir}")
    log(f"splits_dir: {splits_dir}")
    log(f"candidatos: {args.candidates}")
    log(f"split global: train={train_fraction:.0%} | validation={args.val_size:.0%} | test={args.test_size:.0%}")
    log(f"scale_pos_weight grid: {list(CLASS_WEIGHT_GRID)}")
    log(f"round_fractions: {fractions}")
    log(f"round_rounds: {budgets}")
    log(f"survivors: {survivors}")
    log(f"early_stopping final: {args.final_early_stopping}")
    log(f"CPU: workers={args.workers} x threads={args.threads_per_worker} = {requested}/{cpus}")

    train_df, val_df, test_df, split_summary, dist_total = cargar_o_crear_splits(
        input_path,
        splits_dir,
        args.seed,
        args.val_size,
        args.test_size,
        log,
    )

    log("\n[2/7] Verificando distribución de los splits ...")
    for name, frame in (("train", train_df), ("validation", val_df), ("test", test_df)):
        d = distribucion_split(frame)
        log(
            f"   {name:<10}: filas={d['rows']:,} | target0={d['target_0']:,} | "
            f"target1={d['target_1']:,} | churn={d['positive_rate'] * 100:.2f}%"
        )
    log("   TEST queda persistido y NO se utiliza durante el tuning")

    log("\n[3/7] Preparando TRAIN y VALIDATION ...")
    t0 = time.time()
    xtr_full = preparar_features(train_df)
    xva_full = preparar_features(val_df)
    ytr_full = train_df[TARGET_COL].to_numpy(dtype=np.float32)
    yva_full = val_df[TARGET_COL].to_numpy(dtype=np.float32)

    n_pos_train = int((ytr_full == 1).sum())
    n_neg_train = int((ytr_full == 0).sum())
    train_ratio = n_neg_train / n_pos_train
    class_weights = resolve_class_weights(train_ratio)

    nulos_train = int(xtr_full.select_dtypes(include=[np.number]).isna().sum().sum())
    nulos_val = int(xva_full.select_dtypes(include=[np.number]).isna().sum().sum())
    log(f"   ratio natural TRAIN = {n_neg_train:,} / {n_pos_train:,} = {train_ratio:.6f}")
    log(f"   scale_pos_weight evaluados: {[round(x, 6) for x in class_weights]}")
    log(f"   SEXO: categórica nativa de LightGBM ({SEX_CATEGORIES})")
    log(f"   nulos numericos train: {nulos_train:,}")
    log(f"   nulos numericos validation: {nulos_val:,}")
    log(f"   tiempo: {time.time() - t0:.1f}s")

    del test_df
    gc.collect()

    train_order = nested_order(ytr_full, args.seed + 100)
    val_order = nested_order(yva_full, args.seed + 200)
    candidates = make_candidates(args.candidates, args.seed, class_weights)
    by_id = {c["candidate_id"]: c for c in candidates}

    definition = {
        "schema_version": 2,
        "search_method": "parallel_successive_halving_random_nested",
        "input": str(input_path),
        "experiment_name": args.experiment_name,
        "features": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "sex_categories": SEX_CATEGORIES,
        "candidates": args.candidates,
        "seed": args.seed,
        "split": {
            "type": "row_stratified",
            "train_fraction": train_fraction,
            "validation_fraction": args.val_size,
            "test_fraction": args.test_size,
            "splits_dir": str(splits_dir),
            "reuse_existing": True,
        },
        "scale_pos_weight_grid_definition": list(CLASS_WEIGHT_GRID),
        "scale_pos_weight_grid_resolved": class_weights,
        "scale_pos_weight_train_ratio": train_ratio,
        "round_fractions": fractions,
        "round_rounds": budgets,
        "survivors": survivors,
        "final_early_stopping": args.final_early_stopping,
        "early_stopping_metric": "PR-AUC",
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
    }

    definition_path = output_dir / "run_definition.json"
    if definition_path.exists() and not args.force:
        with definition_path.open(encoding="utf-8") as handle:
            old = json.load(handle)
        ignored = {"workers", "threads_per_worker"}
        old_cmp = {k: v for k, v in old.items() if k not in ignored}
        new_cmp = {k: v for k, v in definition.items() if k not in ignored}
        if old_cmp != new_cmp:
            raise RuntimeError(
                "El experimento existente tiene una configuración incompatible. "
                "Ejecute con --force para reiniciar SOLO el tuning; los splits se conservarán."
            )
    save_json(definition_path, definition)

    log("\n[4/7] Iniciando Successive Halving ...")
    log("   criterio único de supervivencia: PR-AUC de validation")
    log("   scale_pos_weight participa dentro del espacio de hiperparámetros")
    log("   early stopping por PR-AUC únicamente en la ronda final")

    active = list(by_id)
    all_frames: list[pd.DataFrame] = []
    global _XTR, _YTR, _XVA, _YVA

    for round_id, (fraction, rounds, keep) in enumerate(
        zip(fractions, budgets, survivors),
        start=1,
    ):
        result_path = output_dir / f"round_{round_id}_results.csv"
        old_frame = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
        records = old_frame.to_dict("records") if not old_frame.empty else []
        completed = set(old_frame["candidate_id"].astype(int)) if not old_frame.empty else set()
        pending = [cid for cid in active if cid not in completed]

        _XTR, _YTR = subset_from_order(xtr_full, ytr_full, train_order, fraction)
        _XVA, _YVA = subset_from_order(xva_full, yva_full, val_order, fraction)
        early_stopping = args.final_early_stopping if round_id == len(fractions) else 0

        log(
            f"\n[LGBM-R{round_id}] candidatos={len(active)} | pendientes={len(pending)} | "
            f"datos={fraction:.0%} | train={len(_YTR):,} | val={len(_YVA):,} | "
            f"arboles={rounds} | sobreviven={min(keep, len(active))}"
        )

        round_started = time.time()
        if pending:
            tasks = [
                (
                    cid,
                    by_id[cid]["params"],
                    rounds,
                    args.threads_per_worker,
                    early_stopping,
                )
                for cid in pending
            ]
            done = 0
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=mp.get_context("fork"),
            ) as executor:
                futures = {executor.submit(evaluate, task): task[0] for task in tasks}
                for future in as_completed(futures):
                    record = future.result()
                    record.update(
                        {
                            "round": round_id,
                            "data_fraction": fraction,
                            "max_rounds": rounds,
                        }
                    )
                    records.append(record)
                    done += 1
                    save_records(records, result_path)

                    elapsed = time.time() - round_started
                    eta = elapsed / max(1, done) * (len(pending) - done)
                    log(
                        f"[LGBM-R{round_id} {done:03d}/{len(pending):03d}] "
                        f"C{int(record['candidate_id']):04d} | "
                        f"w={record['scale_pos_weight']:.4f} | "
                        f"PR-AUC={record['pr_auc']:.6f} | "
                        f"F1={record['f1']:.6f} | "
                        f"Lift@10={record['lift_at_10']:.4f} | "
                        f"iter={int(record['best_iteration'])} | "
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
        all_frames.append(frame)
        round_duration = time.time() - round_started
        save_json(
            output_dir / f"round_{round_id}_survivors.json",
            {
                "round": round_id,
                "survivors": active,
                "best_candidate": int(active[0]),
                "best_pr_auc": float(frame.iloc[0]["pr_auc"]),
                "best_scale_pos_weight": float(frame.iloc[0]["scale_pos_weight"]),
                "duration_seconds": round_duration,
            },
        )
        log(
            f"[LGBM-R{round_id}] mejor=C{active[0]:04d} | "
            f"w={float(frame.iloc[0]['scale_pos_weight']):.4f} | "
            f"PR-AUC={float(frame.iloc[0]['pr_auc']):.6f} | "
            f"pasan={len(active)} | tiempo={round_duration / 60:.1f}m"
        )

        _XTR = _YTR = _XVA = _YVA = None
        gc.collect()

    log("\n[5/7] Confirmando candidato ganador sobre TRAIN y VALIDATION completos ...")
    winner = active[0]
    final_params = dict(by_id[winner]["params"])
    final_params["num_threads"] = args.threads_per_worker
    model, best_iteration = train_booster(
        final_params,
        budgets[-1],
        xtr_full,
        ytr_full,
        xva_full,
        yva_full,
        args.final_early_stopping,
    )
    validation_probabilities = model.predict(xva_full, num_iteration=best_iteration)
    best_metrics = metricas(yva_full, validation_probabilities)
    final_hparams = {**final_params, "num_boost_round": best_iteration}

    save_json(output_dir / "best_hparams.json", final_hparams)
    model.save_model(str(output_dir / "best_model_validation.txt"), num_iteration=best_iteration)

    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv(output_dir / "all_round_results.csv", index=False)
    all_frames[-1].sort_values("pr_auc", ascending=False).head(10).to_csv(
        output_dir / "top10_trials.csv",
        index=False,
    )
    weights_table = weight_summary(all_results)
    weights_table.to_csv(output_dir / "scale_pos_weight_summary.csv", index=False)

    log(f"   ganador: C{winner:04d}")
    log(f"   scale_pos_weight ganador: {final_params['scale_pos_weight']:.6f}")
    log(f"   best_iteration: {best_iteration}")
    log(f"   PR-AUC validation: {best_metrics['pr_auc']:.6f}")
    log(f"   ROC-AUC validation: {best_metrics['roc_auc']:.6f}")
    log(f"   Precision validation: {best_metrics['precision']:.6f}")
    log(f"   Recall validation: {best_metrics['recall']:.6f}")
    log(f"   F1 validation: {best_metrics['f1']:.6f}")
    log(f"   Lift@10 validation: {best_metrics['lift_at_10']:.4f}")
    log(f"   threshold F1 validation: {best_metrics['threshold_f1']:.6f}")

    log("\n[6/7] Generando resumen, tabla y grafico de tuning ...")
    plot_path = docs_dir / "successive_halving_pr_auc.png"
    plot_halving(all_results, plot_path)
    weights_table.to_csv(docs_dir / "scale_pos_weight_summary.csv", index=False)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": args.experiment_name,
        "model": "LightGBM",
        "problem": "churn_cliente",
        "search_method": "parallel_successive_halving_random_nested",
        "objective": "maximize_pr_auc_validation",
        "input": str(input_path),
        "n_rows_total": int(dist_total["rows"]),
        "target_positive_rate_total": float(dist_total["positive_rate"]),
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "n_features_conceptual": len(FEATURE_COLUMNS),
        "split_summary": split_summary,
        "scale_pos_weight_grid_definition": list(CLASS_WEIGHT_GRID),
        "scale_pos_weight_grid_resolved": class_weights,
        "scale_pos_weight_train_ratio": train_ratio,
        "scale_pos_weight_selected": float(final_params["scale_pos_weight"]),
        "candidate_count": args.candidates,
        "best_candidate_id": winner,
        "best_params": final_hparams,
        "best_metrics_validation": best_metrics,
        "round_fractions": fractions,
        "round_rounds": budgets,
        "survivors": survivors,
        "early_stopping_metric": "PR-AUC",
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "test_used_in_tuning": False,
        "plot": str(plot_path),
        "weight_summary": str(docs_dir / "scale_pos_weight_summary.csv"),
        "duration_seconds": round(time.time() - started, 2),
    }

    save_json(output_dir / "best_metrics.json", summary)
    save_json(docs_dir / "run_summary.json", summary)
    save_json(docs_dir / "best_hparams.json", final_hparams)

    GLOBAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with GLOBAL_LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    log(f"   grafico: {plot_path}")
    log(f"   tabla pesos: {docs_dir / 'scale_pos_weight_summary.csv'}")
    log(f"   resumen: {docs_dir / 'run_summary.json'}")

    log("\n[7/7] Proceso finalizado")
    log("[LGBM-HPT] ================================================================")
    log(
        f"[LGBM-HPT] Mejor candidato C{winner:04d} | "
        f"PR-AUC={best_metrics['pr_auc']:.6f} | "
        f"ROC-AUC={best_metrics['roc_auc']:.6f} | "
        f"F1={best_metrics['f1']:.6f} | "
        f"Lift@10={best_metrics['lift_at_10']:.4f}"
    )
    log(
        f"[LGBM-HPT] Iteraciones={best_iteration} | "
        f"threshold={best_metrics['threshold_f1']:.6f} | "
        f"scale_pos_weight seleccionado={final_params['scale_pos_weight']:.6f}"
    )
    log(f"[LGBM-HPT] Artefactos: {output_dir}")
    log(f"[LGBM-HPT] Splits reutilizables: {splits_dir}")
    log(f"[LGBM-HPT] Resumen rastreable: {docs_dir / 'run_summary.json'}")
    log("[LGBM-HPT] TEST NO se utilizó durante tuning, early stopping ni selección de umbral.")
    log(f"[LGBM-HPT] Tiempo total: {(time.time() - started) / 60:.1f} minutos")
    log("[LGBM-HPT] ================================================================")


if __name__ == "__main__":
    main()
