"""Tuning paralelo de LightGBM con Successive Halving.

Usa W0-W3 únicamente. Separa 80/20 de forma estratificada, maximiza PR-AUC
y descarta pronto las configuraciones poco prometedoras. W4 no participa.
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
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from config import MODELS_DIR
from hptuning import DOCS_HPT_DIR, matrix, metrics, resolve_features, save_json, seed_all

ROOT_DIR = Path(__file__).resolve().parent.parent
LOCK_FILE = ROOT_DIR / "data" / "locks" / "lgbm.lock"
GLOBAL_LOG_FILE = ROOT_DIR / "data" / "logs" / "lgbm_hpt_runs.jsonl"

_WORKER_X_TRAIN: np.ndarray | None = None
_WORKER_Y_TRAIN: np.ndarray | None = None
_WORKER_X_VAL: np.ndarray | None = None
_WORKER_Y_VAL: np.ndarray | None = None


def acquire_lock(label: str):
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


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def parse_values(text: str, cast) -> list:
    return [cast(value.strip()) for value in text.split(",") if value.strip()]


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def generate_candidates(count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = []
    for candidate_id in range(count):
        params = {
            "objective": "binary",
            "metric": "None",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
            "seed": seed + candidate_id,
            "feature_fraction_seed": seed + candidate_id,
            "bagging_seed": seed + candidate_id,
            "data_random_seed": seed + candidate_id,
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
            "scale_pos_weight": float(rng.choice([1.0, 2.0, 4.0])),
        }
        candidates.append({"candidate_id": candidate_id, "params": params})
    return candidates


def stratified_subset(
    x: np.ndarray,
    y: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if fraction >= 1.0:
        return x, y
    selected, _ = train_test_split(
        np.arange(len(y)),
        train_size=fraction,
        random_state=seed,
        stratify=y,
    )
    return (
        np.ascontiguousarray(x[selected], dtype=np.float32),
        np.ascontiguousarray(y[selected], dtype=np.float32),
    )


def train_booster(
    params: dict[str, Any],
    num_boost_round: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int,
) -> tuple[lgb.Booster, int]:
    local_params = dict(params)
    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=True)
    callbacks = [lgb.log_evaluation(0)]

    if early_stopping_rounds > 0:
        local_params["metric"] = "binary_logloss"
        val_set = lgb.Dataset(x_val, label=y_val, reference=train_set, free_raw_data=True)
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                first_metric_only=True,
                verbose=False,
            )
        )
        booster = lgb.train(
            local_params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[val_set],
            valid_names=["validation"],
            callbacks=callbacks,
        )
        best_iteration = int(booster.best_iteration or num_boost_round)
    else:
        local_params["metric"] = "None"
        booster = lgb.train(
            local_params,
            train_set,
            num_boost_round=num_boost_round,
            callbacks=callbacks,
        )
        best_iteration = num_boost_round

    return booster, best_iteration


def evaluate_candidate(task: tuple[int, dict[str, Any], int, int, int]) -> dict[str, Any]:
    candidate_id, params, rounds, threads, early_stopping_rounds = task
    if any(value is None for value in (
        _WORKER_X_TRAIN, _WORKER_Y_TRAIN, _WORKER_X_VAL, _WORKER_Y_VAL
    )):
        raise RuntimeError("Datos del worker no inicializados")

    local_params = dict(params)
    local_params["num_threads"] = threads
    started = time.time()
    booster, best_iteration = train_booster(
        local_params,
        rounds,
        _WORKER_X_TRAIN,
        _WORKER_Y_TRAIN,
        _WORKER_X_VAL,
        _WORKER_Y_VAL,
        early_stopping_rounds,
    )
    probabilities = booster.predict(_WORKER_X_VAL, num_iteration=best_iteration)
    result = metrics(_WORKER_Y_VAL, probabilities)
    return {
        "candidate_id": candidate_id,
        "best_iteration": best_iteration,
        "duration_seconds": round(time.time() - started, 3),
        **{key: float(value) for key, value in result.items()},
        "params_json": json.dumps(params, sort_keys=True),
    }


def save_round_records(records: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(records).sort_values("candidate_id")
    frame.to_csv(path, index=False)
    return frame


def plot_history(results: pd.DataFrame, path: Path) -> None:
    if results.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for round_id, group in results.groupby("round"):
        ax.scatter(
            group["candidate_id"],
            group["pr_auc"],
            s=18,
            alpha=0.55,
            label=f"Ronda {int(round_id)}",
        )
    ax.set(
        xlabel="Candidato",
        ylabel="PR-AUC validación",
        title="Successive Halving — LightGBM",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cpus = available_cpus()
    default_workers = min(8, max(1, cpus // 8))
    default_threads = max(1, cpus // default_workers)

    parser = argparse.ArgumentParser(
        description="Tuning LightGBM paralelo con Successive Halving y objetivo PR-AUC"
    )
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--feature-set", choices=["gain95", "all"], required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--candidates", "--trials", dest="candidates", type=int, default=500)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--threads-per-worker", type=int, default=default_threads)
    parser.add_argument("--round-fractions", default="0.10,0.25,0.50,1.0")
    parser.add_argument("--round-rounds", default="64,192,512,1200")
    parser.add_argument("--survivors", default="80,16,5,1")
    parser.add_argument("--final-early-stopping", type=int, default=80)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    fractions = parse_values(args.round_fractions, float)
    round_rounds = parse_values(args.round_rounds, int)
    survivors = parse_values(args.survivors, int)
    if not (
        args.candidates > 0
        and args.workers > 0
        and args.threads_per_worker > 0
        and 0 < args.val_size < 1
        and len(fractions) == len(round_rounds) == len(survivors)
        and fractions[-1] == 1.0
        and survivors[-1] == 1
    ):
        raise ValueError("Configuración de Successive Halving inválida")
    if any(not 0 < value <= 1 for value in fractions):
        raise ValueError("Las fracciones deben estar entre 0 y 1")
    if any(value <= 0 for value in round_rounds + survivors):
        raise ValueError("Rondas y supervivientes deben ser positivos")
    if args.workers * args.threads_per_worker > cpus:
        raise ValueError(
            f"Solicitas {args.workers * args.threads_per_worker} hilos, "
            f"pero esta reserva permite {cpus}"
        )

    seed_all(args.seed)
    train_path = Path(args.train_parquet)
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    out = MODELS_DIR / "hpt" / args.experiment_name
    plots = DOCS_HPT_DIR / args.experiment_name
    if args.force and out.exists():
        shutil.rmtree(out)
    if args.force and plots.exists():
        shutil.rmtree(plots)
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    acquire_lock(f"hpt:{args.experiment_name}")

    started = time.time()
    df = pd.read_parquet(train_path)
    if "target" not in df or df["target"].nunique() != 2:
        raise ValueError("target binario requerido")

    base_features, onehot, excluded = resolve_features(df, args.feature_set)
    x, feature_names = matrix(df, base_features, onehot)
    y = df["target"].to_numpy(np.float32)
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y,
    )
    x_train_full = np.ascontiguousarray(x[train_idx], dtype=np.float32)
    x_val_full = np.ascontiguousarray(x[val_idx], dtype=np.float32)
    y_train_full = np.ascontiguousarray(y[train_idx], dtype=np.float32)
    y_val_full = np.ascontiguousarray(y[val_idx], dtype=np.float32)
    del x, train_idx, val_idx
    gc.collect()

    definition = {
        "schema_version": 2,
        "search_method": "parallel_successive_halving_random",
        "train_parquet": str(train_path.resolve()),
        "feature_set": args.feature_set,
        "candidates": args.candidates,
        "val_size": args.val_size,
        "seed": args.seed,
        "round_fractions": fractions,
        "round_rounds": round_rounds,
        "survivors": survivors,
        "final_early_stopping": args.final_early_stopping,
        "feature_columns": feature_names,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
    }
    definition_path = out / "run_definition.json"
    if definition_path.exists() and not args.force:
        old = json.load(open(definition_path, encoding="utf-8"))
        ignored = {"workers", "threads_per_worker"}
        old_core = {key: value for key, value in old.items() if key not in ignored}
        new_core = {key: value for key, value in definition.items() if key not in ignored}
        if old_core != new_core:
            raise RuntimeError("Experimento existente incompatible; use otro nombre o --force")
    save_json(definition_path, definition)

    print(
        f"[LGBM-HPT] Experimento={args.experiment_name} | feature_set={args.feature_set} "
        f"| filas={len(df):,} | train={len(y_train_full):,} | val={len(y_val_full):,} "
        f"| features={len(feature_names)} | positivos={y.mean()*100:.2f}%"
    )
    print(
        f"[LGBM-HPT] Successive Halving | candidatos={args.candidates} "
        f"| workers={args.workers} × {args.threads_per_worker} hilos "
        f"| CPU total={args.workers * args.threads_per_worker}/{cpus}"
    )
    print("[LGBM-HPT] PR-AUC se calcula una sola vez por candidato; W4 no se utiliza.")

    candidates = generate_candidates(args.candidates, args.seed)
    candidate_by_id = {item["candidate_id"]: item for item in candidates}
    active_ids = [item["candidate_id"] for item in candidates]
    round_frames: list[pd.DataFrame] = []

    global _WORKER_X_TRAIN, _WORKER_Y_TRAIN, _WORKER_X_VAL, _WORKER_Y_VAL

    for round_index, (fraction, rounds, keep) in enumerate(
        zip(fractions, round_rounds, survivors),
        start=1,
    ):
        round_path = out / f"round_{round_index}_results.csv"
        existing_frame = pd.read_csv(round_path) if round_path.exists() else pd.DataFrame()
        records = existing_frame.to_dict("records") if not existing_frame.empty else []
        completed_ids = (
            set(existing_frame["candidate_id"].astype(int))
            if not existing_frame.empty
            else set()
        )
        pending_ids = [candidate_id for candidate_id in active_ids if candidate_id not in completed_ids]

        _WORKER_X_TRAIN, _WORKER_Y_TRAIN = stratified_subset(
            x_train_full,
            y_train_full,
            fraction,
            args.seed + round_index * 10,
        )
        _WORKER_X_VAL, _WORKER_Y_VAL = stratified_subset(
            x_val_full,
            y_val_full,
            fraction,
            args.seed + round_index * 10 + 1,
        )

        early_stopping = args.final_early_stopping if round_index == len(fractions) else 0
        print(
            f"\n[LGBM-R{round_index}] candidatos={len(active_ids)} "
            f"| pendientes={len(pending_ids)} | datos={fraction:.0%} "
            f"| train={len(_WORKER_Y_TRAIN):,} | val={len(_WORKER_Y_VAL):,} "
            f"| árboles máx={rounds} | sobreviven={min(keep, len(active_ids))}",
            flush=True,
        )

        if pending_ids:
            tasks = [
                (
                    candidate_id,
                    candidate_by_id[candidate_id]["params"],
                    rounds,
                    args.threads_per_worker,
                    early_stopping,
                )
                for candidate_id in pending_ids
            ]
            round_started = time.time()
            local_done = 0
            context = mp.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=context,
            ) as executor:
                futures = {executor.submit(evaluate_candidate, task): task[0] for task in tasks}
                for future in as_completed(futures):
                    record = future.result()
                    record.update({
                        "round": round_index,
                        "data_fraction": fraction,
                        "max_rounds": rounds,
                    })
                    records.append(record)
                    local_done += 1
                    save_round_records(records, round_path)
                    elapsed = time.time() - round_started
                    remaining = len(pending_ids) - local_done
                    eta = elapsed / max(1, local_done) * remaining
                    print(
                        f"[LGBM-R{round_index} {local_done:03d}/{len(pending_ids):03d}] "
                        f"C{int(record['candidate_id']):04d} "
                        f"| PR-AUC={record['pr_auc']:.6f} "
                        f"| F1={record['f1']:.6f} "
                        f"| Lift@10={record['lift_at_10']:.4f} "
                        f"| iter={int(record['best_iteration'])} "
                        f"| {record['duration_seconds']:.1f}s "
                        f"| ETA={eta/60:.1f}m",
                        flush=True,
                    )

        round_frame = save_round_records(records, round_path)
        round_frame = round_frame[
            round_frame["candidate_id"].astype(int).isin(active_ids)
        ].copy()
        round_frame = round_frame.sort_values(
            ["pr_auc", "candidate_id"],
            ascending=[False, True],
        )
        keep_count = min(keep, len(round_frame))
        active_ids = round_frame.head(keep_count)["candidate_id"].astype(int).tolist()
        round_frames.append(round_frame)
        save_json(
            out / f"round_{round_index}_survivors.json",
            {
                "round": round_index,
                "survivors": active_ids,
                "best_pr_auc": float(round_frame.iloc[0]["pr_auc"]),
            },
        )
        print(
            f"[LGBM-R{round_index}] mejor=C{active_ids[0]:04d} "
            f"| PR-AUC={float(round_frame.iloc[0]['pr_auc']):.6f} "
            f"| pasan={len(active_ids)}",
            flush=True,
        )

        _WORKER_X_TRAIN = _WORKER_Y_TRAIN = None
        _WORKER_X_VAL = _WORKER_Y_VAL = None
        gc.collect()

    winner_id = active_ids[0]
    winner_params = dict(candidate_by_id[winner_id]["params"])
    winner_params["num_threads"] = args.threads_per_worker
    final_rounds = round_rounds[-1]
    best_booster, best_iteration = train_booster(
        winner_params,
        final_rounds,
        x_train_full,
        y_train_full,
        x_val_full,
        y_val_full,
        args.final_early_stopping,
    )
    probabilities = best_booster.predict(x_val_full, num_iteration=best_iteration)
    best_metrics = metrics(y_val_full, probabilities)

    final_hparams = {
        **winner_params,
        "num_boost_round": best_iteration,
    }
    save_json(out / "best_hparams.json", final_hparams)
    best_booster.save_model(str(out / "best_model.txt"), num_iteration=best_iteration)

    all_results = pd.concat(round_frames, ignore_index=True)
    all_results.to_csv(out / "all_round_results.csv", index=False)
    final_frame = round_frames[-1].sort_values("pr_auc", ascending=False)
    final_frame.head(10).to_csv(out / "top10_trials.csv", index=False)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": args.experiment_name,
        "model": "LightGBM",
        "search_method": "parallel_successive_halving_random",
        "objective": "maximize_pr_auc_validation",
        "feature_set": args.feature_set,
        "train_parquet": str(train_path),
        "n_rows_total": len(df),
        "n_rows_train": len(y_train_full),
        "n_rows_validation": len(y_val_full),
        "target_positive_rate_total": float(y.mean()),
        "split": {
            "type": "row_stratified",
            "validation_size": args.val_size,
            "seed": args.seed,
        },
        "base_feature_columns": base_features,
        "model_feature_columns": feature_names,
        "excluded_columns": excluded,
        "cycle_onehot": onehot,
        "candidate_count": args.candidates,
        "best_candidate_id": winner_id,
        "best_trial_number": winner_id,
        "best_params": final_hparams,
        "best_metrics_validation": best_metrics,
        "round_fractions": fractions,
        "round_rounds": round_rounds,
        "survivors": survivors,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "duration_seconds": round(time.time() - started, 2),
        "pr_auc_evaluation": "once_per_candidate_after_training",
        "early_stopping_metric_final_round": "binary_logloss",
    }
    save_json(out / "best_metrics.json", summary)
    save_json(plots / "run_summary.json", summary)
    save_json(plots / "best_hparams.json", final_hparams)

    GLOBAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    plot_history(all_results, plots / "optimization_history.png")
    precision, recall, _ = precision_recall_curve(y_val_full, probabilities)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label="LightGBM")
    ax.axhline(float(y_val_full.mean()), linestyle="--", label="Prevalencia")
    ax.set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall — validación")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "best_pr_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\n[LGBM-HPT] ================================================================")
    print(
        f"[LGBM-HPT] Mejor candidato C{winner_id:04d} "
        f"| PR-AUC={best_metrics['pr_auc']:.6f} "
        f"| ROC-AUC={best_metrics['roc_auc']:.6f} "
        f"| F1={best_metrics['f1']:.6f} "
        f"| Lift@10={best_metrics['lift_at_10']:.4f}"
    )
    print(
        f"[LGBM-HPT] Iteraciones finales={best_iteration} "
        f"| threshold={best_metrics['threshold_f1']:.6f}"
    )
    print(f"[LGBM-HPT] Artefactos: {out}")
    print(f"[LGBM-HPT] Resumen rastreable: {plots / 'run_summary.json'}")
    print("[LGBM-HPT] W4/test NO se utilizó durante el tuning.")
    print("[LGBM-HPT] ================================================================")


if __name__ == "__main__":
    main()
