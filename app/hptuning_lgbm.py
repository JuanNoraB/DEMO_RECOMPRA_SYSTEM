"""LightGBM tuning with parallel Successive Halving.

Only W0-W3 are used. A fixed stratified 80/20 split is created, and each
round uses a nested stratified subset. PR-AUC selects candidates. W4 is never
used here.
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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import MODELS_DIR
from hptuning import DOCS_HPT_DIR, matrix, metrics, resolve_features, save_json, seed_all

ROOT_DIR = Path(__file__).resolve().parent.parent
LOCK_FILE = ROOT_DIR / "data" / "locks" / "lgbm.lock"
GLOBAL_LOG_FILE = ROOT_DIR / "data" / "logs" / "lgbm_hpt_runs.jsonl"

_XTR: np.ndarray | None = None
_YTR: np.ndarray | None = None
_XVA: np.ndarray | None = None
_YVA: np.ndarray | None = None


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
    handle.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "label": label, "started_at": datetime.now().isoformat()}, ensure_ascii=False))
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


def parse_csv(text: str, cast) -> list:
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def make_candidates(count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out = []
    for cid in range(count):
        params = {
            "objective": "binary", "metric": "None", "boosting_type": "gbdt",
            "verbosity": -1, "deterministic": True, "force_col_wise": True,
            "seed": seed + cid, "feature_fraction_seed": seed + cid,
            "bagging_seed": seed + cid, "data_random_seed": seed + cid,
            "learning_rate": log_uniform(rng, 0.01, 0.15),
            "num_leaves": int(round(log_uniform(rng, 15, 255))),
            "max_depth": int(rng.choice([-1, 6, 8, 10, 12, 16])),
            "min_data_in_leaf": int(round(log_uniform(rng, 50, 2000))),
            "feature_fraction": float(rng.uniform(0.55, 1.0)),
            "bagging_fraction": float(rng.uniform(0.55, 1.0)), "bagging_freq": 1,
            "lambda_l1": log_uniform(rng, 1e-8, 10.0),
            "lambda_l2": log_uniform(rng, 1e-8, 10.0),
            "min_gain_to_split": float(rng.uniform(0.0, 1.0)),
            "max_bin": int(rng.choice([63, 127])),
            "scale_pos_weight": float(rng.choice([1.0, 2.0, 4.0])),
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


def subset_from_order(x, y, order, fraction):
    if fraction >= 1.0:
        return x, y
    idx = order[:max(2, int(round(len(order) * fraction)))]
    return np.ascontiguousarray(x[idx], dtype=np.float32), np.ascontiguousarray(y[idx], dtype=np.float32)


def train_booster(params, rounds, x_train, y_train, x_val, y_val, early_stopping):
    local = dict(params)
    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=True)
    callbacks = [lgb.log_evaluation(0)]
    if early_stopping > 0:
        local["metric"] = "binary_logloss"
        val_set = lgb.Dataset(x_val, label=y_val, reference=train_set, free_raw_data=True)
        callbacks.append(lgb.early_stopping(early_stopping, first_metric_only=True, verbose=False))
        model = lgb.train(local, train_set, num_boost_round=rounds, valid_sets=[val_set], valid_names=["validation"], callbacks=callbacks)
        best_iteration = int(model.best_iteration or rounds)
    else:
        local["metric"] = "None"
        model = lgb.train(local, train_set, num_boost_round=rounds, callbacks=callbacks)
        best_iteration = rounds
    return model, best_iteration


def evaluate(task):
    cid, params, rounds, threads, early_stopping = task
    if any(v is None for v in (_XTR, _YTR, _XVA, _YVA)):
        raise RuntimeError("Datos del worker no inicializados")
    local = dict(params)
    local["num_threads"] = threads
    started = time.time()
    model, best_iteration = train_booster(local, rounds, _XTR, _YTR, _XVA, _YVA, early_stopping)
    result = metrics(_YVA, model.predict(_XVA, num_iteration=best_iteration))
    return {"candidate_id": cid, "best_iteration": best_iteration, "duration_seconds": round(time.time() - started, 3), **{k: float(v) for k, v in result.items()}, "params_json": json.dumps(params, sort_keys=True)}


def save_records(records, path):
    frame = pd.DataFrame(records).sort_values("candidate_id")
    frame.to_csv(path, index=False)
    return frame


def main() -> None:
    cpus = available_cpus()
    parser = argparse.ArgumentParser(description="LightGBM parallel Successive Halving; objective PR-AUC")
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--feature-set", choices=["gain95", "all"], required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--candidates", "--trials", dest="candidates", type=int, default=500)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=16)
    parser.add_argument("--round-fractions", default="0.20,0.40,0.70,1.0")
    parser.add_argument("--round-rounds", default="128,320,700,1600")
    parser.add_argument("--survivors", default="100,25,8,1")
    parser.add_argument("--final-early-stopping", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    fractions = parse_csv(args.round_fractions, float)
    budgets = parse_csv(args.round_rounds, int)
    survivors = parse_csv(args.survivors, int)
    valid = args.candidates > 0 and 0 < args.val_size < 1 and args.workers > 0 and args.threads_per_worker > 0 and len(fractions) == len(budgets) == len(survivors) and fractions == sorted(fractions) and fractions[-1] == 1.0 and survivors[-1] == 1
    if not valid:
        raise ValueError("Configuración de Successive Halving inválida")
    requested = args.workers * args.threads_per_worker
    if requested > cpus:
        raise ValueError(f"Solicitas {requested} hilos, pero la reserva permite {cpus}")

    seed_all(args.seed)
    train_path = Path(args.train_parquet)
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    out = MODELS_DIR / "hpt" / args.experiment_name
    docs = DOCS_HPT_DIR / args.experiment_name
    if args.force and out.exists(): shutil.rmtree(out)
    if args.force and docs.exists(): shutil.rmtree(docs)
    out.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    acquire_lock(f"hpt:{args.experiment_name}")

    started = time.time()
    df = pd.read_parquet(train_path)
    if "target" not in df or df["target"].nunique() != 2:
        raise ValueError("target binario requerido")
    base_features, onehot, excluded = resolve_features(df, args.feature_set)
    x, feature_names = matrix(df, base_features, onehot)
    y = df["target"].to_numpy(np.float32)
    train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=args.val_size, random_state=args.seed, stratify=y)
    xtr_full = np.ascontiguousarray(x[train_idx], dtype=np.float32)
    xva_full = np.ascontiguousarray(x[val_idx], dtype=np.float32)
    ytr_full = np.ascontiguousarray(y[train_idx], dtype=np.float32)
    yva_full = np.ascontiguousarray(y[val_idx], dtype=np.float32)
    del x, train_idx, val_idx
    gc.collect()
    train_order = nested_order(ytr_full, args.seed + 100)
    val_order = nested_order(yva_full, args.seed + 200)

    definition = {"schema_version": 3, "search_method": "parallel_successive_halving_random_nested", "train_parquet": str(train_path.resolve()), "feature_set": args.feature_set, "candidates": args.candidates, "val_size": args.val_size, "seed": args.seed, "round_fractions": fractions, "round_rounds": budgets, "survivors": survivors, "final_early_stopping": args.final_early_stopping, "feature_columns": feature_names, "workers": args.workers, "threads_per_worker": args.threads_per_worker}
    definition_path = out / "run_definition.json"
    if definition_path.exists() and not args.force:
        old = json.load(open(definition_path, encoding="utf-8"))
        ignored = {"workers", "threads_per_worker"}
        if {k: v for k, v in old.items() if k not in ignored} != {k: v for k, v in definition.items() if k not in ignored}:
            raise RuntimeError("Experimento existente incompatible; use otro nombre o --force")
    save_json(definition_path, definition)

    print(f"[LGBM-HPT] Experimento={args.experiment_name} | feature_set={args.feature_set} | filas={len(df):,} | train={len(ytr_full):,} | val={len(yva_full):,} | features={len(feature_names)} | positivos={y.mean()*100:.2f}%")
    print(f"[LGBM-HPT] candidatos={args.candidates} | workers={args.workers} × {args.threads_per_worker} hilos | CPU={requested}/{cpus}")
    print("[LGBM-HPT] Muestras anidadas; objetivo PR-AUC; W4 no se utiliza.")

    candidates = make_candidates(args.candidates, args.seed)
    by_id = {c["candidate_id"]: c for c in candidates}
    active = list(by_id)
    all_frames = []
    global _XTR, _YTR, _XVA, _YVA

    for round_id, (fraction, rounds, keep) in enumerate(zip(fractions, budgets, survivors), start=1):
        result_path = out / f"round_{round_id}_results.csv"
        old_frame = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
        records = old_frame.to_dict("records") if not old_frame.empty else []
        completed = set(old_frame["candidate_id"].astype(int)) if not old_frame.empty else set()
        pending = [cid for cid in active if cid not in completed]
        _XTR, _YTR = subset_from_order(xtr_full, ytr_full, train_order, fraction)
        _XVA, _YVA = subset_from_order(xva_full, yva_full, val_order, fraction)
        early_stopping = args.final_early_stopping if round_id == len(fractions) else 0
        print(f"\n[LGBM-R{round_id}] candidatos={len(active)} | pendientes={len(pending)} | datos={fraction:.0%} | train={len(_YTR):,} | val={len(_YVA):,} | árboles={rounds} | sobreviven={min(keep, len(active))}", flush=True)

        if pending:
            tasks = [(cid, by_id[cid]["params"], rounds, args.threads_per_worker, early_stopping) for cid in pending]
            round_started = time.time()
            done = 0
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as executor:
                futures = {executor.submit(evaluate, task): task[0] for task in tasks}
                for future in as_completed(futures):
                    record = future.result()
                    record.update({"round": round_id, "data_fraction": fraction, "max_rounds": rounds})
                    records.append(record)
                    done += 1
                    save_records(records, result_path)
                    elapsed = time.time() - round_started
                    eta = elapsed / max(1, done) * (len(pending) - done)
                    print(f"[LGBM-R{round_id} {done:03d}/{len(pending):03d}] C{int(record['candidate_id']):04d} | PR-AUC={record['pr_auc']:.6f} | F1={record['f1']:.6f} | Lift@10={record['lift_at_10']:.4f} | iter={int(record['best_iteration'])} | {record['duration_seconds']:.1f}s | ETA={eta/60:.1f}m", flush=True)

        frame = save_records(records, result_path)
        frame = frame[frame["candidate_id"].astype(int).isin(active)].copy().sort_values(["pr_auc", "candidate_id"], ascending=[False, True])
        active = frame.head(min(keep, len(frame)))["candidate_id"].astype(int).tolist()
        all_frames.append(frame)
        save_json(out / f"round_{round_id}_survivors.json", {"round": round_id, "survivors": active, "best_pr_auc": float(frame.iloc[0]["pr_auc"])})
        print(f"[LGBM-R{round_id}] mejor=C{active[0]:04d} | PR-AUC={float(frame.iloc[0]['pr_auc']):.6f} | pasan={len(active)}")
        _XTR = _YTR = _XVA = _YVA = None
        gc.collect()

    winner = active[0]
    final_params = dict(by_id[winner]["params"])
    final_params["num_threads"] = args.threads_per_worker
    model, best_iteration = train_booster(final_params, budgets[-1], xtr_full, ytr_full, xva_full, yva_full, args.final_early_stopping)
    best_metrics = metrics(yva_full, model.predict(xva_full, num_iteration=best_iteration))
    final_hparams = {**final_params, "num_boost_round": best_iteration}
    save_json(out / "best_hparams.json", final_hparams)
    model.save_model(str(out / "best_model.txt"), num_iteration=best_iteration)
    all_results = pd.concat(all_frames, ignore_index=True)
    all_results.to_csv(out / "all_round_results.csv", index=False)
    all_frames[-1].sort_values("pr_auc", ascending=False).head(10).to_csv(out / "top10_trials.csv", index=False)

    summary = {"timestamp": datetime.now().isoformat(), "experiment_name": args.experiment_name, "model": "LightGBM", "search_method": "parallel_successive_halving_random_nested", "objective": "maximize_pr_auc_validation", "feature_set": args.feature_set, "train_parquet": str(train_path), "n_rows_total": len(df), "n_rows_train": len(ytr_full), "n_rows_validation": len(yva_full), "target_positive_rate_total": float(y.mean()), "split": {"type": "row_stratified", "validation_size": args.val_size, "seed": args.seed}, "base_feature_columns": base_features, "model_feature_columns": feature_names, "excluded_columns": excluded, "cycle_onehot": onehot, "candidate_count": args.candidates, "best_candidate_id": winner, "best_trial_number": winner, "best_params": final_hparams, "best_metrics_validation": best_metrics, "round_fractions": fractions, "round_rounds": budgets, "survivors": survivors, "workers": args.workers, "threads_per_worker": args.threads_per_worker, "duration_seconds": round(time.time() - started, 2)}
    save_json(out / "best_metrics.json", summary)
    save_json(docs / "run_summary.json", summary)
    save_json(docs / "best_hparams.json", final_hparams)
    GLOBAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print("\n[LGBM-HPT] ================================================================")
    print(f"[LGBM-HPT] Mejor candidato C{winner:04d} | PR-AUC={best_metrics['pr_auc']:.6f} | ROC-AUC={best_metrics['roc_auc']:.6f} | F1={best_metrics['f1']:.6f} | Lift@10={best_metrics['lift_at_10']:.4f}")
    print(f"[LGBM-HPT] Iteraciones finales={best_iteration} | threshold={best_metrics['threshold_f1']:.6f}")
    print(f"[LGBM-HPT] Artefactos: {out}")
    print(f"[LGBM-HPT] Resumen rastreable: {docs / 'run_summary.json'}")
    print("[LGBM-HPT] W4/test NO se utilizó durante el tuning.")
    print("[LGBM-HPT] ================================================================")


if __name__ == "__main__":
    main()
