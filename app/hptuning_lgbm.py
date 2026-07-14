"""Tuning reproducible de LightGBM para recompra binaria.

Usa el mismo split estratificado, conjuntos de features y métricas que la FNN:
PR-AUC como objetivo; ROC-AUC, F1, precision, recall y Lift@10 como métricas
complementarias. W4/test no participa en el tuning.
"""
from __future__ import annotations

import argparse
import atexit
import fcntl
import json
import os
import shutil
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from config import MODELS_DIR
from hptuning import DOCS_HPT_DIR, matrix, metrics, resolve_features, save_json, seed_all

ROOT_DIR = Path(__file__).resolve().parent.parent
LOCK_FILE = ROOT_DIR / "data" / "locks" / "lgbm.lock"
GLOBAL_LOG_FILE = ROOT_DIR / "data" / "logs" / "lgbm_hpt_runs.jsonl"


def acquire_lock(label: str):
    """Impide dos ejecuciones LightGBM simultáneas."""
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


def pr_auc_eval(predictions: np.ndarray, dataset: lgb.Dataset):
    value = float(average_precision_score(dataset.get_label(), predictions))
    return "pr_auc", value, True


def trial_parameters(trial: optuna.Trial, threads: int, seed: int) -> tuple[dict[str, Any], int]:
    num_boost_round = trial.suggest_int("num_boost_round", 300, 2500, log=True)
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "None",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": threads,
        "deterministic": True,
        "force_col_wise": True,
        "seed": seed + trial.number,
        "feature_fraction_seed": seed + trial.number,
        "bagging_seed": seed + trial.number,
        "data_random_seed": seed + trial.number,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 6, 8, 10, 12, 16]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 2000, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.55, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.55, 1.0),
        "bagging_freq": 1,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
        "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
        "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1.0, 2.0, 4.0]),
    }
    return params, num_boost_round


def train_booster(
    params: dict[str, Any],
    num_boost_round: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int,
) -> lgb.Booster:
    train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    val_set = lgb.Dataset(x_val, label=y_val, reference=train_set, free_raw_data=False)
    callbacks = [lgb.log_evaluation(0)]
    if early_stopping_rounds > 0:
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                first_metric_only=True,
                verbose=False,
            )
        )
    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        valid_names=["validation"],
        feval=pr_auc_eval,
        callbacks=callbacks,
    )


def plot_history(trials: pd.DataFrame, path: Path) -> None:
    complete = trials[trials["state"] == "COMPLETE"].copy()
    if complete.empty:
        return
    complete = complete.sort_values("number")
    values = complete["value"].astype(float).to_numpy()
    best = np.maximum.accumulate(values)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(complete["number"], values, s=18, alpha=0.45, label="Trial")
    ax.plot(complete["number"], best, linewidth=2, label="Mejor acumulado")
    ax.set(xlabel="Trial", ylabel="PR-AUC validación", title="Optimización LightGBM")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tuning LightGBM con objetivo PR-AUC")
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--feature-set", choices=["gain95", "all"], required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.trials <= 0 or args.threads <= 0 or not 0 < args.val_size < 1:
        raise ValueError("Parámetros inválidos")
    seed_all(args.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    x_train = np.ascontiguousarray(x[train_idx], dtype=np.float32)
    x_val = np.ascontiguousarray(x[val_idx], dtype=np.float32)
    y_train = np.ascontiguousarray(y[train_idx], dtype=np.float32)
    y_val = np.ascontiguousarray(y[val_idx], dtype=np.float32)
    del x, train_idx, val_idx

    definition = {
        "train_parquet": str(train_path.resolve()),
        "feature_set": args.feature_set,
        "trials": args.trials,
        "val_size": args.val_size,
        "seed": args.seed,
        "threads": args.threads,
        "early_stopping_rounds": args.early_stopping_rounds,
        "feature_columns": feature_names,
    }
    definition_path = out / "run_definition.json"
    if definition_path.exists() and not args.force:
        old = json.load(open(definition_path, encoding="utf-8"))
        immutable = {k: v for k, v in definition.items() if k != "trials"}
        old_immutable = {k: v for k, v in old.items() if k != "trials"}
        if old_immutable != immutable:
            raise RuntimeError("Experimento existente incompatible; use otro nombre o --force")
    save_json(definition_path, definition)

    print(
        f"[LGBM-HPT] Experimento={args.experiment_name} | feature_set={args.feature_set} "
        f"| filas={len(df):,} | train={len(y_train):,} | val={len(y_val):,} "
        f"| features={len(feature_names)} | positivos={y.mean()*100:.2f}%"
    )
    print(
        f"[LGBM-HPT] Objetivo=PR-AUC | trials objetivo={args.trials} "
        f"| threads={args.threads} | early_stopping={args.early_stopping_rounds}"
    )

    storage = f"sqlite:///{(out / 'study.db').resolve()}"
    study = optuna.create_study(
        study_name=args.experiment_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=TPESampler(seed=args.seed, multivariate=True),
    )
    existing = len(study.trials)
    remaining = max(0, args.trials - existing)
    study_started = time.time()

    def objective(trial: optuna.Trial) -> float:
        trial_started = time.time()
        params, rounds = trial_parameters(trial, args.threads, args.seed)
        booster = train_booster(
            params,
            rounds,
            x_train,
            y_train,
            x_val,
            y_val,
            args.early_stopping_rounds,
        )
        best_iteration = int(booster.best_iteration or rounds)
        probabilities = booster.predict(x_val, num_iteration=best_iteration)
        result = metrics(y_val, probabilities)
        trial.set_user_attr("best_iteration", best_iteration)
        trial.set_user_attr("duration_seconds", time.time() - trial_started)
        for key, value in result.items():
            trial.set_user_attr(key, float(value))
        return float(result["pr_auc"])

    def after_trial(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        completed = len(study_.trials)
        local_done = max(1, completed - existing)
        elapsed = time.time() - study_started
        eta = elapsed / local_done * max(0, args.trials - completed)
        attrs = trial.user_attrs
        if trial.value is None:
            print(
                f"[LGBM-HPT {completed:04d}/{args.trials:04d}] "
                f"T{trial.number:04d} | estado={trial.state.name} | ETA={eta/60:.1f}m",
                flush=True,
            )
        else:
            print(
                f"[LGBM-HPT {completed:04d}/{args.trials:04d}] "
                f"T{trial.number:04d} | PR-AUC={float(trial.value):.6f} "
                f"| F1={attrs.get('f1', float('nan')):.6f} "
                f"| Lift@10={attrs.get('lift_at_10', float('nan')):.4f} "
                f"| iter={attrs.get('best_iteration', 0)} "
                f"| tiempo={attrs.get('duration_seconds', 0):.1f}s "
                f"| ETA={eta/60:.1f}m",
                flush=True,
            )
        study_.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs")).to_csv(
            out / "all_trials.csv", index=False
        )

    if remaining:
        print(f"[LGBM-HPT] Reanudación: existentes={existing} | faltan={remaining}")
        study.optimize(objective, n_trials=remaining, callbacks=[after_trial], gc_after_trial=True)
    else:
        print("[LGBM-HPT] El número objetivo de trials ya está completo.")

    best_trial = study.best_trial
    best_params, sampled_rounds = trial_parameters(
        optuna.trial.FixedTrial(best_trial.params, number=best_trial.number),
        args.threads,
        args.seed,
    )
    best_iteration = int(best_trial.user_attrs.get("best_iteration", sampled_rounds))
    best_booster = train_booster(
        best_params,
        sampled_rounds,
        x_train,
        y_train,
        x_val,
        y_val,
        args.early_stopping_rounds,
    )
    best_iteration = int(best_booster.best_iteration or best_iteration)
    probabilities = best_booster.predict(x_val, num_iteration=best_iteration)
    best_metrics = metrics(y_val, probabilities)

    final_hparams = {
        **best_params,
        "num_boost_round": best_iteration,
    }
    save_json(out / "best_hparams.json", final_hparams)
    best_booster.save_model(str(out / "best_model.txt"), num_iteration=best_iteration)
    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    trials_df.to_csv(out / "all_trials.csv", index=False)
    top10 = trials_df[trials_df["state"] == "COMPLETE"].sort_values("value", ascending=False).head(10)
    top10.to_csv(out / "top10_trials.csv", index=False)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": args.experiment_name,
        "model": "LightGBM",
        "search_method": "optuna_tpe",
        "objective": "maximize_pr_auc_validation",
        "feature_set": args.feature_set,
        "train_parquet": str(train_path),
        "n_rows_total": len(df),
        "n_rows_train": len(y_train),
        "n_rows_validation": len(y_val),
        "target_positive_rate_total": float(y.mean()),
        "split": {"type": "row_stratified", "validation_size": args.val_size, "seed": args.seed},
        "base_feature_columns": base_features,
        "model_feature_columns": feature_names,
        "excluded_columns": excluded,
        "cycle_onehot": onehot,
        "trial_count": len(study.trials),
        "best_trial_number": best_trial.number,
        "best_params": final_hparams,
        "best_metrics_validation": best_metrics,
        "duration_seconds": round(time.time() - started, 2),
        "threads": args.threads,
        "early_stopping_rounds": args.early_stopping_rounds,
    }
    save_json(out / "best_metrics.json", summary)
    save_json(plots / "run_summary.json", summary)
    save_json(plots / "best_hparams.json", final_hparams)

    GLOBAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    plot_history(trials_df, plots / "optimization_history.png")
    precision, recall, _ = precision_recall_curve(y_val, probabilities)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label="LightGBM")
    ax.axhline(float(y_val.mean()), linestyle="--", label="Prevalencia")
    ax.set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall — validación")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "best_pr_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\n[LGBM-HPT] ================================================================")
    print(
        f"[LGBM-HPT] Mejor trial T{best_trial.number:04d} "
        f"| PR-AUC={best_metrics['pr_auc']:.6f} "
        f"| ROC-AUC={best_metrics['roc_auc']:.6f} "
        f"| F1={best_metrics['f1']:.6f} "
        f"| Lift@10={best_metrics['lift_at_10']:.4f}"
    )
    print(f"[LGBM-HPT] Iteraciones finales={best_iteration} | threshold={best_metrics['threshold_f1']:.6f}")
    print(f"[LGBM-HPT] Artefactos: {out}")
    print(f"[LGBM-HPT] Resumen rastreable: {plots / 'run_summary.json'}")
    print("[LGBM-HPT] W4/test NO se utilizó durante el tuning.")
    print("[LGBM-HPT] ================================================================")


if __name__ == "__main__":
    main()
