"""Entrena LightGBM en W0-W3 y evalúa una sola vez en W4."""
from __future__ import annotations

import argparse, atexit, fcntl, json, math, os, shutil, socket, time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score)

from config import MODELS_DIR
from hptuning import matrix, resolve_features, save_json, seed_all

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "final"
LOCK = ROOT / "data" / "locks" / "lgbm.lock"
GLOBAL_LOG = ROOT / "data" / "logs" / "lgbm_final_runs.jsonl"


def acquire_lock(label: str):
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    h = open(LOCK, "a+", encoding="utf-8")
    try:
        fcntl.flock(h.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        h.seek(0); owner = h.read().strip() or "proceso desconocido"; h.close()
        raise RuntimeError(f"Ya existe una ejecución LightGBM activa: {owner}") from exc
    h.seek(0); h.truncate(); h.write(json.dumps({"pid": os.getpid(),
        "host": socket.gethostname(), "label": label,
        "started_at": datetime.now().isoformat()}, ensure_ascii=False)); h.flush()
    def release():
        try: fcntl.flock(h.fileno(), fcntl.LOCK_UN); h.close()
        except Exception: pass
    atexit.register(release)
    return h


def load_hpt(path: Path):
    params_file, metrics_file = path / "best_hparams.json", path / "best_metrics.json"
    if not params_file.exists() or not metrics_file.exists():
        raise FileNotFoundError(f"Faltan artefactos HPT en {path}")
    params = json.load(open(params_file, encoding="utf-8"))
    summary = json.load(open(metrics_file, encoding="utf-8"))
    threshold = float(summary.get("best_metrics_validation", {}).get("threshold_f1", .5))
    return params, summary, threshold


def classification_metrics(y, p, threshold):
    pred = (p >= threshold).astype(np.int8); base = float(y.mean())
    n = max(1, int(math.ceil(len(p) * .10))); top = np.argpartition(p, -n)[-n:]
    top_rate = float(y[top].mean())
    return {"pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "threshold_frozen_from_validation": float(threshold),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "lift_at_10": top_rate / base if base else 0.0,
        "base_rate": base, "top10_positive_rate": top_rate}


def topk_metrics(df, probabilities, k=3):
    required = {"nucleo", "COD_SUBCATEGORIA", "target"}
    if not required.issubset(df.columns):
        return {"precision_at_3": 0., "recall_at_3": 0., "hit_rate_at_3": 0.,
            "n_groups_evaluated": 0, "note": f"Faltan: {sorted(required-set(df.columns))}"}
    w = df[["nucleo", "COD_SUBCATEGORIA", "target"]].copy(); w["proba"] = probabilities
    w = w.groupby(["nucleo", "COD_SUBCATEGORIA"], sort=False, observed=True).agg(
        target=("target", "max"), proba=("proba", "max")).reset_index()
    g = w.groupby("nucleo", sort=False, observed=True)
    w["size"] = g.target.transform("size"); w["pos"] = g.target.transform("sum")
    w["rank"] = g.proba.rank(method="first", ascending=False)
    eligible = (w["size"] >= k) & (w["pos"] > 0)
    den = w.loc[eligible, ["nucleo", "pos"]].drop_duplicates("nucleo").set_index("nucleo").pos
    if den.empty:
        return {"precision_at_3": 0., "recall_at_3": 0., "hit_rate_at_3": 0., "n_groups_evaluated": 0}
    hits = w.loc[eligible & (w["rank"] <= k)].groupby("nucleo", observed=True).target.sum()
    hits = hits.reindex(den.index, fill_value=0).astype(float)
    return {"precision_at_3": float((hits/k).mean()),
        "recall_at_3": float((hits/den).mean()),
        "hit_rate_at_3": float((hits > 0).mean()), "n_groups_evaluated": int(len(den))}


def progress(total, every, rows):
    started = time.time()
    def callback(env):
        i = env.iteration + 1
        if i != 1 and i % every and i != total: return
        elapsed = time.time() - started; eta = elapsed / i * (total-i)
        rows.append({"iteration": i, "elapsed_seconds": round(elapsed, 3), "eta_seconds": round(eta, 3)})
        print(f"[LGBM-FINAL][FIT] {i}/{total} | {elapsed/60:.1f}m | ETA={eta/60:.1f}m", flush=True)
    callback.order = 10; callback.before_iteration = False
    return callback


def stage(name, started, times):
    elapsed = round(time.time()-started, 3); times[name] = elapsed
    print(f"[LGBM-FINAL] {name}: {elapsed:.1f}s", flush=True)


def plots(y, p, importance, out):
    precision, recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(7, 6)); ax.plot(recall, precision, label="LightGBM")
    ax.axhline(float(y.mean()), linestyle="--", label="Prevalencia")
    ax.set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall — W4")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/"external_pr_curve.png", dpi=160); plt.close(fig)
    top = importance.sort_values("gain").tail(20)
    fig, ax = plt.subplots(figsize=(9, max(5, len(top)*.32))); ax.barh(top.feature, top.gain)
    ax.set(xlabel="Gain", title="LightGBM — importancia final"); ax.grid(alpha=.25, axis="x")
    fig.tight_layout(); fig.savefig(out/"feature_importance_gain.png", dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-parquet", required=True); p.add_argument("--eval-parquet", required=True)
    p.add_argument("--feature-set", choices=["gain95", "all"], required=True)
    p.add_argument("--hpt-dir", required=True); p.add_argument("--experiment-name", required=True)
    p.add_argument("--threads", type=int, default=16); p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int); p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--full-predictions", action="store_true"); p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.threads <= 0 or a.log_every <= 0: raise ValueError("threads y log-every deben ser positivos")
    if a.threads > 64: print("[LGBM-FINAL] ADVERTENCIA: use 16 o 32 hilos para un solo modelo", flush=True)

    train_path, eval_path, hpt_dir = Path(a.train_parquet), Path(a.eval_parquet), Path(a.hpt_dir)
    for path in (train_path, eval_path, hpt_dir):
        if not path.exists(): raise FileNotFoundError(path)
    out, docs = MODELS_DIR/"final"/a.experiment_name, DOCS/a.experiment_name
    if out.exists() and not a.force: raise FileExistsError(f"Ya existe {out}. Use otro nombre o --force")
    if a.force and out.exists(): shutil.rmtree(out)
    if a.force and docs.exists(): shutil.rmtree(docs)
    out.mkdir(parents=True, exist_ok=True); docs.mkdir(parents=True, exist_ok=True)
    acquire_lock(f"final:{a.experiment_name}")

    params, hpt, threshold = load_hpt(hpt_dir)
    if hpt.get("feature_set") not in (None, a.feature_set): raise ValueError("HPT y feature-set no coinciden")
    rounds = int(params.pop("num_boost_round")); seed = int(a.seed if a.seed is not None else params.get("seed", 42))
    params.update(num_threads=a.threads, seed=seed, feature_fraction_seed=seed,
        bagging_seed=seed, data_random_seed=seed); seed_all(seed)
    total_start = time.time(); times, progress_rows = {}, []

    t = time.time(); print("[LGBM-FINAL] Cargando datos...", flush=True)
    train_df, eval_df = pd.read_parquet(train_path), pd.read_parquet(eval_path)
    for label, df in (("train", train_df), ("eval", eval_df)):
        if "target" not in df or df.target.nunique() != 2: raise ValueError(f"{label}: target binario requerido")
    stage("carga_datos", t, times)

    t = time.time(); print("[LGBM-FINAL] Construyendo matrices...", flush=True)
    base, onehot, excluded = resolve_features(train_df, a.feature_set)
    x_train, names = matrix(train_df, base, onehot); x_eval, eval_names = matrix(eval_df, base, onehot)
    if names != eval_names or (hpt.get("model_feature_columns") and names != hpt["model_feature_columns"]):
        raise RuntimeError("Las features no coinciden con el tuning")
    y_train, y_eval = train_df.target.to_numpy(np.float32), eval_df.target.to_numpy(np.float32)
    stage("matrices", t, times)
    print(f"[LGBM-FINAL] {a.experiment_name} | train={len(train_df):,} | eval={len(eval_df):,} | features={len(names)}")
    print(f"[LGBM-FINAL] rounds={rounds} | threshold={threshold:.6f} | threads={a.threads}")

    t = time.time(); print("[LGBM-FINAL] Entrenando...", flush=True)
    booster = lgb.train(params, lgb.Dataset(x_train, label=y_train, feature_name=names),
        num_boost_round=rounds, callbacks=[progress(rounds, a.log_every, progress_rows)])
    stage("entrenamiento", t, times)

    t = time.time(); print("[LGBM-FINAL] Prediciendo W4...", flush=True)
    probabilities = booster.predict(x_eval, num_iteration=rounds, num_threads=a.threads)
    stage("prediccion_w4", t, times)

    t = time.time(); print("[LGBM-FINAL] Calculando métricas...", flush=True)
    eval_metrics = classification_metrics(y_eval, probabilities, threshold); ranking = topk_metrics(eval_df, probabilities)
    stage("metricas", t, times)

    t = time.time(); print("[LGBM-FINAL] Guardando artefactos...", flush=True)
    importance = pd.DataFrame({"feature": names, "gain": booster.feature_importance("gain"),
        "split": booster.feature_importance("split")}).sort_values("gain", ascending=False)
    total_gain = float(importance.gain.sum()); importance["gain_pct"] = importance.gain/total_gain if total_gain else 0.
    importance.to_csv(out/"feature_importance.csv", index=False); importance.to_csv(docs/"feature_importance.csv", index=False)
    booster.save_model(str(out/"model.txt"), num_iteration=rounds)
    pd.DataFrame(progress_rows).to_csv(out/"training_progress.csv", index=False)
    pd.DataFrame(progress_rows).to_csv(docs/"training_progress.csv", index=False)
    plots(y_eval, probabilities, importance, docs)
    if a.save_predictions:
        pred = eval_df.copy() if a.full_predictions else eval_df[[c for c in ("nucleo","COD_SUBCATEGORIA","target") if c in eval_df]].copy()
        pred["proba_compra"] = probabilities; pred.to_parquet(out/"eval_predictions.parquet", index=False)
    stage("artefactos", t, times)

    summary = {"timestamp": datetime.now().isoformat(), "experiment_name": a.experiment_name,
        "model": "LightGBM", "feature_set": a.feature_set,
        "training_protocol": "fit_all_W0_W3_then_single_external_W4_evaluation",
        "train_parquet": str(train_path), "eval_parquet": str(eval_path), "hpt_dir": str(hpt_dir),
        "hyperparameters": {**params, "num_boost_round": rounds}, "seed": seed,
        "feature_columns": names, "base_feature_columns": base, "excluded_columns": excluded,
        "cycle_onehot": onehot, "n_train": len(train_df), "n_eval": len(eval_df),
        "train_positive_rate": float(y_train.mean()), "eval_positive_rate": float(y_eval.mean()),
        "threshold_source": "best_validation_trial_threshold_f1", "frozen_threshold": threshold,
        "external_eval_metrics": eval_metrics, "top3_metrics": ranking,
        "duration_seconds": round(time.time()-total_start, 2), "stage_times_seconds": times,
        "threads": a.threads, "prediction_file_mode": "full" if a.full_predictions else "minimal" if a.save_predictions else "not_saved",
        "hpt_validation_summary": {"best_trial_number": hpt.get("best_trial_number"),
            "best_metrics_validation": hpt.get("best_metrics_validation")}}
    save_json(out/"run_summary.json", summary); save_json(docs/"run_summary.json", summary)
    save_json(docs/"eval_metrics.json", {**eval_metrics, **ranking})
    GLOBAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_LOG, "a", encoding="utf-8") as f: f.write(json.dumps(summary, ensure_ascii=False)+"\n")

    print("\n[LGBM-FINAL] ================================================================")
    print(f"[LGBM-FINAL] PR-AUC={eval_metrics['pr_auc']:.6f} | ROC-AUC={eval_metrics['roc_auc']:.6f} | F1={eval_metrics['f1']:.6f}")
    print(f"[LGBM-FINAL] Precision={eval_metrics['precision']:.6f} | Recall={eval_metrics['recall']:.6f} | Lift@10={eval_metrics['lift_at_10']:.4f}")
    print(f"[LGBM-FINAL] Precision@3={ranking['precision_at_3']:.6f} | Recall@3={ranking['recall_at_3']:.6f} | HitRate@3={ranking['hit_rate_at_3']:.6f}")
    print(f"[LGBM-FINAL] Tiempo total={summary['duration_seconds']/60:.1f}m")
    print(f"[LGBM-FINAL] Artefactos: {out}\n[LGBM-FINAL] Resumen: {docs/'run_summary.json'}")
    print("[LGBM-FINAL] ================================================================")


if __name__ == "__main__": main()
