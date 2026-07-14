"""Diagnóstico pareado FNN vs LightGBM GAIN95 sobre validación HPT.

Reconstruye exactamente el split estratificado 80/20 de W0-W3, carga los
modelos HPT ya entrenados, calcula métricas globales y por clase, matrices de
confusión, curvas y complementariedad de errores. No reentrena y no usa W4.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from hptuning import matrix, resolve_features, save_json, seed_all
from train_fnn import FeatureScaler, PurchaseFNN

ROOT = Path(__file__).resolve().parent.parent
HPT_ROOT = ROOT / "data" / "models" / "hpt"
DATA_ROOT = ROOT / "data" / "analysis" / "model_comparison"
DOCS_ROOT = ROOT / "docs" / "model_comparison"
GLOBAL_LOG = ROOT / "data" / "logs" / "model_comparison_runs.jsonl"


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def score(summary: dict[str, Any]) -> float:
    aggregate = summary.get("confirmation", {}).get("winner_aggregate_metrics", {})
    if aggregate.get("mean_pr_auc") is not None:
        return float(aggregate["mean_pr_auc"])
    for key in ("best_single_seed_metrics_validation", "best_metrics_validation"):
        value = summary.get(key, {}).get("pr_auc")
        if value is not None:
            return float(value)
    return float("-inf")


def compatible(path: Path, model: str) -> tuple[bool, dict[str, Any] | None]:
    metrics_file = path / "best_metrics.json"
    if not metrics_file.exists():
        return False, None
    try:
        summary = load_json(metrics_file)
    except Exception:
        return False, None
    if str(summary.get("model", "")).upper() != model.upper() or summary.get("feature_set") != "gain95":
        return False, summary
    required = [path / "best_hparams.json"]
    if model.upper() == "FNN":
        required += [path / "best_model_state.pth", path / "scaler.json"]
    else:
        required += [path / "best_model.txt"]
    return all(item.exists() for item in required), summary


def discover(explicit: str | None, model: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    if explicit:
        path = Path(explicit)
        ok, summary = compatible(path, model)
        if not ok or summary is None:
            raise RuntimeError(f"Artefactos incompletos o incompatibles para {model}: {path}")
        return path, summary, [{"path": str(path), "score": score(summary), "selected": True}]

    found: list[tuple[tuple[int, float, str], Path, dict[str, Any]]] = []
    audit: list[dict[str, Any]] = []
    for metrics_file in HPT_ROOT.glob("*/best_metrics.json"):
        path = metrics_file.parent
        ok, summary = compatible(path, model)
        if not ok or summary is None:
            continue
        method = str(summary.get("search_method", ""))
        robust = int(
            (model.upper() == "FNN" and ("multiseed" in method.lower() or bool(summary.get("confirmation"))))
            or (model.upper() == "LIGHTGBM" and "parallel_successive_halving" in method.lower())
        )
        row = {"path": str(path), "robust_preference": robust, "score": score(summary),
               "search_method": method, "timestamp": str(summary.get("timestamp", "")), "selected": False}
        audit.append(row)
        found.append(((robust, row["score"], row["timestamp"]), path, summary))
    if not found:
        raise FileNotFoundError(f"No se encontraron artefactos {model}/gain95 en {HPT_ROOT}")
    found.sort(key=lambda item: item[0], reverse=True)
    _, path, summary = found[0]
    for row in audit:
        row["selected"] = Path(row["path"]).resolve() == path.resolve()
    return path, summary, audit


def threshold(summary: dict[str, Any], model: str) -> float:
    if model.upper() == "FNN":
        direct = summary.get("best_single_seed_metrics_validation", {}).get("threshold_f1")
        if direct is not None:
            return float(direct)
        best_seed = summary.get("best_seed")
        runs = summary.get("confirmation", {}).get("winner_seed_metrics", [])
        for row in runs:
            if best_seed is not None and int(row.get("seed", -1)) == int(best_seed):
                return float(row.get("threshold_f1", 0.5))
        values = [float(row["threshold_f1"]) for row in runs if row.get("threshold_f1") is not None]
        if values:
            return float(np.median(values))
    value = summary.get("best_metrics_validation", {}).get("threshold_f1")
    return float(value if value is not None else 0.5)


def predict_fnn(path: Path, x: np.ndarray, names: list[str], device: torch.device) -> tuple[np.ndarray, dict]:
    params = load_json(path / "best_hparams.json")
    scaler = FeatureScaler.load(path / "scaler.json")
    x = scaler.transform(x, names).astype(np.float32, copy=False)
    model = PurchaseFNN(len(names), params["hidden_dims"], float(params["dropout"]), params["activation"]).to(device)
    try:
        state = torch.load(path / "best_model_state.pth", map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path / "best_model_state.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    batch = max(int(params.get("batch_size", 65536)), 262144 if device.type == "cuda" else 65536)
    parts = []
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[start:start + batch])).to(device)
            parts.append(torch.sigmoid(model(xb)).float().cpu().numpy())
    return np.concatenate(parts).ravel(), params


def predict_lgbm(path: Path, x: np.ndarray, threads: int) -> tuple[np.ndarray, dict]:
    params = load_json(path / "best_hparams.json")
    booster = lgb.Booster(model_file=str(path / "best_model.txt"))
    rounds = int(params.get("num_boost_round", booster.current_iteration()))
    return np.asarray(booster.predict(x, num_iteration=rounds, num_threads=threads)), params


def evaluate(y: np.ndarray, p: np.ndarray, cut: float, name: str) -> tuple[dict, pd.DataFrame, np.ndarray]:
    pred = (p >= cut).astype(np.int8)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    report = classification_report(
        y, pred, labels=[0, 1], target_names=["target_0_no_compra", "target_1_compra"],
        output_dict=True, zero_division=0,
    )
    base = float(y.mean())
    n_top = max(1, int(math.ceil(len(p) * 0.10)))
    top = np.argpartition(p, -n_top)[-n_top:]
    top_rate = float(y[top].mean())
    clipped = np.clip(p, 1e-8, 1 - 1e-8)
    values = {
        "model": name, "threshold": float(cut), "n_validation": int(len(y)),
        "support_target_0": int((y == 0).sum()), "support_target_1": int((y == 1).sum()),
        "predicted_target_0": int((pred == 0).sum()), "predicted_target_1": int((pred == 1).sum()),
        "predicted_positive_rate": float(pred.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])), "brier_score": float(brier_score_loss(y, p)),
        "precision_target_0": float(report["target_0_no_compra"]["precision"]),
        "recall_target_0_specificity": float(report["target_0_no_compra"]["recall"]),
        "f1_target_0": float(report["target_0_no_compra"]["f1-score"]),
        "precision_target_1": float(precision_score(y, pred, zero_division=0)),
        "recall_target_1_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "f1_target_1": float(f1_score(y, pred, zero_division=0)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "base_rate": base, "top10_positive_rate": top_rate,
        "lift_at_10": top_rate / base if base else 0.0,
        "recall_at_top10": float(y[top].sum() / max(1, y.sum())),
    }
    return values, pd.DataFrame(report).T.reset_index(names="class_or_average"), cm


def top3(df: pd.DataFrame, p: np.ndarray, k: int = 3) -> tuple[dict, pd.DataFrame]:
    work = df[["nucleo", "COD_SUBCATEGORIA", "target"]].copy()
    work["proba"] = p
    work = work.groupby(["nucleo", "COD_SUBCATEGORIA"], sort=False, observed=True).agg(
        target=("target", "max"), proba=("proba", "max")
    ).reset_index()
    group = work.groupby("nucleo", sort=False, observed=True)
    work["size"] = group.target.transform("size")
    work["positives"] = group.target.transform("sum")
    work["rank"] = group.proba.rank(method="first", ascending=False)
    eligible = (work["size"] >= k) & (work["positives"] > 0)
    den = work.loc[eligible, ["nucleo", "positives"]].drop_duplicates("nucleo").set_index("nucleo").positives
    selected = work.loc[eligible & (work["rank"] <= k), ["nucleo", "COD_SUBCATEGORIA", "target", "proba", "rank"]]
    if den.empty:
        return {"precision_at_3": 0.0, "recall_at_3": 0.0, "hit_rate_at_3": 0.0, "n_groups_evaluated": 0}, selected
    hits = selected.groupby("nucleo", observed=True).target.sum().reindex(den.index, fill_value=0).astype(float)
    return {"precision_at_3": float((hits / k).mean()), "recall_at_3": float((hits / den).mean()),
            "hit_rate_at_3": float((hits > 0).mean()), "n_groups_evaluated": int(len(den))}, selected


def confusion_figure(cm: np.ndarray, title: str, path: Path, normalized: bool) -> None:
    values = cm.astype(float)
    if normalized:
        den = values.sum(axis=1, keepdims=True)
        values = np.divide(values, den, out=np.zeros_like(values), where=den > 0)
    fig, ax = plt.subplots(figsize=(6.3, 5.3))
    image = ax.imshow(values); fig.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], ["Predice 0", "Predice 1"]); ax.set_yticks([0, 1], ["Real 0", "Real 1"])
    ax.set(xlabel="Predicción", ylabel="Real", title=title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{values[i,j]:.3f}" if normalized else f"{int(values[i,j]):,}", ha="center", va="center")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def save_both(frame: pd.DataFrame, name: str, data: Path, docs: Path) -> None:
    frame.to_csv(data / name, index=False); frame.to_csv(docs / name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico FNN vs LightGBM gain95 sobre validación HPT")
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--fnn-hpt-dir", default=None)
    parser.add_argument("--lgbm-hpt-dir", default=None)
    parser.add_argument("--experiment-name", default="fnn_lgbm_gain95_validation_diagnostics")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0:
        raise ValueError("--threads debe ser positivo")
    device_name = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible")
    device = torch.device(device_name)
    torch.set_num_threads(max(1, min(args.threads, os.cpu_count() or args.threads)))
    seed_all(42)

    train_path = Path(args.train_parquet)
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    data = DATA_ROOT / args.experiment_name; docs = DOCS_ROOT / args.experiment_name
    if (data.exists() or docs.exists()) and not args.force:
        raise FileExistsError(f"Ya existe {args.experiment_name}; use --force o cambie el nombre")
    if args.force and data.exists(): shutil.rmtree(data)
    if args.force and docs.exists(): shutil.rmtree(docs)
    data.mkdir(parents=True, exist_ok=True); docs.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print("[COMPARE] Localizando artefactos HPT gain95...", flush=True)
    fnn_dir, fnn_summary, fnn_audit = discover(args.fnn_hpt_dir, "FNN")
    lgbm_dir, lgbm_summary, lgbm_audit = discover(args.lgbm_hpt_dir, "LightGBM")
    print(f"[COMPARE] FNN: {fnn_dir}\n[COMPARE] LightGBM: {lgbm_dir}", flush=True)
    split_fnn = fnn_summary.get("split", {}); split_lgbm = lgbm_summary.get("split", {})
    seed = int(split_fnn.get("seed", 42)); val_size = float(split_fnn.get("validation_size", 0.20))
    if seed != int(split_lgbm.get("seed", 42)) or not math.isclose(val_size, float(split_lgbm.get("validation_size", 0.20))):
        raise RuntimeError("Los artefactos no comparten el mismo split")

    print("[COMPARE] Reconstruyendo validación W0-W3...", flush=True)
    df = pd.read_parquet(train_path)
    base, onehot, _ = resolve_features(df, "gain95"); raw, names = matrix(df, base, onehot)
    y_all = df.target.to_numpy(np.int8)
    _, val_idx = train_test_split(np.arange(len(df)), test_size=val_size, random_state=seed, stratify=y_all)
    x_val = np.ascontiguousarray(raw[val_idx], dtype=np.float32); y = np.ascontiguousarray(y_all[val_idx])
    val_df = df.iloc[val_idx].copy().reset_index(names="source_row_index")
    if fnn_summary.get("model_feature_columns") and names != fnn_summary["model_feature_columns"]:
        raise RuntimeError("Features FNN no coinciden")
    if lgbm_summary.get("model_feature_columns") and names != lgbm_summary["model_feature_columns"]:
        raise RuntimeError("Features LightGBM no coinciden")
    print(f"[COMPARE] total={len(df):,} | val={len(y):,} | positivos={y.mean()*100:.2f}% | features={len(names)}", flush=True)

    t = time.time(); print(f"[COMPARE] Inferencia FNN en {device}...", flush=True)
    p_fnn, hp_fnn = predict_fnn(fnn_dir, x_val, names, device); print(f"[COMPARE] FNN: {time.time()-t:.2f}s")
    t = time.time(); print(f"[COMPARE] Inferencia LightGBM ({args.threads} hilos)...", flush=True)
    p_lgbm, hp_lgbm = predict_lgbm(lgbm_dir, x_val, args.threads); print(f"[COMPARE] LightGBM: {time.time()-t:.2f}s")

    cut_fnn, cut_lgbm = threshold(fnn_summary, "FNN"), threshold(lgbm_summary, "LightGBM")
    m_fnn, r_fnn, cm_fnn = evaluate(y, p_fnn, cut_fnn, "FNN_GAIN95")
    m_lgbm, r_lgbm, cm_lgbm = evaluate(y, p_lgbm, cut_lgbm, "LightGBM_GAIN95")
    metrics = pd.DataFrame([m_fnn, m_lgbm])
    m_fnn_05, _, _ = evaluate(y, p_fnn, 0.5, "FNN_GAIN95")
    m_lgbm_05, _, _ = evaluate(y, p_lgbm, 0.5, "LightGBM_GAIN95")
    operating = pd.DataFrame([{**m_fnn, "operating_point": "hpt_f1_threshold"}, {**m_lgbm, "operating_point": "hpt_f1_threshold"},
                              {**m_fnn_05, "operating_point": "threshold_0.5"}, {**m_lgbm_05, "operating_point": "threshold_0.5"}])

    pred_fnn = (p_fnn >= cut_fnn).astype(np.int8); pred_lgbm = (p_lgbm >= cut_lgbm).astype(np.int8)
    correct_fnn, correct_lgbm = pred_fnn == y, pred_lgbm == y
    both = correct_fnn & correct_lgbm; only_fnn = correct_fnn & ~correct_lgbm
    only_lgbm = ~correct_fnn & correct_lgbm; neither = ~correct_fnn & ~correct_lgbm
    discordant = int(only_fnn.sum() + only_lgbm.sum())
    positive_fnn, positive_lgbm = set(np.flatnonzero(pred_fnn)), set(np.flatnonzero(pred_lgbm))
    n_top = max(1, int(math.ceil(len(y) * .10)))
    top_fnn, top_lgbm = set(np.argpartition(p_fnn, -n_top)[-n_top:]), set(np.argpartition(p_lgbm, -n_top)[-n_top:])
    agreement = {
        "both_correct": int(both.sum()), "fnn_only_correct": int(only_fnn.sum()),
        "lgbm_only_correct": int(only_lgbm.sum()), "both_wrong": int(neither.sum()),
        "oracle_accuracy_if_either_correct": float((both | only_fnn | only_lgbm).mean()),
        "prediction_agreement_rate": float((pred_fnn == pred_lgbm).mean()),
        "prediction_cohen_kappa": float(cohen_kappa_score(pred_fnn, pred_lgbm)),
        "probability_pearson": float(pd.Series(p_fnn).corr(pd.Series(p_lgbm), method="pearson")),
        "probability_spearman": float(pd.Series(p_fnn).corr(pd.Series(p_lgbm), method="spearman")),
        "predicted_positive_jaccard": float(len(positive_fnn & positive_lgbm) / max(1, len(positive_fnn | positive_lgbm))),
        "top10_jaccard": float(len(top_fnn & top_lgbm) / max(1, len(top_fnn | top_lgbm))),
        "mcnemar_fnn_only_correct": int(only_fnn.sum()), "mcnemar_lgbm_only_correct": int(only_lgbm.sum()),
        "mcnemar_exact_pvalue": float(binomtest(int(min(only_fnn.sum(), only_lgbm.sum())), n=discordant, p=.5).pvalue) if discordant else 1.0,
    }
    by_target = pd.DataFrame([{
        "target": value, "support": int((y == value).sum()), "both_correct": int((both & (y == value)).sum()),
        "fnn_only_correct": int((only_fnn & (y == value)).sum()), "lgbm_only_correct": int((only_lgbm & (y == value)).sum()),
        "both_wrong": int((neither & (y == value)).sum()),
    } for value in (0, 1)])

    top3_fnn, selected_fnn = top3(val_df, p_fnn); top3_lgbm, selected_lgbm = top3(val_df, p_lgbm)
    top3_metrics = pd.DataFrame([{"model": "FNN_GAIN95", **top3_fnn}, {"model": "LightGBM_GAIN95", **top3_lgbm}])
    overlap = selected_fnn[["nucleo", "COD_SUBCATEGORIA"]].assign(fnn=1).merge(
        selected_lgbm[["nucleo", "COD_SUBCATEGORIA"]].assign(lgbm=1), on=["nucleo", "COD_SUBCATEGORIA"], how="outer"
    ).fillna(0)
    if overlap.empty:
        overlap_group = pd.DataFrame(columns=["nucleo", "intersection", "union", "jaccard"])
        overlap_summary = {"n_groups": 0, "mean_jaccard": 0.0, "median_jaccard": 0.0, "exact_same_top3_rate": 0.0}
    else:
        overlap["intersection"] = ((overlap.fnn == 1) & (overlap.lgbm == 1)).astype(int)
        overlap["union"] = ((overlap.fnn == 1) | (overlap.lgbm == 1)).astype(int)
        overlap_group = overlap.groupby("nucleo", observed=True).agg(intersection=("intersection", "sum"), union=("union", "sum")).reset_index()
        overlap_group["jaccard"] = overlap_group.intersection / overlap_group.union
        overlap_summary = {"n_groups": int(len(overlap_group)), "mean_jaccard": float(overlap_group.jaccard.mean()),
                           "median_jaccard": float(overlap_group.jaccard.median()),
                           "exact_same_top3_rate": float((overlap_group.intersection == 3).mean())}

    predictions = val_df[[c for c in ("source_row_index", "nucleo", "COD_SUBCATEGORIA", "CODIGO_FAMILIA", "target") if c in val_df]].copy()
    predictions["proba_fnn"] = p_fnn; predictions["proba_lgbm"] = p_lgbm
    predictions["pred_fnn"] = pred_fnn; predictions["pred_lgbm"] = pred_lgbm
    predictions["correct_fnn"] = correct_fnn; predictions["correct_lgbm"] = correct_lgbm
    predictions["error_pattern"] = np.select([both, only_fnn, only_lgbm], ["both_correct", "fnn_only_correct", "lgbm_only_correct"], default="both_wrong")
    predictions.to_parquet(data / "validation_predictions_aligned.parquet", index=False)

    save_both(metrics, "metrics_primary_thresholds.csv", data, docs); save_both(operating, "metrics_operating_points.csv", data, docs)
    save_both(r_fnn, "classification_report_fnn.csv", data, docs); save_both(r_lgbm, "classification_report_lgbm.csv", data, docs)
    save_both(pd.DataFrame(cm_fnn, index=["real_0", "real_1"], columns=["pred_0", "pred_1"]).reset_index(names="real"), "confusion_matrix_fnn.csv", data, docs)
    save_both(pd.DataFrame(cm_lgbm, index=["real_0", "real_1"], columns=["pred_0", "pred_1"]).reset_index(names="real"), "confusion_matrix_lgbm.csv", data, docs)
    save_both(by_target, "agreement_by_target.csv", data, docs); save_both(top3_metrics, "top3_metrics.csv", data, docs)
    save_both(overlap_group, "top3_overlap_by_group.csv", data, docs)
    counts = pd.DataFrame({"pattern": ["Ambos aciertan", "Solo FNN", "Solo LightGBM", "Ambos fallan"],
                           "count": [int(both.sum()), int(only_fnn.sum()), int(only_lgbm.sum()), int(neither.sum())]})
    save_both(counts, "agreement_counts.csv", data, docs)
    save_json(data / "agreement_summary.json", agreement); save_json(docs / "agreement_summary.json", agreement)
    save_json(data / "top3_overlap_summary.json", overlap_summary); save_json(docs / "top3_overlap_summary.json", overlap_summary)

    confusion_figure(cm_fnn, f"FNN GAIN95 — conteos (umbral={cut_fnn:.6f})", docs / "confusion_matrix_fnn_counts.png", False)
    confusion_figure(cm_fnn, "FNN GAIN95 — normalizada por clase real", docs / "confusion_matrix_fnn_normalized.png", True)
    confusion_figure(cm_lgbm, f"LightGBM GAIN95 — conteos (umbral={cut_lgbm:.6f})", docs / "confusion_matrix_lgbm_counts.png", False)
    confusion_figure(cm_lgbm, "LightGBM GAIN95 — normalizada por clase real", docs / "confusion_matrix_lgbm_normalized.png", True)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, probabilities in (("FNN", p_fnn), ("LightGBM", p_lgbm)):
        precision, recall, _ = precision_recall_curve(y, probabilities)
        ax.plot(recall, precision, label=f"{name} (PR-AUC={average_precision_score(y, probabilities):.4f})")
    ax.axhline(y.mean(), linestyle="--", label=f"Prevalencia={y.mean():.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Curvas Precision-Recall — validación W0-W3"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(docs / "pr_curves_validation.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, probabilities in (("FNN", p_fnn), ("LightGBM", p_lgbm)):
        fpr, tpr, _ = roc_curve(y, probabilities); ax.plot(fpr, tpr, label=f"{name} (ROC-AUC={roc_auc_score(y, probabilities):.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--"); ax.set(xlabel="FPR", ylabel="TPR", title="Curvas ROC — validación W0-W3")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(docs / "roc_curves_validation.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, name, probabilities in zip(axes, ["FNN", "LightGBM"], [p_fnn, p_lgbm]):
        ax.hist(probabilities[y == 0], bins=60, alpha=.55, density=True, label="target=0")
        ax.hist(probabilities[y == 1], bins=60, alpha=.55, density=True, label="target=1")
        ax.set(xlabel="Probabilidad", ylabel="Densidad", title=name); ax.grid(alpha=.2); ax.legend()
    fig.suptitle("Distribución de probabilidades por clase real"); fig.tight_layout()
    fig.savefig(docs / "probability_distributions_by_target.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6)); hb = ax.hexbin(p_fnn, p_lgbm, gridsize=70, mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="log10(conteo)"); ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set(xlabel="Probabilidad FNN", ylabel="Probabilidad LightGBM", title="Concordancia de probabilidades")
    fig.tight_layout(); fig.savefig(docs / "probability_agreement_hexbin.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    per_class = metrics.melt(id_vars=["model"], value_vars=["precision_target_0", "recall_target_0_specificity", "f1_target_0",
        "precision_target_1", "recall_target_1_sensitivity", "f1_target_1"], var_name="metric_class", value_name="value")
    labels = per_class.metric_class.unique().tolist(); x = np.arange(len(labels)); width = .36
    fig, ax = plt.subplots(figsize=(11, 6))
    for offset, model_name in zip([-width/2, width/2], metrics.model.tolist()):
        values = [float(per_class[(per_class.model == model_name) & (per_class.metric_class == label)].value.iloc[0]) for label in labels]
        ax.bar(x + offset, values, width, label=model_name)
    ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_ylim(0, 1); ax.set(ylabel="Valor", title="Métricas por clase")
    ax.grid(alpha=.2, axis="y"); ax.legend(); fig.tight_layout(); fig.savefig(docs / "per_class_metrics.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(counts.pattern, counts["count"])
    ax.set(ylabel="Registros", title="Complementariedad de errores"); ax.tick_params(axis="x", rotation=20); ax.grid(alpha=.2, axis="y")
    fig.tight_layout(); fig.savefig(docs / "error_complementarity.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    if not overlap_group.empty:
        fig, ax = plt.subplots(figsize=(7.5, 5)); ax.hist(overlap_group.jaccard, bins=np.linspace(0, 1, 8))
        ax.set(xlabel="Jaccard Top-3", ylabel="Núcleos", title="Solapamiento de recomendaciones Top-3"); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(docs / "top3_overlap_distribution.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    duration = round(time.time() - started, 3)
    summary = {"timestamp": datetime.now().isoformat(), "experiment_name": args.experiment_name,
        "analysis": "paired_validation_diagnostics_fnn_vs_lgbm_gain95", "training_performed": False, "w4_used": False,
        "train_parquet": str(train_path), "feature_set": "gain95", "feature_columns": names,
        "split": {"type": "row_stratified", "validation_size": val_size, "seed": seed},
        "n_rows_total": int(len(df)), "n_rows_validation": int(len(y)), "validation_positive_rate": float(y.mean()),
        "selected_artifacts": {"fnn_hpt_dir": str(fnn_dir), "lgbm_hpt_dir": str(lgbm_dir)},
        "artifact_discovery": {"fnn": fnn_audit, "lgbm": lgbm_audit},
        "thresholds": {"fnn": cut_fnn, "lgbm": cut_lgbm}, "hyperparameters": {"fnn": hp_fnn, "lgbm": hp_lgbm},
        "metrics": metrics.to_dict(orient="records"), "agreement": agreement,
        "agreement_by_target": by_target.to_dict(orient="records"), "top3_metrics": top3_metrics.to_dict(orient="records"),
        "top3_overlap": overlap_summary, "duration_seconds": duration, "device_fnn": str(device), "threads_lgbm": args.threads}
    save_json(data / "run_summary.json", summary); save_json(docs / "run_summary.json", summary)
    GLOBAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_LOG, "a", encoding="utf-8") as handle: handle.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")

    with open(docs / "results_overview.md", "w", encoding="utf-8") as handle:
        handle.write("# Diagnóstico FNN vs LightGBM — validación GAIN95\n\n")
        handle.write("No se reentrenó ningún modelo y W4 no fue utilizado.\n\n")
        handle.write(metrics[["model", "pr_auc", "roc_auc", "precision_target_1", "recall_target_1_sensitivity",
            "f1_target_1", "recall_target_0_specificity", "balanced_accuracy", "mcc", "lift_at_10"]].to_markdown(index=False))
        handle.write("\n\n## Complementariedad\n\n")
        for key, value in agreement.items(): handle.write(f"- {key}: {value}\n")

    print("\n[COMPARE] ================================================================")
    for row in metrics.to_dict(orient="records"):
        print(f"[COMPARE] {row['model']} | PR-AUC={row['pr_auc']:.6f} | ROC-AUC={row['roc_auc']:.6f} | "
              f"F1(1)={row['f1_target_1']:.6f} | Precision(1)={row['precision_target_1']:.6f} | "
              f"Recall(1)={row['recall_target_1_sensitivity']:.6f} | Specificity(0)={row['recall_target_0_specificity']:.6f} | "
              f"MCC={row['mcc']:.6f} | Lift@10={row['lift_at_10']:.4f}")
        print(f"[COMPARE] {row['model']} | TN={row['tn']:,} FP={row['fp']:,} FN={row['fn']:,} TP={row['tp']:,} | umbral={row['threshold']:.6f}")
    print(f"[COMPARE] Ambos aciertan={agreement['both_correct']:,} | solo FNN={agreement['fnn_only_correct']:,} | "
          f"solo LightGBM={agreement['lgbm_only_correct']:,} | ambos fallan={agreement['both_wrong']:,}")
    print(f"[COMPARE] Acuerdo={agreement['prediction_agreement_rate']:.4f} | Spearman={agreement['probability_spearman']:.4f} | McNemar p={agreement['mcnemar_exact_pvalue']:.6g}")
    print(f"[COMPARE] Tiempo={duration:.1f}s\n[COMPARE] Datos: {data}\n[COMPARE] Gráficas: {docs}")
    print("[COMPARE] W4/test NO se utilizó y no se entrenó ningún modelo.")
    print("[COMPARE] ================================================================")


if __name__ == "__main__":
    main()
