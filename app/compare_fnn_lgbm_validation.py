"""Compara predicciones finales FNN y LightGBM GAIN95 sobre W4."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FNN = ROOT / "data/models/final/fnn_gain95_w4_final"
DEFAULT_LGBM = ROOT / "data/models/final/lgbm_gain95_w4_final"
DATA_ROOT = ROOT / "data/analysis/model_comparison"
DOCS_ROOT = ROOT / "docs/model_comparison"
GLOBAL_LOG = ROOT / "data/logs/model_comparison_runs.jsonl"
KEYS = ["nucleo", "COD_SUBCATEGORIA", "target"]


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, value: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)


def load_run(model_dir: Path, label: str):
    pred_path = model_dir / "eval_predictions.parquet"
    summary_path = model_dir / "run_summary.json"
    if not pred_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"{label}: faltan predicciones o run_summary en {model_dir}")
    df = pd.read_parquet(pred_path)
    missing = [c for c in KEYS + ["proba_compra"] if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: faltan columnas {missing}")
    summary = load_json(summary_path)
    threshold = summary.get("frozen_threshold")
    if threshold is None:
        threshold = summary.get("external_eval_metrics", {}).get(
            "threshold_frozen_from_validation", 0.5
        )
    return df[KEYS + ["proba_compra"]].copy(), summary, float(threshold)


def align(fnn: pd.DataFrame, lgbm: pd.DataFrame) -> pd.DataFrame:
    if len(fnn) == len(lgbm) and all(
        fnn[c].reset_index(drop=True).equals(lgbm[c].reset_index(drop=True))
        for c in KEYS
    ):
        return pd.DataFrame({
            "nucleo": fnn.nucleo.to_numpy(),
            "COD_SUBCATEGORIA": fnn.COD_SUBCATEGORIA.to_numpy(),
            "target": fnn.target.to_numpy(np.int8),
            "proba_fnn": fnn.proba_compra.to_numpy(float),
            "proba_lgbm": lgbm.proba_compra.to_numpy(float),
        })
    a, b = fnn.copy(), lgbm.copy()
    a["_occ"] = a.groupby(KEYS, sort=False).cumcount()
    b["_occ"] = b.groupby(KEYS, sort=False).cumcount()
    out = a.merge(b, on=KEYS + ["_occ"], how="inner", validate="one_to_one",
                  suffixes=("_fnn", "_lgbm"))
    if len(out) != len(a) or len(out) != len(b):
        raise RuntimeError("No se pudieron alinear todas las predicciones.")
    return out.rename(columns={
        "proba_compra_fnn": "proba_fnn",
        "proba_compra_lgbm": "proba_lgbm",
    }).drop(columns="_occ")


def evaluate(y: np.ndarray, p: np.ndarray, cut: float, model: str):
    pred = (p >= cut).astype(np.int8)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    report = classification_report(
        y, pred, labels=[0, 1],
        target_names=["target_0_no_compra", "target_1_compra"],
        output_dict=True, zero_division=0,
    )
    base = float(y.mean())
    n_top = max(1, int(math.ceil(len(y) * 0.10)))
    top = np.argpartition(p, -n_top)[-n_top:]
    top_rate = float(y[top].mean())
    row = {
        "model": model, "threshold": cut, "n_w4": int(len(y)),
        "support_target_0": int((y == 0).sum()),
        "support_target_1": int((y == 1).sum()),
        "predicted_target_0": int((pred == 0).sum()),
        "predicted_target_1": int((pred == 1).sum()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "precision_target_0": float(report["target_0_no_compra"]["precision"]),
        "recall_target_0": float(report["target_0_no_compra"]["recall"]),
        "f1_target_0": float(report["target_0_no_compra"]["f1-score"]),
        "precision_target_1": float(precision_score(y, pred, zero_division=0)),
        "recall_target_1": float(recall_score(y, pred, zero_division=0)),
        "f1_target_1": float(f1_score(y, pred, zero_division=0)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "base_rate": base, "top10_positive_rate": top_rate,
        "lift_at_10": top_rate / base if base else 0.0,
    }
    return row, pd.DataFrame(report).T.reset_index(names="class_or_average"), cm, pred


def top3(df: pd.DataFrame, probability: str, k: int = 3):
    w = df[["nucleo", "COD_SUBCATEGORIA", "target", probability]].copy()
    w = w.groupby(["nucleo", "COD_SUBCATEGORIA"], sort=False, observed=True).agg(
        target=("target", "max"), proba=(probability, "max")
    ).reset_index()
    g = w.groupby("nucleo", sort=False, observed=True)
    w["size"] = g.target.transform("size")
    w["positives"] = g.target.transform("sum")
    w["rank"] = g.proba.rank(method="first", ascending=False)
    eligible = (w["size"] >= k) & (w["positives"] > 0)
    den = w.loc[eligible, ["nucleo", "positives"]].drop_duplicates(
        "nucleo"
    ).set_index("nucleo").positives
    selected = w.loc[
        eligible & (w["rank"] <= k),
        ["nucleo", "COD_SUBCATEGORIA", "target", "rank"],
    ]
    if den.empty:
        return {"precision_at_3": 0.0, "recall_at_3": 0.0,
                "hit_rate_at_3": 0.0, "n_groups_evaluated": 0}, selected
    hits = selected.groupby("nucleo", observed=True).target.sum().reindex(
        den.index, fill_value=0
    ).astype(float)
    return {
        "precision_at_3": float((hits / k).mean()),
        "recall_at_3": float((hits / den).mean()),
        "hit_rate_at_3": float((hits > 0).mean()),
        "n_groups_evaluated": int(len(den)),
    }, selected


def overlap(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    sa = a.groupby("nucleo", observed=True).COD_SUBCATEGORIA.agg(set)
    sb = b.groupby("nucleo", observed=True).COD_SUBCATEGORIA.agg(set)
    common = sa.index.intersection(sb.index)
    if not len(common):
        return {"n_groups_compared": 0, "mean_common_items": 0.0, "mean_jaccard": 0.0}
    intersections, jaccards = [], []
    for key in common:
        x, y = sa.loc[key], sb.loc[key]
        intersections.append(len(x & y))
        jaccards.append(len(x & y) / len(x | y))
    return {
        "n_groups_compared": int(len(common)),
        "mean_common_items": float(np.mean(intersections)),
        "mean_jaccard": float(np.mean(jaccards)),
    }


def confusion_plot(cm, title, path, normalized=False):
    values = cm.astype(float)
    if normalized:
        den = values.sum(axis=1, keepdims=True)
        values = np.divide(values, den, out=np.zeros_like(values), where=den > 0)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(values)
    fig.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], ["Predice 0", "Predice 1"])
    ax.set_yticks([0, 1], ["Real 0", "Real 1"])
    ax.set(xlabel="Predicción", ylabel="Clase real", title=title)
    for i in range(2):
        for j in range(2):
            text = f"{values[i,j]:.3f}" if normalized else f"{int(values[i,j]):,}"
            ax.text(j, i, text, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def metric_plot(metrics: pd.DataFrame, path: Path):
    labels = ["Precisión C0", "Recall C0", "F1 C0",
              "Precisión C1", "Recall C1", "F1 C1"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    columns = ["precision_target_0", "recall_target_0", "f1_target_0",
               "precision_target_1", "recall_target_1", "f1_target_1"]
    for i, (_, row) in enumerate(metrics.iterrows()):
        ax.bar(x + (i - 0.5) * width, [row[c] for c in columns],
               width=width, label=row.model)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set(ylabel="Valor", title="Métricas por clase sobre W4")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pr_plot(y, p_fnn, p_lgbm, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, p in (("FNN", p_fnn), ("LightGBM", p_lgbm)):
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, label=f"{name} (PR-AUC={average_precision_score(y,p):.4f})")
    ax.axhline(float(y.mean()), linestyle="--", label=f"Prevalencia={y.mean():.4f}")
    ax.set(xlabel="Recall", ylabel="Precisión", title="Curvas Precision-Recall sobre W4")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fnn-dir", default=str(DEFAULT_FNN))
    parser.add_argument("--lgbm-dir", default=str(DEFAULT_LGBM))
    parser.add_argument("--experiment-name", default="fnn_lgbm_gain95_w4_diagnostics")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    started = time.time()
    data_out = DATA_ROOT / args.experiment_name
    docs_out = DOCS_ROOT / args.experiment_name
    for path in (data_out, docs_out):
        if path.exists() and not args.force:
            raise FileExistsError(f"Ya existe {path}. Use --force.")
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    print("[COMPARE-W4] Cargando predicciones finales...", flush=True)
    fnn, fnn_summary, cut_fnn = load_run(Path(args.fnn_dir), "FNN")
    lgbm, lgbm_summary, cut_lgbm = load_run(Path(args.lgbm_dir), "LightGBM")
    aligned = align(fnn, lgbm)
    y = aligned.target.to_numpy(np.int8)
    p_fnn = aligned.proba_fnn.to_numpy(float)
    p_lgbm = aligned.proba_lgbm.to_numpy(float)

    print(f"[COMPARE-W4] Filas={len(y):,} | positivos={int(y.sum()):,} ({y.mean()*100:.2f}%)")
    print(f"[COMPARE-W4] Umbrales | FNN={cut_fnn:.6f} | LightGBM={cut_lgbm:.6f}")

    m_fnn, r_fnn, cm_fnn, pred_fnn = evaluate(y, p_fnn, cut_fnn, "FNN")
    m_lgbm, r_lgbm, cm_lgbm, pred_lgbm = evaluate(y, p_lgbm, cut_lgbm, "LightGBM")
    t_fnn, s_fnn = top3(aligned, "proba_fnn")
    t_lgbm, s_lgbm = top3(aligned, "proba_lgbm")
    top_overlap = overlap(s_fnn, s_lgbm)

    correct_fnn, correct_lgbm = pred_fnn == y, pred_lgbm == y
    categories = np.select(
        [correct_fnn & correct_lgbm, correct_fnn & ~correct_lgbm,
         ~correct_fnn & correct_lgbm],
        ["Ambos aciertan", "Solo FNN acierta", "Solo LightGBM acierta"],
        default="Ambos fallan",
    )
    aligned["pred_fnn"], aligned["pred_lgbm"] = pred_fnn, pred_lgbm
    aligned["correct_fnn"], aligned["correct_lgbm"] = correct_fnn, correct_lgbm
    aligned["agreement_category"] = categories

    order = ["Ambos aciertan", "Solo FNN acierta",
             "Solo LightGBM acierta", "Ambos fallan"]
    agreement = pd.Series(categories).value_counts().reindex(
        order, fill_value=0
    ).rename_axis("category").reset_index(name="count")
    agreement["percentage"] = agreement["count"] / len(aligned)
    by_target = aligned.groupby(
        ["target", "agreement_category"], observed=True
    ).size().rename("count").reset_index()
    by_target["percentage_within_target"] = by_target["count"] / by_target.groupby(
        "target", observed=True
    )["count"].transform("sum")

    metrics = pd.DataFrame([m_fnn, m_lgbm])
    reports = pd.concat([r_fnn.assign(model="FNN"),
                         r_lgbm.assign(model="LightGBM")], ignore_index=True)
    confusion = pd.DataFrame([
        {"model": "FNN", "tn": m_fnn["tn"], "fp": m_fnn["fp"],
         "fn": m_fnn["fn"], "tp": m_fnn["tp"]},
        {"model": "LightGBM", "tn": m_lgbm["tn"], "fp": m_lgbm["fp"],
         "fn": m_lgbm["fn"], "tp": m_lgbm["tp"]},
    ])
    top3_table = pd.DataFrame([{"model": "FNN", **t_fnn},
                               {"model": "LightGBM", **t_lgbm}])

    for path in (data_out, docs_out):
        metrics.to_csv(path / "metrics_by_model.csv", index=False)
        reports.to_csv(path / "classification_reports.csv", index=False)
        confusion.to_csv(path / "confusion_matrices.csv", index=False)
        top3_table.to_csv(path / "top3_metrics.csv", index=False)
        agreement.to_csv(path / "agreement_summary.csv", index=False)
        by_target.to_csv(path / "agreement_by_target.csv", index=False)
    aligned.to_parquet(data_out / "w4_predictions_aligned.parquet", index=False)

    confusion_plot(cm_fnn, "FNN — matriz de confusión W4",
                   docs_out / "confusion_matrix_fnn_counts.png")
    confusion_plot(cm_fnn, "FNN — matriz normalizada W4",
                   docs_out / "confusion_matrix_fnn_normalized.png", True)
    confusion_plot(cm_lgbm, "LightGBM — matriz de confusión W4",
                   docs_out / "confusion_matrix_lgbm_counts.png")
    confusion_plot(cm_lgbm, "LightGBM — matriz normalizada W4",
                   docs_out / "confusion_matrix_lgbm_normalized.png", True)
    metric_plot(metrics, docs_out / "per_class_metrics.png")
    pr_plot(y, p_fnn, p_lgbm, docs_out / "pr_curves_w4.png")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(agreement.category, agreement["count"])
    ax.set(ylabel="Número de registros", title="Complementariedad de aciertos sobre W4")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(docs_out / "error_complementarity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": args.experiment_name,
        "analysis_protocol": "compare_saved_final_predictions_on_external_W4",
        "w4_is_external_test": True,
        "retrained_models": False,
        "repeated_inference": False,
        "n_w4": int(len(y)),
        "positive_rate_w4": float(y.mean()),
        "thresholds": {"FNN": cut_fnn, "LightGBM": cut_lgbm},
        "metrics": {"FNN": m_fnn, "LightGBM": m_lgbm},
        "top3": {"FNN": t_fnn, "LightGBM": t_lgbm, "overlap": top_overlap},
        "agreement": agreement.to_dict(orient="records"),
        "agreement_by_target": by_target.to_dict(orient="records"),
        "binary_prediction_agreement": float(np.mean(pred_fnn == pred_lgbm)),
        "probability_correlation_pearson": float(np.corrcoef(p_fnn, p_lgbm)[0, 1]),
        "duration_seconds": round(time.time() - started, 3),
        "source_experiments": {
            "FNN": fnn_summary.get("experiment_name"),
            "LightGBM": lgbm_summary.get("experiment_name"),
        },
    }
    save_json(data_out / "run_summary.json", summary)
    save_json(docs_out / "run_summary.json", summary)
    GLOBAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")

    print("\n[COMPARE-W4] ================================================================")
    for row in (m_fnn, m_lgbm):
        print(f"[COMPARE-W4] {row['model']} | PR-AUC={row['pr_auc']:.6f} "
              f"| ROC-AUC={row['roc_auc']:.6f} | Lift@10={row['lift_at_10']:.4f}")
        print(f"[COMPARE-W4] {row['model']} | C0 P={row['precision_target_0']:.6f} "
              f"R={row['recall_target_0']:.6f} F1={row['f1_target_0']:.6f} "
              f"| C1 P={row['precision_target_1']:.6f} "
              f"R={row['recall_target_1']:.6f} F1={row['f1_target_1']:.6f}")
        print(f"[COMPARE-W4] {row['model']} | TN={row['tn']:,} FP={row['fp']:,} "
              f"FN={row['fn']:,} TP={row['tp']:,}")
    for _, row in agreement.iterrows():
        print(f"[COMPARE-W4] {row.category}: {int(row['count']):,} "
              f"({row.percentage*100:.2f}%)")
    print(f"[COMPARE-W4] Acuerdo binario={summary['binary_prediction_agreement']*100:.2f}% "
          f"| correlación={summary['probability_correlation_pearson']:.4f}")
    print(f"[COMPARE-W4] Top-3 comunes={top_overlap['mean_common_items']:.3f} "
          f"| Jaccard={top_overlap['mean_jaccard']:.4f}")
    print(f"[COMPARE-W4] Datos: {data_out}")
    print(f"[COMPARE-W4] Gráficas: {docs_out}")
    print(f"[COMPARE-W4] Tiempo total: {time.time()-started:.2f}s")
    print("[COMPARE-W4] ================================================================")


if __name__ == "__main__":
    main()
