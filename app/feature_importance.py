"""
feature_importance.py — Análisis de importancia de features con LightGBM.

Qué hace:
  1. Carga parquet de features
  2. Entrena LightGBM
  3. Extrae importancia por GAIN y SPLIT
  4. Opcionalmente compara distribuciones: compradores (target=1) vs no-compradores (target=0)
  5. Guarda CSV en docs/feature_analysis/feature_summary.csv
  6. Guarda log JSONL en data/logs/feature_importance_runs.jsonl (append)
  7. Guarda gráficoS PNG en docs/feature_analysis/feature_importance.png

GAIN:  reducción total de pérdida al usar una feature para hacer un corte.
       Alta ganancia = feature discrimina bien compradores vs no-compradores.
       → Usar GAIN para seleccionar features.
SPLIT: cuántas veces se usó la feature como punto de corte (frecuencia, no calidad).
       → Útil solo para detectar features redundantes.

Flags:
  --parquet PATH    parquet alternativo (default: features_train.parquet)
  --all-features    usa todas las columnas numéricas del parquet (no solo FEATURE_COLUMNS)
  --compare-groups  imprime media/mediana por grupo: compradores vs no-compradores

Ejemplos:
  python feature_importance.py
  python feature_importance.py --all-features --compare-groups

Outputs (siempre sobreescriben en cada ejecución):
  docs/feature_analysis/feature_summary.csv
  docs/feature_analysis/feature_importance.png
  docs/feature_analysis/corr_with_target.png
  docs/feature_analysis/corr_features.png
  data/logs/feature_importance_runs.jsonl      (append)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    FEATURE_COLUMNS,
    FEATURES_TRAIN_FILE,
    TIPO_CICLO_CATEGORIES,
    TIPO_CICLO_COL,
)

OUTPUT_CSV = Path(__file__).resolve().parent.parent / "docs" / "feature_analysis" / "feature_summary.csv"
OUTPUT_PNG = Path(__file__).resolve().parent.parent / "docs" / "feature_analysis" / "feature_importance.png"
LOG_FILE   = Path(__file__).resolve().parent.parent / "data" / "logs" / "feature_importance_runs.jsonl"


_SKIP_COLS = {"nucleo", "COD_SUBCATEGORIA", "target"}


def _build_X(df: pd.DataFrame, feat_cols_base: list) -> tuple[np.ndarray, list[str]]:
    """Construye matriz de features: columnas numéricas + one-hot de TIPO_CICLO_COL."""
    X_base = df[feat_cols_base].copy()
    if TIPO_CICLO_COL in df.columns:
        dummies = pd.get_dummies(
            pd.Categorical(df[TIPO_CICLO_COL], categories=TIPO_CICLO_CATEGORIES),
            prefix="tipo",
            drop_first=True,
        )
        dummies.index = X_base.index
        X_base = pd.concat([X_base, dummies], axis=1)
    col_names = list(X_base.columns)
    X = np.nan_to_num(X_base.values.astype(np.float32), nan=0.0)
    return X, col_names


def _compare_groups(df: pd.DataFrame, feat_cols_base: list) -> pd.DataFrame:
    """Compara distribución de features entre compradores (target=1) y no-compradores (target=0)."""
    df1 = df[df["target"] == 1]
    df0 = df[df["target"] == 0]
    rows = []
    for col in feat_cols_base:
        if col not in df.columns or df[col].dtype == object:
            continue
        rows.append({
            "feature":            col,
            "media_compradores":  round(float(df1[col].mean()), 4),
            "media_no_compra":    round(float(df0[col].mean()), 4),
            "mediana_compradores": round(float(df1[col].median()), 4),
            "mediana_no_compra":  round(float(df0[col].median()), 4),
            "diferencia_medias":  round(float(df1[col].mean() - df0[col].mean()), 4),
        })
    compare_df = pd.DataFrame(rows)
    compare_df["diff_abs"] = compare_df["diferencia_medias"].abs()
    compare_df = compare_df.sort_values("diff_abs", ascending=False).reset_index(drop=True)
    return compare_df.drop(columns=["diff_abs"])


def _plot_correlations(df: pd.DataFrame, feat_cols_base: list, out_dir: Path) -> pd.Series:
    """Genera corr_with_target.png (barras) y corr_features.png (heatmap)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    numeric_cols = [c for c in feat_cols_base if c in df.columns and df[c].dtype != object]

    # ── 1. Correlación de cada feature con el target ──
    corr_target = (
        df[numeric_cols + ["target"]].corr()["target"].drop("target").sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, max(5, len(numeric_cols) * 0.38)))
    colors = ["#e74c3c" if v >= 0 else "#3498db" for v in corr_target.values]
    bars = ax.barh(corr_target.index[::-1], corr_target.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Correlación de Pearson con target")
    ax.set_title("Correlación features vs target\nRojo = positiva (más compras) | Azul = negativa")
    for bar, val in zip(bars, corr_target.values[::-1]):
        offset = 0.002 if val >= 0 else -0.002
        ha = "left" if val >= 0 else "right"
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha=ha, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "corr_with_target.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FeatImportance] Guardado: {out_dir / 'corr_with_target.png'}")

    # ── 2. Heatmap feature×feature ──
    corr_matrix = df[numeric_cols].corr().round(2)
    n = len(numeric_cols)
    fig_size = max(10, n * 0.58)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    try:
        import seaborn as sns
        font_size = max(4, min(8, int(120 / n)))
        sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap="RdBu_r",
                    center=0, square=True, ax=ax,
                    annot_kws={"size": font_size}, linewidths=0.3)
    except ImportError:
        im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(numeric_cols, rotation=90, fontsize=7)
        ax.set_yticklabels(numeric_cols, fontsize=7)
        plt.colorbar(im, ax=ax)
    ax.set_title("Matriz de correlación (features numéricas)")
    plt.tight_layout()
    plt.savefig(out_dir / "corr_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FeatImportance] Guardado: {out_dir / 'corr_features.png'}")

    return corr_target


def _default_params(n_pos: int, n_total: int) -> dict:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": 64,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.8,
        "feature_fraction": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 0.01,
        "scale_pos_weight": (n_total - n_pos) / max(n_pos, 1),
    }


def select_until_threshold(result_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Devuelve las features hasta la primera fila donde acum_gain_pct >= threshold (inclusive)."""
    reached = result_df["acum_gain_pct"] >= threshold

    if not reached.any():
        return result_df.copy()

    last_idx = reached.idxmax()
    return result_df.loc[:last_idx].copy()


def main():
    t_start = time.time()
    parser = argparse.ArgumentParser(description="Feature importance con LightGBM")
    parser.add_argument("--parquet", type=str, default=str(FEATURES_TRAIN_FILE),
                        help="Path al parquet de features (default: features_train.parquet)")
    parser.add_argument("--all-features", action="store_true",
                        help="Usar todas las columnas numéricas del parquet (ignora FEATURE_COLUMNS)")
    parser.add_argument("--compare-groups", action="store_true",
                        help="Imprimir media/mediana por grupo: compradores vs no-compradores")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"[ERROR] No se encontró: {parquet_path}")
        return

    print(f"[FeatImportance] Cargando: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    n_pos = int(df["target"].sum())
    n_neg = len(df) - n_pos
    print(f"[FeatImportance] {len(df):,} filas | {df['nucleo'].nunique():,} familias | "
          f"target+={n_pos:,} ({df['target'].mean()*100:.1f}%) | target-={n_neg:,}")

    # feat_cols_base: numéricas sin one-hot — base para _build_X, compare-groups y correlaciones
    if args.all_features:
        feat_cols_base = [c for c in df.columns
                          if c not in _SKIP_COLS and c != TIPO_CICLO_COL and df[c].dtype != object]
        print(f"[FeatImportance] --all-features: {len(feat_cols_base)} columnas numéricas del parquet")
    else:
        feat_cols_base = [c for c in FEATURE_COLUMNS if c in df.columns]

    X, col_names = _build_X(df, feat_cols_base)
    print(f"[FeatImportance] Features enviadas al modelo: {len(col_names)} → {col_names}")
    y = df["target"].values.astype(np.float32)

    params = _default_params(n_pos, len(y))
    n_estimators = 300
    print("[FeatImportance] Usando params neutros (sin sesgo de HPT)")

    print(f"[FeatImportance] Entrenando LightGBM ({n_estimators} árboles)...")
    ds = lgb.Dataset(X, label=y, feature_name=col_names)
    booster = lgb.train(params, ds, num_boost_round=n_estimators, callbacks=[lgb.log_evaluation(-1)])

    # Extraer importancia por gain y split
    gain_vals   = booster.feature_importance(importance_type="gain")
    split_vals  = booster.feature_importance(importance_type="split")
    total_gain  = gain_vals.sum() if gain_vals.sum() > 0 else 1.0
    total_split = split_vals.sum() if split_vals.sum() > 0 else 1.0

    result_df = pd.DataFrame({
        "feature":    col_names,
        "gain_abs":   gain_vals,
        "gain_pct":   (gain_vals / total_gain * 100).round(2),
        "split_abs":  split_vals,
        "split_pct":  (split_vals / total_split * 100).round(2),
    }).sort_values("gain_pct", ascending=False).reset_index(drop=True)

    result_df["rank"] = result_df.index + 1
    result_df["acum_gain_pct"] = result_df["gain_pct"].cumsum().round(2)

    # Correlación de Pearson con target — columna extra en el CSV (feat_cols_base ya es todo numérico)
    corr_series = df[feat_cols_base + ["target"]].corr()["target"].drop("target").round(4)
    result_df["corr_target"] = result_df["feature"].map(corr_series)

    # Mostrar en consola
    print(f"\n{'#':>3}  {'Feature':<35} {'Gain%':>7} {'Acum%':>7} {'Split%':>7}")
    print("-" * 65)
    for _, row in result_df.iterrows():
        print(f"{int(row['rank']):>3}  {row['feature']:<35} {row['gain_pct']:>7.2f} {row['acum_gain_pct']:>7.2f} {row['split_pct']:>7.2f}")

    print(f"\n[FeatImportance] Total features: {len(result_df)}")
    print(f"[FeatImportance] Features con gain=0: {(result_df['gain_pct']==0).sum()}")
    print(f"[FeatImportance] Top-3 acumulan: {result_df.head(3)['gain_pct'].sum():.1f}% del gain")
    print(f"[FeatImportance] Top-5 acumulan: {result_df.head(5)['gain_pct'].sum():.1f}% del gain")

    for threshold in [80, 90, 95]:
        top_t = select_until_threshold(result_df, threshold)
        feats = list(top_t["feature"])
        print(f"\n[FeatImportance] Features que acumulan al menos {threshold}% del gain ({len(feats)} features):")
        print(f"  {feats}")
    print()

    # ── Comparación compradores vs no-compradores ─────────────────────────
    compare_df = None
    if args.compare_groups:
        compare_df = _compare_groups(df, feat_cols_base)
        print(f"\n{'Feature':<35} {'Media_1':>10} {'Media_0':>10} {'Diferencia':>12}")
        print("-" * 70)
        for _, row in compare_df.iterrows():
            print(f"{row['feature']:<35} {row['media_compradores']:>10.4f} "
                  f"{row['media_no_compra']:>10.4f} {row['diferencia_medias']:>12.4f}")
        print()

    # ── Log JSONL ─────────────────────────────────────────────────────────
    duracion_s = round(time.time() - t_start, 1)
    ranking_list = result_df[["rank", "feature", "gain_pct", "acum_gain_pct", "split_pct"]].to_dict(orient="records")
    run_log = {
        "timestamp":          datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "duracion_seg":        duracion_s,
        "parquet_path":        str(parquet_path),
        "n_features_modelo":   len(col_names),
        "n_filas_parquet":     len(df),
        "n_series_target1":    n_pos,
        "n_series_target0":    n_neg,
        "modo_all_features":   args.all_features,
        "hparams_usados":      "defaults",
        "n_estimators":        n_estimators,
        "features_80pct":      [r["feature"] for r in ranking_list if r["acum_gain_pct"] <= 80],
        "features_90pct":      [r["feature"] for r in ranking_list if r["acum_gain_pct"] <= 90],
        "ranking":             ranking_list,
    }
    if compare_df is not None:
        run_log["compare_groups"] = compare_df.to_dict(orient="records")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_log, ensure_ascii=False) + "\n")
    print(f"[FeatImportance] Log guardado: {LOG_FILE}")
    print(f"[FeatImportance] Duración total: {duracion_s}s")

    # Guardar CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[FeatImportance] CSV guardado: {OUTPUT_CSV}")

    # ── Gráficos ───────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, max(5, len(result_df) * 0.35)))
    colors = ["#e74c3c" if g >= 5 else "#3498db" if g >= 1 else "#bdc3c7"
              for g in result_df["gain_pct"]]
    bars = ax.barh(result_df["feature"][::-1], result_df["gain_pct"][::-1], color=colors[::-1])
    ax.set_xlabel("Gain %")
    ax.set_title("Feature Importance (LightGBM — Gain %)\nRojo ≥5% | Azul ≥1% | Gris <1%")
    for bar, val in zip(bars, result_df["gain_pct"][::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FeatImportance] PNG guardado: {OUTPUT_PNG}")

    _plot_correlations(df, feat_cols_base, OUTPUT_CSV.parent)


if __name__ == "__main__":
    main()
