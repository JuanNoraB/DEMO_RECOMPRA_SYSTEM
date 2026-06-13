"""
feature_importance.py — Análisis de importancia de features con LightGBM.

Qué hace:
  1. Carga features_train.parquet
  2. Entrena LightGBM con los mejores hparams (best_hparams_lgbm.json) o defaults
  3. Extrae feature importance por GAIN y SPLIT
  4. Guarda tabla CSV en docs/feature_analysis/feature_summary.csv
  5. Imprime ranking en consola
  6. Guarda gráfico PNG en docs/feature_analysis/feature_importance.png (si matplotlib disponible)

Uso:
  python feature_importance.py
  python feature_importance.py --parquet data/features_store/features_train.parquet
  python feature_importance.py --top 20
"""
from __future__ import annotations

import argparse
import json
import sys
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

BEST_LGBM_HPARAMS_FILE = Path(__file__).resolve().parent.parent / "data" / "models" / "best_hparams_lgbm.json"
OUTPUT_CSV  = Path(__file__).resolve().parent.parent / "docs" / "feature_analysis" / "feature_summary.csv"
OUTPUT_PNG  = Path(__file__).resolve().parent.parent / "docs" / "feature_analysis" / "feature_importance.png"


_SKIP_COLS = {"nucleo", "COD_SUBCATEGORIA", "target"}


def _build_X(df: pd.DataFrame, all_features: bool = False) -> tuple[np.ndarray, list[str]]:
    if all_features:
        # Usar todas las columnas numéricas del parquet excepto claves y target
        feat_cols_base = [
            c for c in df.columns
            if c not in _SKIP_COLS
            and c != TIPO_CICLO_COL
            and df[c].dtype != object
        ]
    else:
        feat_cols_base = [c for c in FEATURE_COLUMNS if c in df.columns]

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


def main():
    parser = argparse.ArgumentParser(description="Feature importance con LightGBM")
    parser.add_argument("--parquet", type=str, default=str(FEATURES_TRAIN_FILE),
                        help="Path al parquet de features (default: features_train.parquet)")
    parser.add_argument("--top", type=int, default=None,
                        help="Mostrar solo top-N features (default: todas)")
    parser.add_argument("--no-plot", action="store_true",
                        help="No generar gráfico PNG")
    parser.add_argument("--all-features", action="store_true",
                        help="Usar todas las columnas numéricas del parquet (ignora FEATURE_COLUMNS)")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"[ERROR] No se encontró: {parquet_path}")
        return

    print(f"[FeatImportance] Cargando: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"[FeatImportance] {len(df):,} filas | {df['nucleo'].nunique():,} familias | "
          f"target+={int(df['target'].sum()):,} ({df['target'].mean()*100:.1f}%)")

    if args.all_features:
        print("[FeatImportance] Modo --all-features: usando todas las columnas numéricas del parquet")
    X, col_names = _build_X(df, all_features=args.all_features)
    print(f"[FeatImportance] Features enviadas al modelo: {len(col_names)} → {col_names}")
    y = df["target"].values.astype(np.float32)
    n_pos = int(y.sum())

    # Cargar mejores hparams si existen
    if BEST_LGBM_HPARAMS_FILE.exists():
        with open(BEST_LGBM_HPARAMS_FILE) as f:
            saved = json.load(f)
        precision_saved = saved.pop("precision@3", None)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "scale_pos_weight": (len(y) - n_pos) / max(n_pos, 1),
            **saved,
        }
        n_estimators = int(params.pop("n_estimators", 300))
        print(f"[FeatImportance] Usando best_hparams_lgbm.json (Precision@3={precision_saved})")
    else:
        params = _default_params(n_pos, len(y))
        n_estimators = 300
        print("[FeatImportance] best_hparams_lgbm.json no encontrado — usando defaults")

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

    # Mostrar en consola
    top_n = args.top if args.top else len(result_df)
    print(f"\n{'#':>3}  {'Feature':<35} {'Gain%':>7} {'Acum%':>7} {'Split%':>7}")
    print("-" * 65)
    for _, row in result_df.head(top_n).iterrows():
        print(f"{int(row['rank']):>3}  {row['feature']:<35} {row['gain_pct']:>7.2f} {row['acum_gain_pct']:>7.2f} {row['split_pct']:>7.2f}")

    print(f"\n[FeatImportance] Total features: {len(result_df)}")
    print(f"[FeatImportance] Features con gain=0: {(result_df['gain_pct']==0).sum()}")
    print(f"[FeatImportance] Top-3 acumulan: {result_df.head(3)['gain_pct'].sum():.1f}% del gain")
    print(f"[FeatImportance] Top-5 acumulan: {result_df.head(5)['gain_pct'].sum():.1f}% del gain")

    for threshold in [80, 90]:
        top_t = result_df[result_df["acum_gain_pct"] <= threshold]
        if len(top_t) == 0:
            top_t = result_df.head(1)
        feats = list(top_t["feature"])
        print(f"\n[FeatImportance] Features que explican {threshold}% del gain ({len(feats)} features):")
        print(f"  {feats}")
    print()

    # Guardar CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[FeatImportance] CSV guardado: {OUTPUT_CSV}")

    # Gráfico PNG
    if not args.no_plot:
        try:
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
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=8)
            plt.tight_layout()
            plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[FeatImportance] PNG guardado: {OUTPUT_PNG}")
        except ImportError:
            print("[FeatImportance] matplotlib no disponible — omitiendo gráfico")


if __name__ == "__main__":
    main()
