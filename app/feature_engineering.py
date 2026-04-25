"""
feature_engineering.py — Cálculo de features + target.

Uso:
  python feature_engineering.py --historico data/raw/historico_base.csv
  python feature_engineering.py --historico data/raw/historico_base.csv --filtro series.csv
  python feature_engineering.py --historico data/raw/historico_base.csv --filtro series.csv --workers 8

El script:
  1. Lee el histórico del path indicado.
  2. Toma fecha_min y fecha_max del histórico automáticamente.
  3. Features se calculan con datos hasta (fecha_max - 21 días).
  4. Target = ¿hubo compra en los últimos 21 días? (binario por serie).
  5. Si se pasa --filtro, solo calcula para esas series (familia, subcategoria).
  6. Al guardar: UPSERT — reemplaza series existentes, agrega nuevas.
  7. Procesamiento en paralelo por familia.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from config import (
    FEATURES_TRAIN_FILE,
    RAW_DTYPES,
    NUMERIC_COLUMNS,
    FEATURE_COLUMNS,
    PREDICTION_WINDOW_DAYS,
)
from features import (
    calcular_ciclos_por_bloques,
    compute_recency_features,
    compute_frequency_features,
    compute_sow_features,
    compute_seasonality_features,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

def load_historical_dataset(path: Path) -> pd.DataFrame:
    """Carga y limpia el CSV de transacciones históricas."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró: {path}")

    df = pd.read_csv(path, encoding="utf-8", sep=";")

    raw_dates = df["DIM_PERIODO"].copy()
    df["DIM_PERIODO"] = pd.to_datetime(raw_dates, format="%d-%b-%y", errors="coerce")
    if df["DIM_PERIODO"].isna().all():
        df["DIM_PERIODO"] = pd.to_datetime(raw_dates, errors="coerce")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in RAW_DTYPES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df = df.dropna(subset=["DIM_PERIODO", "CODIGO_FAMILIA", "COD_SUBCATEGORIA"])
    df["CODIGO_FAMILIA"] = df["CODIGO_FAMILIA"].astype(int)
    df["COD_SUBCATEGORIA"] = df["COD_SUBCATEGORIA"].astype(int)
    return df


def load_filter_series(path: Path) -> pd.DataFrame:
    """
    Carga CSV de filtro con columnas CODIGO_FAMILIA y COD_SUBCATEGORIA.
    Retorna DataFrame con las series (familia, subcategoria) a calcular.
    """
    df = pd.read_csv(path, sep=";")
    df["CODIGO_FAMILIA"] = df["CODIGO_FAMILIA"].astype(int)
    df["COD_SUBCATEGORIA"] = df["COD_SUBCATEGORIA"].astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  CÁLCULO DE FEATURES PARA UNA FAMILIA
# ═══════════════════════════════════════════════════════════════════════════════

def compute_features_for_family(
    df_family: pd.DataFrame,
    family_code: int,
    fecha_corte: pd.Timestamp,
) -> pd.DataFrame:
    """
    Calcula todas las features para una familia dada, usando datos
    hasta fecha_corte (inclusive).
    Retorna un DataFrame con una fila por subcategoría.
    """
    df_family = df_family[df_family["DIM_PERIODO"] <= fecha_corte].copy()
    if df_family.empty:
        return pd.DataFrame()

    subcat_agg = (
        df_family.groupby("COD_SUBCATEGORIA")
        .agg(
            total_cantidad=("CANTIDAD_SUELTA", "sum"),
            total_venta_neta=("VENTA_NETA", "sum"),
            total_descuento=("DESCUENTO", "sum"),
            total_pvp=("PVP", "sum"),
            promedio_pvp=("PVP", "mean"),
            facturas_unicas=("DIM_FACTURA", "nunique"),
            registros=("DIM_FACTURA", "count"),
            primera_compra=("DIM_PERIODO", "min"),
            ultima_compra=("DIM_PERIODO", "max"),
        )
        .reset_index()
    )
    if subcat_agg.empty:
        return pd.DataFrame()

    subcat_agg = subcat_agg.sort_values("total_venta_neta", ascending=False).reset_index(drop=True)

    # ── Ciclos estacionales ──────────────────────────────────────────────
    ciclos_estacionales = calcular_ciclos_por_bloques(
        df_ventas=df_family,
        familia_id=family_code,
        today=fecha_corte,
        clase_de_calculo=1,
    )
    ciclos_debug = calcular_ciclos_por_bloques(
        df_ventas=df_family,
        familia_id=family_code,
        today=fecha_corte,
        clase_de_calculo=0,
    )
    if ciclos_estacionales.empty:
        return pd.DataFrame()

    # ── Features individuales ────────────────────────────────────────────
    recency_features = compute_recency_features(
        subcat_agg=subcat_agg,
        ciclos_estacionales=ciclos_estacionales,
        fecha_corte=fecha_corte,
    )
    freq_features = compute_frequency_features(
        df_family=df_family,
        ciclos_estacionales=ciclos_estacionales,
        fecha_corte=fecha_corte,
    )
    sow_features = compute_sow_features(
        df_family=df_family,
        ciclos_estacionales=ciclos_estacionales,
        fecha_corte=fecha_corte,
    )
    seasonality_features = compute_seasonality_features(
        df_family=df_family,
        ciclos_estacionales=ciclos_estacionales,
        fecha_corte=fecha_corte,
    )

    # ── Merge de todas las features ──────────────────────────────────────
    features_final = recency_features.merge(freq_features, on="COD_SUBCATEGORIA", how="left")
    features_final = features_final.merge(sow_features, on="COD_SUBCATEGORIA", how="left")
    features_final = features_final.merge(seasonality_features, on="COD_SUBCATEGORIA", how="left")
    features_final = features_final.merge(ciclos_estacionales, on="COD_SUBCATEGORIA", how="left")
    features_final = features_final.merge(ciclos_debug, on="COD_SUBCATEGORIA", how="left")
    features_final = features_final.fillna(0.0)

    # ── Score final (legacy, se mantiene por compatibilidad) ─────────────
    features_final["score_final"] = (
        0.4 * features_final["recencia_hl"]
        + 0.3 * features_final["freq_media"]
        + 0.1 * features_final["sow_24m"]
        + 0.2 * features_final["season_ratio"]
    )

    # ── Renombrado por origen (debug) ────────────────────────────────────
    score_columns = [
        "COD_SUBCATEGORIA",
        "recencia_hl",
        "freq_baja",
        "freq_media",
        "freq_alta",
        "cv_invertido",
        "sow_24m",
        "season_ratio",
        "score_final",
    ]

    rename_map = {}

    def _add_renames(cols, prefix):
        for col in cols:
            if col in features_final.columns and col not in score_columns:
                rename_map[col] = f"{prefix}_{col}"

    _add_renames(["tipo_ciclo_b", "ciclo_binario"], "Debug_ciclos")
    _add_renames(
        [
            "CODIGO_FAMILIA", "COD_SUBCATEGORIA", "ciclo_dias", "cv",
            "tipo_ciclo", "razon", "gaps_originales_dias", "gaps_normalizados",
            "gaps_ciclos_bloques", "ciclo_binario_c",
        ],
        "Ciclos",
    )
    _add_renames(
        ["COD_SUBCATEGORIA", "recencia_hl", "castigo_recencia",
         "l_compra_sobre_ciclo", "dias_desde_ultima_compra", "recencia"],
        "Recencia",
    )
    _add_renames(
        ["COD_SUBCATEGORIA", "freq_baja", "freq_media", "freq_alta",
         "cv_invertido", "compras_reales", "periodo_revision"],
        "Freq",
    )
    _add_renames(["COD_SUBCATEGORIA", "sow_24m", "transacciones_netas"], "Sow")
    _add_renames(
        ["COD_SUBCATEGORIA", "season_ratio", "compras_actual",
         "compras_pasado", "ratio_temporal"],
        "Seasonality",
    )

    features_final = features_final.rename(columns=rename_map)

    final_cols = list(score_columns) + list(rename_map.values())
    final_cols = list(dict.fromkeys(final_cols))
    final_cols = [c for c in final_cols if c in features_final.columns]

    features_final = features_final[final_cols]
    features_final["nucleo"] = family_code

    # ── ciclo_dias_mu (valor medio del ciclo) ────────────────────────────
    if "Ciclos_ciclo_dias" in features_final.columns:
        features_final["ciclo_dias_mu"] = features_final["Ciclos_ciclo_dias"].apply(
            lambda x: float(x[1]) if isinstance(x, (list, np.ndarray)) and len(x) >= 2 else 0.0
        )
    else:
        features_final["ciclo_dias_mu"] = 0.0

    return features_final


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE TARGET
# ═══════════════════════════════════════════════════════════════════════════════

def generate_target_for_family(
    df_family: pd.DataFrame,
    features_subcats: np.ndarray,
    fecha_corte_features: pd.Timestamp,
    fecha_max: pd.Timestamp,
) -> pd.DataFrame:
    """
    Target binario: 1 si la subcategoría fue comprada en
    (fecha_corte_features, fecha_max], 0 si no.
    """
    mask = (
        (df_family["DIM_PERIODO"] > fecha_corte_features)
        & (df_family["DIM_PERIODO"] <= fecha_max)
    )
    purchased = df_family.loc[mask, "COD_SUBCATEGORIA"].unique()

    target_df = pd.DataFrame({"COD_SUBCATEGORIA": features_subcats})
    target_df["target"] = target_df["COD_SUBCATEGORIA"].isin(purchased).astype(int)
    return target_df


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER PARALELO (top-level para pickle)
# ═══════════════════════════════════════════════════════════════════════════════

def _worker_family(args):
    """Procesa una familia: features + target. Recibe tupla."""
    df_family, family_code, fecha_max, fecha_corte_features = args

    feats = compute_features_for_family(df_family, family_code, fecha_corte_features)
    if feats.empty:
        return pd.DataFrame()

    target_df = generate_target_for_family(
        df_family=df_family,
        features_subcats=feats["COD_SUBCATEGORIA"].values,
        fecha_corte_features=fecha_corte_features,
        fecha_max=fecha_max,
    )
    feats = feats.merge(target_df, on="COD_SUBCATEGORIA", how="left")
    feats["target"] = feats["target"].fillna(0).astype(int)
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    historico_path: Path,
    filtro_path: Optional[Path] = None,
    prediction_window: int = PREDICTION_WINDOW_DAYS,
    n_workers: int = None,
    output_path: Path = None,
) -> pd.DataFrame:
    """
    Pipeline completo: carga histórico → calcula features + target → upsert.

    Args:
        historico_path: Path al CSV de transacciones.
        filtro_path: Path a CSV con columnas CODIGO_FAMILIA, COD_SUBCATEGORIA.
                     Si None, calcula para TODAS las series del histórico.
        prediction_window: Días de la ventana de target (default 21).
        n_workers: Procesos paralelos. None = cpu_count.
        output_path: Path del parquet de salida. None = FEATURES_TRAIN_FILE.
    """
    if n_workers is None:
        n_workers = os.cpu_count()
    if output_path is None:
        output_path = FEATURES_TRAIN_FILE

    # 1. Cargar histórico
    print(f"[Pipeline] Cargando histórico: {historico_path}")
    df_raw = load_historical_dataset(historico_path)

    fecha_min = df_raw["DIM_PERIODO"].min()
    fecha_max = df_raw["DIM_PERIODO"].max()
    fecha_corte_features = fecha_max - pd.Timedelta(days=prediction_window)

    print(f"[Pipeline] Registros: {len(df_raw)} | "
          f"Rango: {fecha_min.date()} → {fecha_max.date()}")
    print(f"[Pipeline] Features hasta: {fecha_corte_features.date()} | "
          f"Target: {fecha_corte_features.date()} → {fecha_max.date()} ({prediction_window}d)")

    # 2. Filtrar series si se pide
    if filtro_path is not None:
        series_df = load_filter_series(filtro_path)
        filter_families = set(series_df["CODIGO_FAMILIA"].unique())
        df_raw = df_raw[df_raw["CODIGO_FAMILIA"].isin(filter_families)]
        print(f"[Pipeline] Filtro aplicado: {len(filter_families)} familias, "
              f"{len(series_df)} series")

    families_list = sorted(df_raw["CODIGO_FAMILIA"].unique())
    total = len(families_list)
    print(f"[Pipeline] Familias a procesar: {total} | Workers: {n_workers}")

    # 3. Preparar tareas (una por familia)
    tasks = []
    for fam in families_list:
        df_fam = df_raw[df_raw["CODIGO_FAMILIA"] == fam].copy()
        if not df_fam.empty:
            tasks.append((df_fam, fam, fecha_max, fecha_corte_features))

    # 4. Ejecutar en paralelo
    results: List[pd.DataFrame] = []
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker_family, t): t[1] for t in tasks}
        for future in as_completed(futures):
            fam_code = futures[future]
            try:
                r = future.result()
                if not r.empty:
                    results.append(r)
            except Exception as e:
                print(f"  [ERROR] Familia {fam_code}: {e}")
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  Procesadas {done}/{len(tasks)} familias")

    if not results:
        print("[Pipeline] No se generaron features.")
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    # 5. Solo columnas esenciales (incluye tipo_ciclo para one-hot en training)
    from config import TIPO_CICLO_COL
    keep = ["nucleo", "COD_SUBCATEGORIA"] + FEATURE_COLUMNS + [TIPO_CICLO_COL] + ["target"]
    keep = [c for c in keep if c in combined.columns]
    new_features = combined[keep]

    # 6. Filtrar por series específicas si había filtro de subcategorías
    if filtro_path is not None:
        series_keys = set(zip(series_df["CODIGO_FAMILIA"], series_df["COD_SUBCATEGORIA"]))
        mask = new_features.apply(
            lambda r: (r["nucleo"], r["COD_SUBCATEGORIA"]) in series_keys, axis=1
        )
        new_features = new_features[mask].reset_index(drop=True)

    # 7. UPSERT: cargar existentes, reemplazar series recalculadas, agregar nuevas
    if output_path.exists():
        previous = pd.read_parquet(output_path)
        recalculated_keys = set(
            zip(new_features["nucleo"], new_features["COD_SUBCATEGORIA"])
        )
        mask_keep = previous.apply(
            lambda r: (r["nucleo"], r["COD_SUBCATEGORIA"]) not in recalculated_keys,
            axis=1,
        )
        previous = previous[mask_keep]
        result = pd.concat([previous, new_features], ignore_index=True)
        print(f"[Pipeline] Upsert: {len(new_features)} nuevas/actualizadas, "
              f"{len(previous)} conservadas → {len(result)} total")
    else:
        result = new_features
        print(f"[Pipeline] Archivo nuevo: {len(result)} filas")

    # 8. Guardar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"[Pipeline] Guardado: {output_path}")

    n_pos = int(result["target"].sum()) if "target" in result.columns else 0
    print(f"[Pipeline] {len(result)} filas | {result['nucleo'].nunique()} familias | "
          f"Target: {n_pos} positivos ({n_pos/len(result)*100:.1f}%)")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Engineering (paralelo + upsert)")
    parser.add_argument("--historico", type=str, required=True,
                        help="Path al CSV de transacciones históricas")
    parser.add_argument("--filtro", type=str, default=None,
                        help="Path a CSV con CODIGO_FAMILIA;COD_SUBCATEGORIA para filtrar series")
    parser.add_argument("--prediction-window", type=int, default=PREDICTION_WINDOW_DAYS)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Path del parquet de salida. Default: features_train.parquet")
    args = parser.parse_args()

    run_pipeline(
        historico_path=Path(args.historico),
        filtro_path=Path(args.filtro) if args.filtro else None,
        prediction_window=args.prediction_window,
        n_workers=args.workers,
        output_path=Path(args.output) if args.output else None,
    )
