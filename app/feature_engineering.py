"""
feature_engineering.py — Cálculo de features + target.

Uso:
  python feature_engineering.py --historico data/raw/historico_base.csv
  python feature_engineering.py --historico data/raw/historico_base.parquet
  python feature_engineering.py --historico data/raw/historico_base.parquet --workers 128 --batch-size 10000 --skip-final-concat

El script:
  1. Lee histórico CSV o Parquet.
  2. Toma fecha_min y fecha_max del histórico automáticamente.
  3. Features se calculan con datos hasta (fecha_max - 21 días).
  4. Target = ¿hubo compra en los últimos 21 días? (binario por serie).
  5. Si se pasa --filtro, solo calcula para esas series (familia, subcategoria).
  6. Procesa familias por lotes para evitar acumular todo en RAM.
  7. Guarda parquets parciales por lote.
  8. Opcionalmente concatena los parciales en el parquet final.
"""
from __future__ import annotations

import gc
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config import (
    FEATURES_TRAIN_FILE,
    HISTORICO_FILE,
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
    compute_units_features,
)

MIN_REQUIRED_COLUMNS = [
    "DIM_PERIODO",
    "CODIGO_FAMILIA",
    "COD_SUBCATEGORIA",
    "DIM_FACTURA",
    "CANTIDAD_SUELTA",
    "PVP",
    "VENTA_NETA",
    "DESCUENTO",
]


def _parse_dates_safe(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%d-%b-%y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], errors="coerce")
    return parsed


def _normalize_historical_df(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [c for c in MIN_REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas en histórico: {missing_cols}")

    df = df[MIN_REQUIRED_COLUMNS].copy()
    df["DIM_PERIODO"] = _parse_dates_safe(df["DIM_PERIODO"])

    for col in ["CODIGO_FAMILIA", "COD_SUBCATEGORIA", "DIM_FACTURA"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["DIM_PERIODO", "CODIGO_FAMILIA", "COD_SUBCATEGORIA"])

    df["CODIGO_FAMILIA"] = df["CODIGO_FAMILIA"].astype("int64")
    df["COD_SUBCATEGORIA"] = df["COD_SUBCATEGORIA"].astype("int32")
    df["DIM_FACTURA"] = df["DIM_FACTURA"].fillna(0).astype("int64")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")

    return df


def load_historical_dataset(path: Path) -> pd.DataFrame:
    """Carga y limpia transacciones históricas desde CSV o Parquet."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(
            path,
            encoding="utf-8",
            sep=";",
            usecols=lambda c: c in MIN_REQUIRED_COLUMNS,
            low_memory=False,
        )
    else:
        raise ValueError(f"Formato no soportado: {path.suffix}. Use .csv o .parquet")

    return _normalize_historical_df(df)


def load_filter_series(path: Path) -> pd.DataFrame:
    """
    Carga CSV de filtro con columnas CODIGO_FAMILIA y COD_SUBCATEGORIA.
    Retorna DataFrame con las series (familia, subcategoria) a calcular.
    """
    df = pd.read_csv(path, sep=";")
    df["CODIGO_FAMILIA"] = df["CODIGO_FAMILIA"].astype(int)
    df["COD_SUBCATEGORIA"] = df["COD_SUBCATEGORIA"].astype(int)
    return df


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

    subcat_agg["ticket_promedio"] = (
        subcat_agg["total_venta_neta"] / subcat_agg["facturas_unicas"].replace(0, np.nan)
    ).fillna(0.0)

    n_subcats_familia = int(df_family["COD_SUBCATEGORIA"].nunique())

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
    units_features = compute_units_features(
        df_family=df_family,
        fecha_corte=fecha_corte,
    )

    features_final = recency_features.merge(freq_features, on="COD_SUBCATEGORIA", how="left")
    features_final = features_final.merge(sow_features, on="COD_SUBCATEGORIA", how="left")
    features_final = features_final.merge(seasonality_features, on="COD_SUBCATEGORIA", how="left")
    features_final = features_final.merge(ciclos_estacionales, on="COD_SUBCATEGORIA", how="left")
    features_final = features_final.merge(ciclos_debug, on="COD_SUBCATEGORIA", how="left")
    features_final = features_final.merge(units_features, on="COD_SUBCATEGORIA", how="left")
    features_final = features_final.merge(
        subcat_agg[["COD_SUBCATEGORIA", "ticket_promedio"]],
        on="COD_SUBCATEGORIA",
        how="left",
    )
    features_final["n_subcats_familia"] = n_subcats_familia
    features_final = features_final.fillna(0.0)

    features_final["score_final"] = (
        0.4 * features_final["recencia_hl"]
        + 0.3 * features_final["freq_media"]
        + 0.1 * features_final["sow_24m"]
        + 0.2 * features_final["season_ratio"]
    )

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
        "dias_desde_ultima_compra",
        "l_compra_sobre_ciclo",
        "compras_reales",
        "ratio_temporal",
        "ticket_promedio",
        "n_subcats_familia",
        "total_compras_12m",
        "avg_unidades",
        "ratio_ultimo_vs_prom",
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

    if "Ciclos_ciclo_dias" in features_final.columns:
        features_final["ciclo_dias_mu"] = features_final["Ciclos_ciclo_dias"].apply(
            lambda x: float(x[1]) if isinstance(x, (list, np.ndarray)) and len(x) >= 2 else 0.0
        )
    else:
        features_final["ciclo_dias_mu"] = 0.0

    _list_cols = [
        c for c in features_final.columns
        if features_final[c].dtype == object
        and len(features_final[c].dropna()) > 0
        and isinstance(features_final[c].dropna().iloc[0], (list, np.ndarray))
    ]
    if _list_cols:
        features_final = features_final.drop(columns=_list_cols)

    return features_final


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


def _filter_exact_series(df_features: pd.DataFrame, series_keys: Set[Tuple[int, int]]) -> pd.DataFrame:
    """Filtra resultados a pares exactos (familia/nucleo, subcategoria) si se usa --filtro."""
    if df_features.empty:
        return df_features
    mask = [
        (int(nucleo), int(subcat)) in series_keys
        for nucleo, subcat in zip(df_features["nucleo"], df_features["COD_SUBCATEGORIA"])
    ]
    return df_features.loc[mask].reset_index(drop=True)


def _write_run_metadata(
    output_path: Path,
    fecha_corte_features: pd.Timestamp,
    prediction_window: int,
    historico_path: Path,
    parts_dir: Path,
    skip_final_concat: bool,
) -> None:
    _run_meta = {
        "fecha_corte": str(fecha_corte_features.date()),
        "prediction_window_days": prediction_window,
        "historico_path": str(historico_path),
        "parts_dir": str(parts_dir),
        "skip_final_concat": skip_final_concat,
    }
    _meta_path = output_path.with_suffix(".meta.json")
    with open(_meta_path, "w") as _mf:
        json.dump(_run_meta, _mf, indent=2)


def run_pipeline(
    historico_path: Path,
    filtro_path: Optional[Path] = None,
    prediction_window: int = PREDICTION_WINDOW_DAYS,
    n_workers: int = None,
    output_path: Path = None,
    batch_size: int = 10000,
    parts_dir: Optional[Path] = None,
    skip_final_concat: bool = False,
    resume: bool = False,
) -> pd.DataFrame:
    """
    Pipeline completo: carga histórico → calcula features + target → guarda partes.

    Para datasets grandes, usar --skip-final-concat y luego data/raw/concat_feature_parts.py.
    """
    t_pipeline_start = time.time()
    if n_workers is None:
        n_workers = os.cpu_count()
    if output_path is None:
        output_path = FEATURES_TRAIN_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if parts_dir is None:
        parts_dir = output_path.parent / f"{output_path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    if not resume:
        for old_part in parts_dir.glob("part_*.parquet"):
            old_part.unlink()

    print(f"[Pipeline] Cargando histórico: {historico_path}")
    df_raw = load_historical_dataset(historico_path)

    fecha_min = df_raw["DIM_PERIODO"].min()
    fecha_max = df_raw["DIM_PERIODO"].max()
    fecha_corte_features = fecha_max - pd.Timedelta(days=prediction_window)

    print(f"[Pipeline] Registros: {len(df_raw):,} | Rango: {fecha_min.date()} → {fecha_max.date()}")
    print(f"[Pipeline] Features hasta: {fecha_corte_features.date()} | Target: {fecha_corte_features.date()} → {fecha_max.date()} ({prediction_window}d)")

    series_keys = None
    if filtro_path is not None:
        series_df = load_filter_series(filtro_path)
        filter_families = set(series_df["CODIGO_FAMILIA"].unique())
        df_raw = df_raw[df_raw["CODIGO_FAMILIA"].isin(filter_families)].copy()
        series_keys = set(zip(series_df["CODIGO_FAMILIA"], series_df["COD_SUBCATEGORIA"]))
        print(f"[Pipeline] Filtro aplicado: {len(filter_families):,} familias, {len(series_df):,} series")

    families_list = sorted(df_raw["CODIGO_FAMILIA"].unique())
    total = len(families_list)
    print(f"[Pipeline] Familias a procesar: {total:,} | Workers: {n_workers} | Batch size: {batch_size:,}")
    print(f"[Pipeline] Partes en: {parts_dir}")

    if total == 0:
        print("[Pipeline] No hay familias para procesar.")
        return pd.DataFrame()

    part_files: List[Path] = []
    total_rows_written = 0
    total_pos_written = 0
    processed_families = 0

    for batch_idx, start in enumerate(range(0, total, batch_size)):
        batch_families = families_list[start:start + batch_size]
        end = min(start + batch_size, total)
        part_file = parts_dir / f"part_{batch_idx:05d}.parquet"

        if resume and part_file.exists():
            print(f"[Batch {batch_idx:05d}] Ya existe, se omite: {part_file}")
            part_files.append(part_file)
            processed_families += len(batch_families)
            continue

        print()
        print("=" * 80)
        print(f"[Batch {batch_idx:05d}] Familias {start + 1:,}–{end:,} de {total:,}")
        print("=" * 80)

        batch_set = set(batch_families)
        df_batch = df_raw[df_raw["CODIGO_FAMILIA"].isin(batch_set)]
        if df_batch.empty:
            print(f"[Batch {batch_idx:05d}] Sin registros. Saltando.")
            continue

        tasks = []
        for fam, df_fam in df_batch.groupby("CODIGO_FAMILIA", sort=False):
            if not df_fam.empty:
                tasks.append((df_fam.copy(), int(fam), fecha_max, fecha_corte_features))

        batch_results = []
        done = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker_family, t): t[1] for t in tasks}
            for future in as_completed(futures):
                fam_code = futures[future]
                try:
                    r = future.result()
                    if not r.empty:
                        batch_results.append(r)
                except Exception as e:
                    print(f"  [ERROR] Familia {fam_code}: {e}")

                done += 1
                if done % 50 == 0 or done == len(tasks):
                    global_done = processed_families + done
                    print(f"  Procesadas {global_done:,}/{total:,} familias (batch {done:,}/{len(tasks):,})")

        if batch_results:
            batch_df = pd.concat(batch_results, ignore_index=True)
            if series_keys is not None:
                batch_df = _filter_exact_series(batch_df, series_keys)

            if not batch_df.empty:
                batch_df.to_parquet(part_file, index=False)
                part_files.append(part_file)
                rows = len(batch_df)
                positives = int(batch_df["target"].sum()) if "target" in batch_df.columns else 0
                total_rows_written += rows
                total_pos_written += positives
                print(f"[Batch {batch_idx:05d}] Guardado: {rows:,} filas | target+={positives:,} → {part_file}")
            else:
                print(f"[Batch {batch_idx:05d}] Sin filas luego de filtro exacto.")
        else:
            print(f"[Batch {batch_idx:05d}] Sin resultados.")

        processed_families += len(batch_families)
        del df_batch, tasks, batch_results
        if "batch_df" in locals():
            del batch_df
        gc.collect()

    if not part_files:
        print("[Pipeline] No se generaron partes.")
        return pd.DataFrame()

    _write_run_metadata(
        output_path=output_path,
        fecha_corte_features=fecha_corte_features,
        prediction_window=prediction_window,
        historico_path=historico_path,
        parts_dir=parts_dir,
        skip_final_concat=skip_final_concat,
    )

    if skip_final_concat:
        duracion_s = time.time() - t_pipeline_start
        n_neg = total_rows_written - total_pos_written
        logs_dir = output_path.parent.parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "feature_engineering_runs.jsonl"
        run_log = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "duracion_min": round(duracion_s / 60, 2),
            "duracion_seg": round(duracion_s, 1),
            "num_familias": int(total),
            "num_pares_fam_subcat": int(total_rows_written),
            "target_1_positivos": int(total_pos_written),
            "target_0_negativos": int(n_neg),
            "workers": n_workers,
            "batch_size": batch_size,
            "prediction_window_days": prediction_window,
            "historico_path": str(historico_path),
            "output_path": str(output_path),
            "parts_dir": str(parts_dir),
            "num_parts": len(part_files),
            "con_filtro": filtro_path is not None,
            "skip_final_concat": True,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_log, ensure_ascii=False) + "\n")

        print()
        print("=" * 80)
        print("[Pipeline] PARTES GENERADAS. Concat final omitido por --skip-final-concat.")
        print(f"[Pipeline] Partes: {len(part_files)} | Filas escritas: {total_rows_written:,}")
        print(f"[Pipeline] Usa concat_feature_parts.py para crear: {output_path}")
        print(f"[Pipeline] Log guardado: {log_file}")
        print(f"[Pipeline] Duración total: {duracion_s/60:.1f} min ({duracion_s:.0f}s)")
        print("=" * 80)
        return pd.DataFrame()

    print()
    print("=" * 80)
    print("[Pipeline] Concatenando partes en memoria para parquet final...")
    print("=" * 80)

    new_features = pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)

    if output_path.exists():
        previous = pd.read_parquet(output_path)
        recalculated_keys = set(zip(new_features["nucleo"], new_features["COD_SUBCATEGORIA"]))
        mask_keep = previous.apply(
            lambda r: (r["nucleo"], r["COD_SUBCATEGORIA"]) not in recalculated_keys,
            axis=1,
        )
        previous = previous[mask_keep]
        result = pd.concat([previous, new_features], ignore_index=True)
        print(f"[Pipeline] Upsert: {len(new_features):,} nuevas/actualizadas, {len(previous):,} conservadas → {len(result):,} total")
    else:
        result = new_features
        print(f"[Pipeline] Archivo nuevo: {len(result):,} filas")

    result.to_parquet(output_path, index=False)
    print(f"[Pipeline] Guardado: {output_path}")

    n_pos = int(result["target"].sum()) if "target" in result.columns else 0
    n_neg = len(result) - n_pos
    print(f"[Pipeline] {len(result):,} filas | {result['nucleo'].nunique():,} familias | Target: {n_pos:,} positivos ({n_pos/len(result)*100:.1f}%)")

    duracion_s = time.time() - t_pipeline_start
    feature_cols = [c for c in result.columns if c not in ("nucleo", "COD_SUBCATEGORIA", "target")]
    series_con_ciclo = 0
    if "Debug_ciclos_tipo_ciclo_b" in result.columns:
        series_con_ciclo = int((result["Debug_ciclos_tipo_ciclo_b"] != "no_ciclico").sum())

    run_log = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "duracion_min": round(duracion_s / 60, 2),
        "duracion_seg": round(duracion_s, 1),
        "num_familias": int(result["nucleo"].nunique()),
        "num_pares_fam_subcat": len(result),
        "num_features": len(feature_cols),
        "features_calculadas": feature_cols,
        "target_1_positivos": n_pos,
        "target_0_negativos": n_neg,
        "series_con_ciclo": series_con_ciclo,
        "workers": n_workers,
        "batch_size": batch_size,
        "prediction_window_days": prediction_window,
        "historico_path": str(historico_path),
        "output_path": str(output_path),
        "parts_dir": str(parts_dir),
        "num_parts": len(part_files),
        "con_filtro": filtro_path is not None,
        "skip_final_concat": False,
    }

    logs_dir = output_path.parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "feature_engineering_runs.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_log, ensure_ascii=False) + "\n")
    print(f"[Pipeline] Log guardado: {log_file}")
    print(f"[Pipeline] Duración total: {duracion_s/60:.1f} min ({duracion_s:.0f}s)")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Engineering (paralelo + batches + parquet)")
    parser.add_argument("--historico", type=str, required=True,
                        help="Path al CSV o Parquet de transacciones históricas")
    parser.add_argument("--filtro", type=str, default=None,
                        help="Path a CSV con CODIGO_FAMILIA;COD_SUBCATEGORIA para filtrar series")
    parser.add_argument("--prediction-window", type=int, default=PREDICTION_WINDOW_DAYS)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="Familias por batch. Default: 10000")
    parser.add_argument("--parts-dir", type=str, default=None,
                        help="Directorio para parquets parciales. Default: <output_stem>_parts")
    parser.add_argument("--skip-final-concat", action="store_true",
                        help="Solo genera partes; no concatena al parquet final dentro de este script")
    parser.add_argument("--resume", action="store_true",
                        help="Reusa part_*.parquet ya existentes y continúa con los faltantes")
    parser.add_argument("--output", type=str, default=None,
                        help="Path del parquet de salida. Default: features_train.parquet")
    args = parser.parse_args()

    run_pipeline(
        historico_path=Path(args.historico),
        filtro_path=Path(args.filtro) if args.filtro else None,
        prediction_window=args.prediction_window,
        n_workers=args.workers,
        output_path=Path(args.output) if args.output else None,
        batch_size=args.batch_size,
        parts_dir=Path(args.parts_dir) if args.parts_dir else None,
        skip_final_concat=args.skip_final_concat,
        resume=args.resume,
    )
