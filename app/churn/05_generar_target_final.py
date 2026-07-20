from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORICO = REPO_ROOT / "data" / "raw" / "HISTORICO_300K_FULL.parquet"
DEFAULT_FEATURES = REPO_ROOT / "data" / "churn" / "caracteristicas_churn.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "churn" / "dataset_churn_final.parquet"
DEFAULT_LOG = REPO_ROOT / "data" / "churn" / "target_final.log"

COLS_HIST = ["IDENTIFICACION", "DIM_PERIODO"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generacion e integracion del target final de churn")
    p.add_argument("--historico", type=Path, default=DEFAULT_HISTORICO)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--alpha", type=float, default=2.25)
    p.add_argument("--gamma", type=float, default=1.25)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.alpha <= 0:
        raise ValueError("--alpha debe ser mayor que 0")
    if args.gamma < 0:
        raise ValueError("--gamma no puede ser negativo")

    started = time.perf_counter()

    historico_path = args.historico.expanduser().resolve()
    features_path = args.features.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    log_path = args.log.expanduser().resolve()

    print(f"[1/5] Leyendo features: {features_path}")
    features = pd.read_parquet(features_path)
    if "IDENTIFICACION" not in features.columns:
        raise ValueError("El dataset de caracteristicas no contiene IDENTIFICACION")

    features["IDENTIFICACION"] = features["IDENTIFICACION"].astype("string").str.strip()
    if features["IDENTIFICACION"].duplicated().any():
        raise AssertionError("El dataset de caracteristicas contiene IDENTIFICACION duplicada")

    ids_features = set(features["IDENTIFICACION"].dropna().unique())
    print(f"   clientes en features: {len(ids_features):,}")

    print(f"\n[2/5] Leyendo historico completo: {historico_path}")
    df = pd.read_parquet(historico_path, columns=COLS_HIST)
    df["IDENTIFICACION"] = df["IDENTIFICACION"].astype("string").str.strip()
    df["DIM_PERIODO"] = pd.to_datetime(df["DIM_PERIODO"], errors="coerce")
    df = df.dropna(subset=COLS_HIST)

    # Se trabaja unicamente con los clientes ya presentes en el dataset de caracteristicas.
    df = df[df["IDENTIFICACION"].isin(ids_features)].copy()

    # Misma logica temporal del script 02: una fecha distinta por cliente.
    df = (
        df.drop_duplicates(subset=["IDENTIFICACION", "DIM_PERIODO"])
        .sort_values(["IDENTIFICACION", "DIM_PERIODO"])
        .reset_index(drop=True)
    )

    fecha_max = df["DIM_PERIODO"].max()
    T = fecha_max - pd.DateOffset(years=1)
    hist_start = T - pd.DateOffset(years=2) + pd.Timedelta(days=1)

    hist = df[df["DIM_PERIODO"].between(hist_start, T)].copy()
    future = df[df["DIM_PERIODO"] > T].copy()

    print(f"   fecha maxima dataset : {fecha_max.date()}")
    print(f"   fecha de corte T     : {T.date()}")
    print(f"   inicio historico     : {hist_start.date()}")
    print(f"   clientes filtrados   : {hist['IDENTIFICACION'].nunique():,}")

    print("\n[3/5] Calculando horizonte hibrido y target ...")

    hist["gap_dias"] = hist.groupby("IDENTIFICACION", sort=False)["DIM_PERIODO"].diff().dt.days

    perfil = (
        hist.groupby("IDENTIFICACION", sort=False)
        .agg(
            n_compras_historicas=("DIM_PERIODO", "size"),
            mediana_cliente=("gap_dias", "median"),
        )
        .reset_index()
        .dropna(subset=["mediana_cliente"])
    )

    if len(perfil) != len(ids_features):
        faltantes = len(ids_features) - len(perfil)
        raise AssertionError(
            f"No se pudo construir el perfil temporal para todos los clientes. Faltantes: {faltantes:,}"
        )

    # Referencia global exactamente como en el script 02.
    B = float(perfil["mediana_cliente"].mean())

    primera_futura = (
        future.groupby("IDENTIFICACION", sort=False)["DIM_PERIODO"]
        .min()
        .rename("primera_compra_futura")
        .reset_index()
    )

    target_df = perfil.merge(primera_futura, on="IDENTIFICACION", how="left")

    H = args.alpha * B + args.gamma * target_df["mediana_cliente"]
    H = H.clip(lower=1.0)
    target_df["H_dias"] = np.ceil(H).astype(int)
    target_df["limite_H"] = T + pd.to_timedelta(target_df["H_dias"], unit="D")

    target_df["target"] = (
        target_df["primera_compra_futura"].isna()
        | (target_df["primera_compra_futura"] > target_df["limite_H"])
    ).astype("int8")

    seguimiento_incompleto = int((target_df["limite_H"] > fecha_max).sum())

    print(f"   alpha                 : {args.alpha:.2f}")
    print(f"   gamma                 : {args.gamma:.2f}")
    print(f"   B media medianas dias : {B:.2f}")
    print(f"   H media dias          : {target_df['H_dias'].mean():.2f}")
    print(f"   H mediana dias        : {target_df['H_dias'].median():.2f}")
    print(f"   seguimiento incompleto: {seguimiento_incompleto:,}")

    print("\n[4/5] Integrando target con caracteristicas ...")

    target_minimo = target_df[["IDENTIFICACION", "target"]].copy()
    final = features.merge(
        target_minimo,
        on="IDENTIFICACION",
        how="inner",
        validate="one_to_one",
    )

    if len(final) != len(features):
        raise AssertionError(
            f"El dataset final debe conservar {len(features):,} clientes y obtuvo {len(final):,}"
        )
    if final["target"].isna().any():
        raise AssertionError("Existen targets nulos en el dataset final")

    target_counts = final["target"].value_counts().sort_index()
    n_target_0 = int(target_counts.get(0, 0))
    n_target_1 = int(target_counts.get(1, 0))
    pct_target_1 = n_target_1 / len(final) * 100.0

    print(f"   filas finales          : {len(final):,}")
    print(f"   clientes unicos        : {final['IDENTIFICACION'].nunique():,}")
    print(f"   target 0               : {n_target_0:,}")
    print(f"   target 1               : {n_target_1:,}")
    print(f"   churn %                : {pct_target_1:.2f}%")

    print("\n[5/5] Guardando dataset final y log ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    final.to_parquet(output_path, index=False)

    resumen = (
        "=== TARGET FINAL CHURN ===\n"
        f"historico: {historico_path}\n"
        f"features: {features_path}\n"
        f"output: {output_path}\n"
        f"hist: {hist_start.date()} -> {T.date()}\n"
        f"Dmax: {fecha_max.date()}\n"
        f"clientes: {len(final)}\n"
        f"alpha: {args.alpha:.4f}\n"
        f"gamma: {args.gamma:.4f}\n"
        f"B_media_medianas_dias: {B:.4f}\n"
        f"H_media_dias: {target_df['H_dias'].mean():.4f}\n"
        f"H_mediana_dias: {target_df['H_dias'].median():.4f}\n"
        f"target_0: {n_target_0}\n"
        f"target_1: {n_target_1}\n"
        f"churn_pct: {pct_target_1:.4f}\n"
        f"seguimiento_incompleto: {seguimiento_incompleto}\n"
        f"tiempo_total_s: {time.perf_counter() - started:.2f}\n"
    )

    log_path.write_text(resumen, encoding="utf-8")

    print(f"   dataset: {output_path}")
    print(f"   tamanio: {output_path.stat().st_size / 1e6:.1f} MB")
    print(f"   log: {log_path}")


if __name__ == "__main__":
    main()
