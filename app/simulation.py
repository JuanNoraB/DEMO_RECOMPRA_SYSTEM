"""
simulation.py — Utilidades para crear subconjuntos del histórico para pruebas.

Funciones:
  - sample_series: Toma N series de tiempo aleatorias (familia+subcategoria).
  - sample_series_trimmed: Toma N series pero quitando M meses del final.
  - split_historico: Divide en base (sin últimos M meses) + simulación (últimos M meses).

Uso CLI:
  python simulation.py sample --n 1000 --historico Historico.csv --output data/raw/historico_1000.csv
  python simulation.py sample --n 1000 --trim-months 3 --historico Historico.csv --output data/raw/historico_1000_trim3.csv
  python simulation.py split  --trim-months 2 --historico data/raw/historico_1000.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from feature_engineering import load_historical_dataset


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES DE SIMULACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def sample_series(
    historico_path: Path,
    n: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Toma N series de tiempo aleatorias del histórico.
    Una serie = (CODIGO_FAMILIA, COD_SUBCATEGORIA) única.
    Retorna el subset del histórico que contiene esas N series.
    """
    df = load_historical_dataset(historico_path)

    series = (
        df.groupby(["CODIGO_FAMILIA", "COD_SUBCATEGORIA"])
        .size()
        .reset_index(name="count")
    )
    n_available = len(series)
    n_sample = min(n, n_available)
    print(f"[Sample] Series disponibles: {n_available} | Seleccionando: {n_sample}")

    sampled = series.sample(n=n_sample, random_state=seed)
    keys = set(zip(sampled["CODIGO_FAMILIA"], sampled["COD_SUBCATEGORIA"]))

    df_out = df[
        df.apply(lambda r: (r["CODIGO_FAMILIA"], r["COD_SUBCATEGORIA"]) in keys, axis=1)
    ].copy()

    print(f"[Sample] Registros: {len(df_out)} | Familias: {df_out['CODIGO_FAMILIA'].nunique()} | "
          f"Series: {n_sample}")
    return df_out


def sample_series_trimmed(
    historico_path: Path,
    n: int,
    trim_months: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Igual que sample_series pero primero quita los últimos trim_months meses.
    """
    df = load_historical_dataset(historico_path)
    fecha_max = df["DIM_PERIODO"].max()
    fecha_corte = fecha_max - pd.DateOffset(months=trim_months)

    df = df[df["DIM_PERIODO"] < fecha_corte].copy()
    print(f"[SampleTrimmed] Quitados últimos {trim_months} meses "
          f"(corte: {fecha_corte.date()}) | Registros restantes: {len(df)}")

    series = (
        df.groupby(["CODIGO_FAMILIA", "COD_SUBCATEGORIA"])
        .size()
        .reset_index(name="count")
    )
    n_available = len(series)
    n_sample = min(n, n_available)

    sampled = series.sample(n=n_sample, random_state=seed)
    keys = set(zip(sampled["CODIGO_FAMILIA"], sampled["COD_SUBCATEGORIA"]))

    df_out = df[
        df.apply(lambda r: (r["CODIGO_FAMILIA"], r["COD_SUBCATEGORIA"]) in keys, axis=1)
    ].copy()

    print(f"[SampleTrimmed] Registros: {len(df_out)} | Familias: {df_out['CODIGO_FAMILIA'].nunique()} | "
          f"Series: {n_sample} | Rango: {df_out['DIM_PERIODO'].min().date()} → {df_out['DIM_PERIODO'].max().date()}")
    return df_out


def split_historico(
    historico_path: Path,
    trim_months: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el histórico en:
      - base: todo menos los últimos trim_months meses
      - simulación: los últimos trim_months meses

    Guarda ambos CSVs junto al original con sufijos _base y _sim.
    """
    df = load_historical_dataset(historico_path)
    fecha_max = df["DIM_PERIODO"].max()
    fecha_corte = fecha_max - pd.DateOffset(months=trim_months)

    base = df[df["DIM_PERIODO"] < fecha_corte].copy()
    sim = df[df["DIM_PERIODO"] >= fecha_corte].copy()

    stem = historico_path.stem
    parent = historico_path.parent

    base_path = parent / f"{stem}_base.csv"
    sim_path = parent / f"{stem}_sim.csv"

    base.to_csv(base_path, index=False, sep=";")
    sim.to_csv(sim_path, index=False, sep=";")

    print(f"[Split] Base: {len(base)} registros | {base['CODIGO_FAMILIA'].nunique()} familias | "
          f"{base['DIM_PERIODO'].min().date()} → {base['DIM_PERIODO'].max().date()}")
    print(f"[Split] Sim:  {len(sim)} registros | {sim['CODIGO_FAMILIA'].nunique()} familias | "
          f"{sim['DIM_PERIODO'].min().date()} → {sim['DIM_PERIODO'].max().date()}")
    print(f"[Split] Guardados: {base_path} | {sim_path}")
    return base, sim


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulación — subsets del histórico")
    sub = parser.add_subparsers(dest="command")

    # sample
    p_sample = sub.add_parser("sample", help="Tomar N series aleatorias")
    p_sample.add_argument("--n", type=int, required=True, help="Número de series")
    p_sample.add_argument("--historico", type=str, required=True)
    p_sample.add_argument("--output", type=str, required=True, help="Path del CSV de salida")
    p_sample.add_argument("--trim-months", type=int, default=0,
                          help="Si > 0, quita los últimos M meses antes de samplear")
    p_sample.add_argument("--seed", type=int, default=42)

    # split
    p_split = sub.add_parser("split", help="Dividir en base + simulación")
    p_split.add_argument("--historico", type=str, required=True)
    p_split.add_argument("--trim-months", type=int, default=2)

    args = parser.parse_args()

    if args.command == "sample":
        if args.trim_months > 0:
            df_out = sample_series_trimmed(
                Path(args.historico), args.n, args.trim_months, args.seed
            )
        else:
            df_out = sample_series(Path(args.historico), args.n, args.seed)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out, index=False, sep=";")
        print(f"[Sample] Guardado: {out}")

    elif args.command == "split":
        split_historico(Path(args.historico), args.trim_months)

    else:
        parser.print_help()
