"""
csv_to_parquet.py — Convierte CSV histórico grande a Parquet mínimo.

Uso:
  python data/raw/csv_to_parquet.py \
    --csv data/raw/20230101_20251210_C.csv \
    --output data/raw/20230101_20251210_C_min.parquet \
    --chunksize 2000000 \
    --force
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REQUIRED_COLUMNS = [
    "DIM_PERIODO",
    "CODIGO_FAMILIA",
    "COD_SUBCATEGORIA",
    "DIM_FACTURA",
    "CANTIDAD_SUELTA",
    "PVP",
    "VENTA_NETA",
    "DESCUENTO",
]

FLOAT_COLUMNS = ["CANTIDAD_SUELTA", "PVP", "VENTA_NETA", "DESCUENTO"]


def parse_dates_safe(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%d-%b-%y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], errors="coerce")
    return parsed


def normalize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in chunk.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {missing}")

    df = chunk[REQUIRED_COLUMNS].copy()
    df["DIM_PERIODO"] = parse_dates_safe(df["DIM_PERIODO"])

    for col in ["CODIGO_FAMILIA", "COD_SUBCATEGORIA", "DIM_FACTURA"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["DIM_PERIODO", "CODIGO_FAMILIA", "COD_SUBCATEGORIA"])

    df["CODIGO_FAMILIA"] = df["CODIGO_FAMILIA"].astype("int64")
    df["COD_SUBCATEGORIA"] = df["COD_SUBCATEGORIA"].astype("int32")
    df["DIM_FACTURA"] = df["DIM_FACTURA"].fillna(0).astype("int64")

    for col in FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")

    return df


def convert_csv_to_parquet(csv_path: Path, output_path: Path, chunksize: int, sep: str, encoding: str, compression: str) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe CSV: {csv_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total_rows = 0
    total_raw = 0
    t0 = time.time()

    try:
        reader = pd.read_csv(
            csv_path,
            sep=sep,
            encoding=encoding,
            usecols=lambda c: c in REQUIRED_COLUMNS,
            chunksize=chunksize,
            low_memory=False,
        )

        for i, chunk in enumerate(reader, start=1):
            total_raw += len(chunk)
            df = normalize_chunk(chunk)
            table = pa.Table.from_pandas(df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression=compression)

            writer.write_table(table)
            total_rows += len(df)
            elapsed = time.time() - t0
            print(f"[Chunk {i}] raw={len(chunk):,} limpio={len(df):,} total={total_rows:,} elapsed={elapsed/60:.1f} min")

    finally:
        if writer is not None:
            writer.close()

    print("=" * 80)
    print(f"CSV leído: {csv_path}")
    print(f"Parquet generado: {output_path}")
    print(f"Filas raw: {total_raw:,}")
    print(f"Filas limpias: {total_rows:,}")
    print(f"Duración: {(time.time() - t0)/60:.1f} min")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convertir CSV histórico a Parquet mínimo")
    parser.add_argument("--csv", required=True, type=str, help="Path del CSV histórico")
    parser.add_argument("--output", required=True, type=str, help="Path del parquet de salida")
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    parser.add_argument("--sep", type=str, default=";")
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--compression", type=str, default="snappy")
    parser.add_argument("--force", action="store_true", help="Sobrescribir output si existe")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_path = Path(args.output)

    if output_path.exists() and not args.force:
        raise FileExistsError(f"Ya existe {output_path}. Usa --force para sobrescribir.")
    if output_path.exists() and args.force:
        output_path.unlink()

    convert_csv_to_parquet(
        csv_path=csv_path,
        output_path=output_path,
        chunksize=args.chunksize,
        sep=args.sep,
        encoding=args.encoding,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
