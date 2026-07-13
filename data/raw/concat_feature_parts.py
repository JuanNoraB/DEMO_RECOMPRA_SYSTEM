"""
concat_feature_parts.py — Une parquets parciales generados por feature_engineering.py.

Uso:
  python data/raw/concat_feature_parts.py \
    --parts-dir data/features_store/features_train_w0_parts \
    --output data/features_store/features_train_w0.parquet \
    --force

No carga todas las partes en pandas a la vez; escribe con PyArrow por partes.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq


def concat_parts(parts_dir: Path, output_path: Path, pattern: str, compression: str, force: bool) -> None:
    if not parts_dir.exists():
        raise FileNotFoundError(f"No existe parts-dir: {parts_dir}")

    part_files = sorted(parts_dir.glob(pattern))
    if not part_files:
        raise FileNotFoundError(f"No se encontraron partes con patrón {pattern} en {parts_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if not force:
            raise FileExistsError(f"Ya existe {output_path}. Usa --force para sobrescribir.")
        output_path.unlink()

    writer = None
    total_rows = 0
    t0 = time.time()

    try:
        for i, part in enumerate(part_files, start=1):
            table = pq.read_table(part)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
            writer.write_table(table)
            total_rows += table.num_rows
            print(f"[{i}/{len(part_files)}] {part.name}: {table.num_rows:,} filas | total={total_rows:,}")
    finally:
        if writer is not None:
            writer.close()

    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "parts_dir": str(parts_dir),
        "output_path": str(output_path),
        "pattern": pattern,
        "num_parts": len(part_files),
        "total_rows": total_rows,
        "duration_seconds": round(time.time() - t0, 2),
    }
    meta_path = output_path.with_suffix(".concat.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Parquet final: {output_path}")
    print(f"Partes unidas: {len(part_files)}")
    print(f"Filas totales: {total_rows:,}")
    print(f"Metadata: {meta_path}")
    print(f"Duración: {(time.time() - t0)/60:.1f} min")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenar parquets parciales de features")
    parser.add_argument("--parts-dir", required=True, type=str, help="Carpeta con part_*.parquet")
    parser.add_argument("--output", required=True, type=str, help="Parquet final de salida")
    parser.add_argument("--pattern", default="part_*.parquet")
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    concat_parts(
        parts_dir=Path(args.parts_dir),
        output_path=Path(args.output),
        pattern=args.pattern,
        compression=args.compression,
        force=args.force,
    )


if __name__ == "__main__":
    main()
