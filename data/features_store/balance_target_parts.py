"""
balance_target_parts.py — Submuestreo target=0 por partes parquet.

Lee todos los part_*.parquet de una carpeta, conserva todos los registros target=1,
toma aleatoriamente ratio * n_target_1 registros target=0 por cada parte, y escribe
un único parquet balanceado.

Uso recomendado desde la raíz del repo:
  python data/features_store/balance_target_parts.py \
    --parts-dir data/features_store/features_train_w0_parts \
    --output data/features_store/features_train_w0_1to4.parquet \
    --ratio 4 \
    --force
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PARTS_DIR = SCRIPT_DIR / "features_train_w0_parts"
DEFAULT_OUTPUT = SCRIPT_DIR / "features_train_w0_1to4.parquet"


def balance_parts(
    parts_dir: Path,
    output_path: Path,
    ratio: int,
    target_col: str,
    pattern: str,
    seed: int,
    compression: str,
    force: bool,
) -> None:
    if ratio <= 0:
        raise ValueError("--ratio debe ser mayor que 0")

    if not parts_dir.exists():
        raise FileNotFoundError(f"No existe parts-dir: {parts_dir}")

    part_files = sorted(parts_dir.glob(pattern))
    if not part_files:
        raise FileNotFoundError(f"No se encontraron archivos {pattern} en {parts_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if not force:
            raise FileExistsError(f"Ya existe {output_path}. Usa --force para sobrescribir.")
        output_path.unlink()

    summary_path = output_path.with_suffix(".summary.json")
    if summary_path.exists() and force:
        summary_path.unlink()

    print(f"[Balance] Parts dir: {parts_dir}")
    print(f"[Balance] Partes encontradas: {len(part_files)}")
    print(f"[Balance] Ratio objetivo: target0 = {ratio} * target1")
    print(f"[Balance] Output: {output_path}")

    writer = None
    schema = None
    t0 = time.time()

    total_pos = 0
    total_neg_available = 0
    total_neg_selected = 0
    total_rows_written = 0
    per_part = []

    try:
        for idx, part_path in enumerate(part_files):
            print()
            print(f"[Balance] Leyendo parte {idx + 1}/{len(part_files)}: {part_path}")
            df = pd.read_parquet(part_path)

            if target_col not in df.columns:
                raise ValueError(f"La columna target no existe en {part_path}: {target_col}")

            df_pos = df[df[target_col] == 1]
            df_neg = df[df[target_col] == 0]

            n_pos = len(df_pos)
            n_neg = len(df_neg)
            n_neg_target = n_pos * ratio
            n_neg_sample = min(n_neg, n_neg_target)

            if n_pos == 0:
                selected = df.iloc[0:0].copy()
                print(f"[Balance] AVISO: {part_path.name} no tiene target=1; no se seleccionan filas.")
            else:
                neg_sample = df_neg.sample(n=n_neg_sample, random_state=seed + idx) if n_neg_sample > 0 else df_neg.iloc[0:0]
                selected = pd.concat([df_pos, neg_sample], ignore_index=True)
                selected = selected.sample(frac=1.0, random_state=seed + idx).reset_index(drop=True)

            rows = len(selected)
            total_pos += n_pos
            total_neg_available += n_neg
            total_neg_selected += n_neg_sample
            total_rows_written += rows

            print(
                f"[Balance] {part_path.name}: target1={n_pos:,} | "
                f"target0_disp={n_neg:,} | target0_sel={n_neg_sample:,} | filas_out={rows:,}"
            )

            per_part.append({
                "part_file": str(part_path),
                "target_1": int(n_pos),
                "target_0_available": int(n_neg),
                "target_0_selected": int(n_neg_sample),
                "rows_written": int(rows),
            })

            if rows > 0:
                table = pa.Table.from_pandas(selected, preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(output_path, schema, compression=compression)
                elif table.schema != schema:
                    raise ValueError(
                        f"Schema distinto en {part_path}. No se puede unir de forma segura."
                    )
                writer.write_table(table)

            del df, df_pos, df_neg, selected
            if "neg_sample" in locals():
                del neg_sample
            gc.collect()

    finally:
        if writer is not None:
            writer.close()

    if total_rows_written == 0:
        raise RuntimeError("No se escribió ninguna fila. Revisa si existen target=1 en las partes.")

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "parts_dir": str(parts_dir),
        "output_path": str(output_path),
        "pattern": pattern,
        "target_col": target_col,
        "ratio_target0_por_target1": ratio,
        "seed": seed,
        "n_parts": len(part_files),
        "total_target_1": int(total_pos),
        "total_target_0_available": int(total_neg_available),
        "total_target_0_selected": int(total_neg_selected),
        "total_rows_written": int(total_rows_written),
        "final_target0_target1_ratio": round(total_neg_selected / max(total_pos, 1), 4),
        "duration_sec": round(time.time() - t0, 2),
        "per_part": per_part,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print(f"[Balance] Guardado: {output_path}")
    print(f"[Balance] Summary: {summary_path}")
    print(
        f"[Balance] Total: target1={total_pos:,} | "
        f"target0_sel={total_neg_selected:,} | filas={total_rows_written:,} | "
        f"ratio={total_neg_selected / max(total_pos, 1):.2f}:1"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Balancea target 0/1 por partes parquet")
    parser.add_argument("--parts-dir", type=str, default=str(DEFAULT_PARTS_DIR),
                        help="Carpeta con part_*.parquet")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Parquet final balanceado")
    parser.add_argument("--ratio", type=int, default=4,
                        help="Cantidad de target=0 por cada target=1")
    parser.add_argument("--target-col", type=str, default="target",
                        help="Nombre de la columna target")
    parser.add_argument("--pattern", type=str, default="part_*.parquet",
                        help="Patrón de archivos parquet dentro de parts-dir")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para muestreo reproducible")
    parser.add_argument("--compression", type=str, default="snappy",
                        help="Compresión parquet")
    parser.add_argument("--force", action="store_true",
                        help="Sobrescribe output si existe")
    args = parser.parse_args()

    balance_parts(
        parts_dir=Path(args.parts_dir),
        output_path=Path(args.output),
        ratio=args.ratio,
        target_col=args.target_col,
        pattern=args.pattern,
        seed=args.seed,
        compression=args.compression,
        force=args.force,
    )


if __name__ == "__main__":
    main()
