"""EDA simple de intervalos de compra para churn a nivel cliente.

Lee solo CODIGO_FAMILIA y DIM_PERIODO, conserva los dos anios mas
recientes del historico y genera una tabla estadistica sencilla.
No construye el target ni entrena modelos.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "HISTORICO_300K_FULL.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "churn" / "logs"
REQUIRED_COLUMNS = ["CODIGO_FAMILIA", "DIM_PERIODO"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA simple de churn con pandas")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def summarize(series: pd.Series, variable: str) -> dict[str, float | int | str]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "variable": variable,
        "n": int(values.size),
        "media": float(values.mean()),
        "mediana": float(values.median()),
        "desviacion_std": float(values.std(ddof=0)),
        "varianza": float(values.var(ddof=0)),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p90": float(values.quantile(0.90)),
    }


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print(f"[EDA] Leyendo: {input_path}")
    df = pd.read_parquet(input_path, columns=REQUIRED_COLUMNS)
    df["DIM_PERIODO"] = pd.to_datetime(df["DIM_PERIODO"], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS)

    fecha_max = df["DIM_PERIODO"].max()
    fecha_inicio = fecha_max - pd.DateOffset(years=2) + pd.Timedelta(days=1)
    df = df[df["DIM_PERIODO"].between(fecha_inicio, fecha_max)].copy()

    compras = (
        df[REQUIRED_COLUMNS]
        .drop_duplicates()
        .sort_values(["CODIGO_FAMILIA", "DIM_PERIODO"])
        .reset_index(drop=True)
    )

    compras["intervalo_dias"] = (
        compras.groupby("CODIGO_FAMILIA", sort=False)["DIM_PERIODO"]
        .diff()
        .dt.days
    )

    perfil = (
        compras.groupby("CODIGO_FAMILIA", sort=False)
        .agg(
            n_fechas_compra=("DIM_PERIODO", "size"),
            primera_compra=("DIM_PERIODO", "min"),
            ultima_compra=("DIM_PERIODO", "max"),
            media_intervalos=("intervalo_dias", "mean"),
            mediana_intervalos=("intervalo_dias", "median"),
            desviacion_intervalos=("intervalo_dias", lambda x: x.std(ddof=0)),
            varianza_intervalos=("intervalo_dias", lambda x: x.var(ddof=0)),
        )
        .reset_index()
    )

    perfil["n_intervalos"] = perfil["n_fechas_compra"] - 1
    perfil["duracion_relacion_dias"] = (
        perfil["ultima_compra"] - perfil["primera_compra"]
    ).dt.days

    variables = [
        "n_fechas_compra",
        "n_intervalos",
        "duracion_relacion_dias",
        "media_intervalos",
        "mediana_intervalos",
        "desviacion_intervalos",
        "varianza_intervalos",
    ]
    tabla = pd.DataFrame([summarize(perfil[col], col) for col in variables])

    resumen_general = pd.DataFrame(
        {
            "metrica": [
                "fecha_inicio",
                "fecha_fin",
                "filas_transaccionales",
                "clientes",
                "fechas_cliente_unicas",
                "clientes_con_intervalos",
            ],
            "valor": [
                str(fecha_inicio.date()),
                str(fecha_max.date()),
                int(len(df)),
                int(perfil["CODIGO_FAMILIA"].nunique()),
                int(len(compras)),
                int((perfil["n_intervalos"] >= 1).sum()),
            ],
        }
    )

    csv_path = output_dir / "eda_simple_churn.csv"
    log_path = output_dir / "eda_simple_churn.log"
    tabla.to_csv(csv_path, index=False)

    output_text = (
        "\n=== RESUMEN GENERAL ===\n"
        + resumen_general.to_string(index=False)
        + "\n\n=== TABLA ESTADISTICA ===\n"
        + tabla.to_string(index=False, float_format=lambda x: f"{x:,.2f}")
        + f"\n\nTiempo total: {time.perf_counter() - started:.2f} s\n"
    )
    print(output_text)
    log_path.write_text(output_text, encoding="utf-8")

    print(f"[EDA] Tabla guardada en: {csv_path}")
    print(f"[EDA] Log guardado en: {log_path}")


if __name__ == "__main__":
    main()
