"""EDA simple de intervalos de compra para churn a nivel cliente.

Lee solo CODIGO_FAMILIA y DIM_PERIODO, conserva los dos anios mas
recientes del historico y genera:
1. Una tabla estadistica general.
2. Una tabla por terciles excluyentes (P0-P33, P33-P66 y P66-P100).
3. Tres graficos simples de distribucion.

No construye el target ni entrena modelos.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def summarize_terciles(series: pd.Series, variable: str) -> list[dict[str, float | int | str]]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    p33 = float(values.quantile(1 / 3))
    p66 = float(values.quantile(2 / 3))

    grupos = [
        (f"P0-P33 (<= {p33:.2f})", values <= p33),
        (f"P33-P66 ({p33:.2f}, {p66:.2f}]", (values > p33) & (values <= p66)),
        (f"P66-P100 (> {p66:.2f})", values > p66),
    ]

    rows = []
    for tramo, mask in grupos:
        group = values.loc[mask]
        rows.append(
            {
                "variable": variable,
                "tramo": tramo,
                "clientes": int(group.size),
                "media": float(group.mean()),
                "desviacion_std": float(group.std(ddof=0)),
                "varianza": float(group.var(ddof=0)),
            }
        )
    return rows


def save_distribution_plot(
    series: pd.Series,
    title: str,
    xlabel: str,
    output_path: Path,
    bin_width: int = 5,
) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    upper = float(values.quantile(0.98))
    visible = values[values <= upper]

    max_value = max(bin_width, int(np.ceil(upper / bin_width) * bin_width))
    bins = np.arange(0, max_value + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(visible, bins=bins)
    ax.set_title(f"{title} (hasta P98 = {upper:.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Numero de clientes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


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

    variables_terciles = [
        "n_intervalos",
        "media_intervalos",
        "mediana_intervalos",
        "desviacion_intervalos",
    ]
    tabla_terciles = pd.DataFrame(
        [
            row
            for variable in variables_terciles
            for row in summarize_terciles(perfil[variable], variable)
        ]
    )

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
    terciles_path = output_dir / "eda_simple_churn_terciles.csv"
    log_path = output_dir / "eda_simple_churn.log"

    tabla.to_csv(csv_path, index=False)
    tabla_terciles.to_csv(terciles_path, index=False)

    graficos = [
        (
            "n_intervalos",
            "Distribucion del numero de intervalos por cliente",
            "Numero de intervalos",
            output_dir / "distribucion_n_intervalos.png",
        ),
        (
            "mediana_intervalos",
            "Distribucion de la mediana de intervalos por cliente",
            "Mediana de intervalos (dias)",
            output_dir / "distribucion_mediana_intervalos.png",
        ),
        (
            "desviacion_intervalos",
            "Distribucion de la desviacion de intervalos por cliente",
            "Desviacion de intervalos (dias)",
            output_dir / "distribucion_desviacion_intervalos.png",
        ),
    ]
    for variable, title, xlabel, path in graficos:
        save_distribution_plot(perfil[variable], title, xlabel, path)

    output_text = (
        "\n=== RESUMEN GENERAL ===\n"
        + resumen_general.to_string(index=False)
        + "\n\n=== TABLA ESTADISTICA ===\n"
        + tabla.to_string(index=False, float_format=lambda x: f"{x:,.2f}")
        + "\n\n=== TABLA POR TERCILES EXCLUYENTES ===\n"
        + tabla_terciles.to_string(index=False, float_format=lambda x: f"{x:,.2f}")
        + f"\n\nTiempo total: {time.perf_counter() - started:.2f} s\n"
    )
    print(output_text)
    log_path.write_text(output_text, encoding="utf-8")

    print(f"[EDA] Tabla general: {csv_path}")
    print(f"[EDA] Tabla por terciles: {terciles_path}")
    print(f"[EDA] Graficos: {output_dir}/distribucion_*.png")
    print(f"[EDA] Log: {log_path}")


if __name__ == "__main__":
    main()
