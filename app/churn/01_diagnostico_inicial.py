"""Diagnostico inicial del historico para churn a nivel CODIGO_FAMILIA.

Analiza solamente los dos anios finales del Parquet. No limpia transacciones, no
define todavia la poblacion elegible y no construye el target de churn.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "HISTORICO_300K_FULL.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "churn" / "diagnostico_inicial"
REQUIRED_COLUMNS = {"CODIGO_FAMILIA", "DIM_PERIODO", "DIM_FACTURA"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostico de los dos anios finales del historico de churn"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--memory-limit", default=None, help="Ejemplo: 220GB")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("diagnostico_churn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_dir / "diagnostico.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def as_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def query(
    connection: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
    label: str,
    statement: str,
) -> pd.DataFrame:
    started = time.perf_counter()
    logger.info("INICIO | %s", label)
    result = connection.execute(statement).fetchdf()
    logger.info("FIN    | %s | %.2f s", label, time.perf_counter() - started)
    return result


def execute(
    connection: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
    label: str,
    statement: str,
) -> None:
    started = time.perf_counter()
    logger.info("INICIO | %s", label)
    connection.execute(statement)
    logger.info("FIN    | %s | %.2f s", label, time.perf_counter() - started)


def create_plots(
    output_dir: Path,
    monthly: pd.DataFrame,
    gap_distribution: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    monthly = monthly.copy()
    monthly["MES"] = pd.to_datetime(monthly["MES"])
    fig, axis_families = plt.subplots(figsize=(12, 6))
    axis_families.plot(
        monthly["MES"], monthly["FAMILIAS"], marker="o", color="#35618f"
    )
    axis_families.set_xlabel("Mes")
    axis_families.set_ylabel("Familias activas", color="#35618f")
    axis_families.tick_params(axis="y", labelcolor="#35618f")

    axis_invoices = axis_families.twinx()
    axis_invoices.plot(
        monthly["MES"], monthly["FACTURAS"], marker="o", color="#b14f3d"
    )
    axis_invoices.set_ylabel("Facturas", color="#b14f3d")
    axis_invoices.tick_params(axis="y", labelcolor="#b14f3d")
    axis_families.set_title("Actividad mensual en los dos anios analizados")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "actividad_mensual.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 6))
    axis.bar(
        gap_distribution["INTERVALO_DIAS"],
        gap_distribution["CANTIDAD_INTERVALOS"],
        color="#4e7d5b",
    )
    axis.set_title("Intervalos entre compras consecutivas por familia")
    axis.set_xlabel("Dias entre compras")
    axis.set_ylabel("Cantidad de intervalos")
    axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / "intervalos_compra.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"No existe el Parquet: {input_path}")
    if args.threads < 1:
        raise ValueError("--threads debe ser mayor que cero")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    started = time.perf_counter()
    logger.info("Diagnostico de churn por CODIGO_FAMILIA")
    logger.info("Archivo: %s", input_path)
    logger.info("CPU solicitada: %s hilos", args.threads)

    connection = duckdb.connect(database=":memory:")
    connection.execute(f"SET threads = {args.threads}")
    connection.execute("SET preserve_insertion_order = false")
    if args.memory_limit:
        connection.execute(f"SET memory_limit = {sql_string(args.memory_limit)}")
    parquet_path = sql_string(str(input_path))

    schema = query(
        connection,
        logger,
        "lectura del esquema",
        f"DESCRIBE SELECT * FROM read_parquet({parquet_path})",
    )
    missing = REQUIRED_COLUMNS - set(schema["column_name"])
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    full_range = query(
        connection,
        logger,
        "rango completo",
        f"""
        SELECT
            COUNT(*)::BIGINT AS FILAS_ARCHIVO,
            MIN(CAST(DIM_PERIODO AS DATE)) AS FECHA_MINIMA_ARCHIVO,
            MAX(CAST(DIM_PERIODO AS DATE)) AS FECHA_MAXIMA_ARCHIVO
        FROM read_parquet({parquet_path})
        """,
    ).iloc[0]
    max_date = pd.Timestamp(full_range["FECHA_MAXIMA_ARCHIVO"])
    analysis_start = max_date - pd.DateOffset(years=2) + pd.Timedelta(days=1)

    logger.info(
        "Ventana seleccionada: %s a %s",
        analysis_start.date(),
        max_date.date(),
    )
    execute(
        connection,
        logger,
        "vista de los dos anios finales",
        f"""
        CREATE TEMP VIEW tx AS
        SELECT
            CODIGO_FAMILIA,
            CAST(DIM_PERIODO AS DATE) AS FECHA_COMPRA,
            DIM_FACTURA
        FROM read_parquet({parquet_path})
        WHERE CAST(DIM_PERIODO AS DATE)
              BETWEEN DATE {sql_string(str(analysis_start.date()))}
                  AND DATE {sql_string(str(max_date.date()))}
        """,
    )

    overview = query(
        connection,
        logger,
        "resumen de la ventana",
        """
        SELECT
            COUNT(*)::BIGINT AS FILAS,
            COUNT(DISTINCT CODIGO_FAMILIA)::BIGINT AS FAMILIAS,
            COUNT(DISTINCT DIM_FACTURA)::BIGINT AS FACTURAS,
            COUNT(DISTINCT FECHA_COMPRA)::BIGINT AS DIAS_CON_VENTAS,
            COUNT(*) FILTER (WHERE CODIGO_FAMILIA IS NULL)::BIGINT AS FAMILIA_NULA,
            COUNT(*) FILTER (WHERE FECHA_COMPRA IS NULL)::BIGINT AS FECHA_NULA,
            COUNT(*) FILTER (WHERE DIM_FACTURA IS NULL)::BIGINT AS FACTURA_NULA
        FROM tx
        """,
    )

    monthly = query(
        connection,
        logger,
        "actividad mensual",
        """
        SELECT
            CAST(date_trunc('month', FECHA_COMPRA) AS DATE) AS MES,
            COUNT(*)::BIGINT AS FILAS,
            COUNT(DISTINCT CODIGO_FAMILIA)::BIGINT AS FAMILIAS,
            COUNT(DISTINCT DIM_FACTURA)::BIGINT AS FACTURAS
        FROM tx
        WHERE CODIGO_FAMILIA IS NOT NULL AND FECHA_COMPRA IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """,
    )
    monthly.to_csv(output_dir / "actividad_mensual.csv", index=False)

    execute(
        connection,
        logger,
        "fechas de compra por familia",
        """
        CREATE TEMP TABLE family_days AS
        SELECT
            CODIGO_FAMILIA,
            FECHA_COMPRA,
            COUNT(DISTINCT DIM_FACTURA)::BIGINT AS FACTURAS_DIA
        FROM tx
        WHERE CODIGO_FAMILIA IS NOT NULL AND FECHA_COMPRA IS NOT NULL
        GROUP BY 1, 2
        """,
    )
    execute(
        connection,
        logger,
        "perfil por familia",
        """
        CREATE TEMP TABLE family_profile AS
        SELECT
            CODIGO_FAMILIA,
            COUNT(*)::BIGINT AS DIAS_COMPRA,
            SUM(FACTURAS_DIA)::BIGINT AS FACTURAS,
            MIN(FECHA_COMPRA) AS PRIMERA_COMPRA,
            MAX(FECHA_COMPRA) AS ULTIMA_COMPRA
        FROM family_days
        GROUP BY 1
        """,
    )
    execute(
        connection,
        logger,
        "intervalos entre compras",
        """
        CREATE TEMP TABLE family_gaps AS
        SELECT
            CODIGO_FAMILIA,
            date_diff('day', FECHA_ANTERIOR, FECHA_COMPRA)::BIGINT AS INTERVALO_DIAS
        FROM (
            SELECT
                CODIGO_FAMILIA,
                FECHA_COMPRA,
                LAG(FECHA_COMPRA) OVER (
                    PARTITION BY CODIGO_FAMILIA ORDER BY FECHA_COMPRA
                ) AS FECHA_ANTERIOR
            FROM family_days
        ) ordered_purchases
        WHERE FECHA_ANTERIOR IS NOT NULL
        """,
    )
    execute(
        connection,
        logger,
        "ciclos medios por familia",
        """
        CREATE TEMP TABLE family_cycles AS
        SELECT
            CODIGO_FAMILIA,
            COUNT(*)::BIGINT AS INTERVALOS,
            AVG(INTERVALO_DIAS)::DOUBLE AS MEDIA_CICLO,
            STDDEV_POP(INTERVALO_DIAS)::DOUBLE AS DESVIACION_CICLO
        FROM family_gaps
        GROUP BY 1
        """,
    )

    family_summary = query(
        connection,
        logger,
        "estadisticos de actividad por familia",
        """
        SELECT
            COUNT(*)::BIGINT AS FAMILIAS,
            AVG(DIAS_COMPRA)::DOUBLE AS MEDIA_DIAS_COMPRA,
            STDDEV_POP(DIAS_COMPRA)::DOUBLE AS DESVIACION_DIAS_COMPRA,
            approx_quantile(DIAS_COMPRA, 0.25) AS P25_DIAS_COMPRA,
            approx_quantile(DIAS_COMPRA, 0.50) AS P50_DIAS_COMPRA,
            approx_quantile(DIAS_COMPRA, 0.75) AS P75_DIAS_COMPRA,
            approx_quantile(DIAS_COMPRA, 0.90) AS P90_DIAS_COMPRA,
            AVG(FACTURAS)::DOUBLE AS MEDIA_FACTURAS,
            STDDEV_POP(FACTURAS)::DOUBLE AS DESVIACION_FACTURAS
        FROM family_profile
        """,
    )
    gap_summary = query(
        connection,
        logger,
        "estadisticos de intervalos",
        """
        SELECT
            COUNT(*)::BIGINT AS INTERVALOS,
            AVG(INTERVALO_DIAS)::DOUBLE AS MEDIA,
            STDDEV_POP(INTERVALO_DIAS)::DOUBLE AS DESVIACION,
            approx_quantile(INTERVALO_DIAS, 0.50) AS P50,
            approx_quantile(INTERVALO_DIAS, 0.75) AS P75,
            approx_quantile(INTERVALO_DIAS, 0.90) AS P90,
            approx_quantile(INTERVALO_DIAS, 0.95) AS P95,
            approx_quantile(INTERVALO_DIAS, 0.99) AS P99
        FROM family_gaps
        """,
    )
    cycle_summary = query(
        connection,
        logger,
        "distribucion del ciclo medio por familia",
        """
        SELECT
            COUNT(*)::BIGINT AS FAMILIAS_CON_CICLO,
            AVG(MEDIA_CICLO)::DOUBLE AS MEDIA,
            STDDEV_POP(MEDIA_CICLO)::DOUBLE AS DESVIACION,
            approx_quantile(MEDIA_CICLO, 0.50) AS P50,
            approx_quantile(MEDIA_CICLO, 0.75) AS P75,
            approx_quantile(MEDIA_CICLO, 0.90) AS P90,
            approx_quantile(MEDIA_CICLO, 0.95) AS P95
        FROM family_cycles
        """,
    )
    eligibility = query(
        connection,
        logger,
        "cobertura por minimo de fechas de compra",
        """
        SELECT
            LIMITE AS MINIMO_DIAS_COMPRA,
            COUNT(*) FILTER (WHERE DIAS_COMPRA >= LIMITE)::BIGINT AS FAMILIAS,
            ROUND(
                100.0 * COUNT(*) FILTER (WHERE DIAS_COMPRA >= LIMITE) / COUNT(*),
                4
            ) AS PORCENTAJE
        FROM family_profile
        CROSS JOIN (VALUES (2), (3), (4), (5), (6), (10)) limits(LIMITE)
        GROUP BY LIMITE
        ORDER BY LIMITE
        """,
    )
    gap_distribution = query(
        connection,
        logger,
        "distribucion agrupada de intervalos",
        """
        SELECT
            CASE
                WHEN INTERVALO_DIAS <= 7 THEN '1-7'
                WHEN INTERVALO_DIAS <= 14 THEN '8-14'
                WHEN INTERVALO_DIAS <= 30 THEN '15-30'
                WHEN INTERVALO_DIAS <= 60 THEN '31-60'
                WHEN INTERVALO_DIAS <= 90 THEN '61-90'
                WHEN INTERVALO_DIAS <= 180 THEN '91-180'
                WHEN INTERVALO_DIAS <= 365 THEN '181-365'
                ELSE '366+'
            END AS INTERVALO_DIAS,
            COUNT(*)::BIGINT AS CANTIDAD_INTERVALOS,
            MIN(INTERVALO_DIAS)::BIGINT AS ORDEN
        FROM family_gaps
        GROUP BY 1
        ORDER BY ORDEN
        """,
    )
    gap_distribution.drop(columns="ORDEN").to_csv(
        output_dir / "intervalos_compra.csv", index=False
    )
    create_plots(output_dir, monthly, gap_distribution)

    overview_row = as_records(overview)[0]
    gap_row = as_records(gap_summary)[0]
    logger.info(
        "Resultado: %s filas | %s familias | %s facturas",
        f"{overview_row['FILAS']:,}",
        f"{overview_row['FAMILIAS']:,}",
        f"{overview_row['FACTURAS']:,}",
    )
    logger.info(
        "Intervalos preliminares: P50=%s | P90=%s | P95=%s | P99=%s dias",
        gap_row["P50"], gap_row["P90"], gap_row["P95"], gap_row["P99"],
    )

    summary = {
        "archivo": {
            "ruta": input_path,
            "tamano_mb": round(input_path.stat().st_size / 1024**2, 2),
            "filas_completas": int(full_range["FILAS_ARCHIVO"]),
            "fecha_minima": full_range["FECHA_MINIMA_ARCHIVO"],
            "fecha_maxima": full_range["FECHA_MAXIMA_ARCHIVO"],
            "columnas": schema[["column_name", "column_type"]].to_dict("records"),
        },
        "ventana_analizada": {
            "inicio": analysis_start.date(),
            "fin": max_date.date(),
            "unidad": "CODIGO_FAMILIA",
        },
        "resumen": overview_row,
        "actividad_por_familia": as_records(family_summary)[0],
        "intervalos_globales": gap_row,
        "ciclo_medio_por_familia": as_records(cycle_summary)[0],
        "elegibilidad_exploratoria": as_records(eligibility),
        "duracion_segundos": round(time.perf_counter() - started, 2),
    }
    with open(output_dir / "resumen.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=json_default)

    connection.close()
    logger.info("Diagnostico completado en %.2f segundos", time.perf_counter() - started)
    logger.info("Se generaron 6 artefactos en %s", output_dir)


if __name__ == "__main__":
    main()

