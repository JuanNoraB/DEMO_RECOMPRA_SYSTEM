"""Diagnóstico inicial del histórico transaccional para el experimento de churn.

Esta etapa no limpia datos, no construye el horizonte H y no genera el target. Su
objetivo es producir evidencia para tomar esas decisiones en la siguiente iteración.

Ejemplo de ejecución en servidor:

    python app/churn/01_diagnostico_inicial.py \
      --input /data/HISTORICO_300K_FULL.parquet \
      --output-dir /data/churn/diagnostico_inicial \
      --threads 128 \
      --memory-limit 220GB

DuckDB procesa el Parquet en paralelo. Pandas solo recibe resultados agregados.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "churn" / "diagnostico_inicial"

DEFAULT_COLUMNS = {
    "customer": "CODIGO_FAMILIA",
    "identity": "IDENTIFICACION",
    "date": "DIM_PERIODO",
    "invoice": "DIM_FACTURA",
    "item": "COD_ITEM",
    "subcategory": "COD_SUBCATEGORIA",
    "subcategory_name": "NOMBRE_SUBCATEGORIA",
    "category": "COD_CATEGORIA",
    "category_name": "NOMBRE_CATEGORIA",
    "store": "COD_LOCAL",
    "store_name": "NOMBRE_LOCAL",
    "quantity": "CANTIDAD_SUELTA",
    "price": "PVP",
    "net_sales": "VENTA_NETA",
    "margin": "MARGEN_FRONT",
    "discount": "DESCUENTO",
    "type": "TIPO",
}

QUANTILES = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnóstico reproducible de un histórico transaccional Parquet"
    )
    parser.add_argument("--input", required=True, type=Path, help="Archivo Parquet de entrada")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta raíz donde se creará un directorio por ejecución",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nombre opcional de la ejecución; por defecto usa fecha y hora",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Hilos de DuckDB; en el servidor puede usarse 128",
    )
    parser.add_argument(
        "--memory-limit",
        default=None,
        help="Límite de memoria de DuckDB, por ejemplo 220GB; por defecto DuckDB decide",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=None,
        help="Directorio para spill de DuckDB si una consulta excede la RAM",
    )
    parser.add_argument("--customer-col", default=DEFAULT_COLUMNS["customer"])
    parser.add_argument("--date-col", default=DEFAULT_COLUMNS["date"])
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("churn_diagnostico")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "diagnostico.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return str(value)


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def timed_df(
    con: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
    label: str,
    sql: str,
) -> pd.DataFrame:
    logger.info("INICIO | %s", label)
    started = time.perf_counter()
    result = con.execute(sql).fetchdf()
    logger.info("FIN    | %s | %.2f s | %s filas", label, time.perf_counter() - started, f"{len(result):,}")
    return result


def timed_sql(
    con: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
    label: str,
    sql: str,
) -> None:
    logger.info("INICIO | %s", label)
    started = time.perf_counter()
    con.execute(sql)
    logger.info("FIN    | %s | %.2f s", label, time.perf_counter() - started)


def configure_duckdb(args: argparse.Namespace, run_dir: Path, logger: logging.Logger) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET threads = {args.threads}")
    con.execute("SET preserve_insertion_order = false")

    if args.memory_limit:
        con.execute(f"SET memory_limit = {quote_string(args.memory_limit)}")

    temp_dir = (args.temp_dir or (run_dir / "duckdb_tmp")).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = {quote_string(str(temp_dir))}")

    logger.info("DuckDB %s | threads=%s | memory_limit=%s", duckdb.__version__, args.threads, args.memory_limit or "auto")
    logger.info("DuckDB temp_directory=%s", temp_dir)
    return con


def parquet_metadata(input_path: Path, run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    parquet_file = pq.ParquetFile(input_path)
    metadata = parquet_file.metadata
    arrow_schema = parquet_file.schema_arrow

    schema_df = pd.DataFrame(
        {
            "column": arrow_schema.names,
            "arrow_type": [str(field.type) for field in arrow_schema],
            "nullable": [field.nullable for field in arrow_schema],
        }
    )
    save_csv(schema_df, run_dir / "schema.csv")

    row_group_rows = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]
    result = {
        "file_size_bytes": input_path.stat().st_size,
        "file_size_mb": round(input_path.stat().st_size / 1024**2, 2),
        "metadata_rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "columns": metadata.num_columns,
        "created_by": metadata.created_by,
        "row_group_rows": row_group_rows,
    }
    return result, schema_df


def validate_columns(schema_df: pd.DataFrame, customer_col: str, date_col: str) -> None:
    available = set(schema_df["column"])
    required = {
        customer_col,
        date_col,
        DEFAULT_COLUMNS["invoice"],
        DEFAULT_COLUMNS["item"],
        DEFAULT_COLUMNS["subcategory"],
        DEFAULT_COLUMNS["category"],
        DEFAULT_COLUMNS["net_sales"],
        DEFAULT_COLUMNS["quantity"],
    }
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}. Disponibles: {sorted(available)}")


def build_quantile_profile(
    con: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
    table: str,
    metrics: list[str],
    output_path: Path,
    label: str,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    quantile_sql = ", ".join(str(q) for q in QUANTILES)
    for metric in metrics:
        col = quote_identifier(metric)
        row = timed_df(
            con,
            logger,
            f"{label}: {metric}",
            f"""
            SELECT
                COUNT({col})::BIGINT AS n,
                MIN({col}) AS exact_min,
                MAX({col}) AS exact_max,
                AVG({col})::DOUBLE AS mean,
                STDDEV_POP({col})::DOUBLE AS std,
                approx_quantile({col}, [{quantile_sql}]) AS quantiles
            FROM {table}
            WHERE {col} IS NOT NULL
            """,
        ).iloc[0]
        values = row["quantiles"]
        record = {"metric": metric, "n": int(row["n"]), "mean": row["mean"], "std": row["std"]}
        for quantile, value in zip(QUANTILES, values, strict=True):
            record[f"p{int(quantile * 100):02d}"] = value
        record["p00"] = row["exact_min"]
        record["p100"] = row["exact_max"]
        output.append(record)

    result = pd.DataFrame(output)
    save_csv(result, output_path)
    return result


def profile_nulls(
    con: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
    columns: list[str],
    total_rows: int,
) -> pd.DataFrame:
    expressions = [
        f"COUNT(*) FILTER (WHERE {quote_identifier(column)} IS NULL)::BIGINT AS {quote_identifier(column)}"
        for column in columns
    ]
    row = timed_df(con, logger, "perfil de nulos", "SELECT " + ", ".join(expressions) + " FROM tx").iloc[0]
    result = pd.DataFrame(
        {
            "column": columns,
            "null_count": [int(row[column]) for column in columns],
        }
    )
    result["null_pct"] = 100.0 * result["null_count"] / max(total_rows, 1)
    return result.sort_values(["null_pct", "column"], ascending=[False, True]).reset_index(drop=True)


def profile_numeric(
    con: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
    available_columns: set[str],
) -> pd.DataFrame:
    numeric_columns = [
        DEFAULT_COLUMNS["quantity"],
        DEFAULT_COLUMNS["price"],
        DEFAULT_COLUMNS["net_sales"],
        DEFAULT_COLUMNS["margin"],
        DEFAULT_COLUMNS["discount"],
    ]
    numeric_columns = [column for column in numeric_columns if column in available_columns]
    quantile_sql = ", ".join(str(q) for q in [0.01, 0.05, 0.50, 0.95, 0.99])
    expressions: list[str] = []
    for column in numeric_columns:
        col = quote_identifier(column)
        prefix = column.lower()
        expressions.extend(
            [
                f"COUNT(*) FILTER (WHERE {col} IS NULL)::BIGINT AS {prefix}_nulls",
                f"COUNT(*) FILTER (WHERE {col} = 0)::BIGINT AS {prefix}_zeros",
                f"COUNT(*) FILTER (WHERE {col} < 0)::BIGINT AS {prefix}_negatives",
                f"MIN({col})::DOUBLE AS {prefix}_min",
                f"MAX({col})::DOUBLE AS {prefix}_max",
                f"AVG({col})::DOUBLE AS {prefix}_mean",
                f"approx_quantile({col}, [{quantile_sql}]) AS {prefix}_quantiles",
            ]
        )
    row = timed_df(con, logger, "perfil numérico", "SELECT " + ", ".join(expressions) + " FROM tx").iloc[0]
    output = []
    for column in numeric_columns:
        prefix = column.lower()
        quantiles = row[f"{prefix}_quantiles"]
        output.append(
            {
                "column": column,
                "null_count": int(row[f"{prefix}_nulls"]),
                "zero_count": int(row[f"{prefix}_zeros"]),
                "negative_count": int(row[f"{prefix}_negatives"]),
                "min": row[f"{prefix}_min"],
                "p01": quantiles[0],
                "p05": quantiles[1],
                "p50": quantiles[2],
                "p95": quantiles[3],
                "p99": quantiles[4],
                "max": row[f"{prefix}_max"],
                "mean": row[f"{prefix}_mean"],
            }
        )
    return pd.DataFrame(output)


def make_plots(
    run_dir: Path,
    monthly: pd.DataFrame,
    purchase_day_bins: pd.DataFrame,
    gap_bins: pd.DataFrame,
    top_categories: pd.DataFrame,
    null_profile: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.plot(monthly["month"], monthly["rows"], color="#35618f", marker="o", label="Filas")
    ax1.set_ylabel("Filas transaccionales", color="#35618f")
    ax1.tick_params(axis="y", labelcolor="#35618f")
    ax2 = ax1.twinx()
    ax2.plot(monthly["month"], monthly["customers"], color="#b14f3d", marker="o", label="Clientes")
    ax2.set_ylabel("Clientes únicos", color="#b14f3d")
    ax2.tick_params(axis="y", labelcolor="#b14f3d")
    ax1.set_title("Actividad mensual del histórico")
    ax1.set_xlabel("Mes")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(run_dir / "fig_01_actividad_mensual.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(monthly["month"], monthly["rows_per_invoice"], color="#6c4d7d", marker="o")
    ax.set_title("Líneas transaccionales por factura a lo largo del tiempo")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Filas / facturas")
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(run_dir / "fig_01b_lineas_por_factura.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(purchase_day_bins["bin"], purchase_day_bins["customers"], color="#35618f")
    ax.set_title("Clientes por número de fechas distintas de compra")
    ax.set_xlabel("Fechas distintas de compra")
    ax.set_ylabel("Clientes")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(run_dir / "fig_02_compras_por_cliente.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(gap_bins["bin"], gap_bins["intervals"], color="#4e7d5b")
    ax.set_title("Distribución preliminar de intervalos completos")
    ax.set_xlabel("Días entre fechas consecutivas de compra")
    ax.set_ylabel("Intervalos")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(run_dir / "fig_03_intervalos_compra.png", dpi=160)
    plt.close(fig)

    top = top_categories.head(15).sort_values("net_sales", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top["category_label"], top["net_sales"], color="#8a6742")
    ax.set_title("Principales categorías por venta neta")
    ax.set_xlabel("Venta neta")
    fig.tight_layout()
    fig.savefig(run_dir / "fig_04_top_categorias.png", dpi=160)
    plt.close(fig)

    missing = null_profile.sort_values("null_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(missing["column"], missing["null_pct"], color="#9a4f55")
    ax.set_title("Porcentaje de valores nulos por columna")
    ax.set_xlabel("Nulos (%)")
    fig.tight_layout()
    fig.savefig(run_dir / "fig_05_nulos.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"No existe el Parquet: {input_path}")
    if args.threads <= 0:
        raise ValueError("--threads debe ser mayor que cero")

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output_dir.expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logging(run_dir)
    started = time.perf_counter()

    logger.info("=" * 78)
    logger.info("DIAGNÓSTICO INICIAL DE CHURN")
    logger.info("Input: %s", input_path)
    logger.info("Output: %s", run_dir)
    logger.info("Esta etapa NO construye H ni target.")
    logger.info("=" * 78)

    hardware = {
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "threads_requested": args.threads,
        "memory_limit": args.memory_limit or "duckdb_auto",
        "gpu_used": False,
    }
    logger.info(
        "Recursos: CPU lógica=%s | CPU física=%s | RAM=%.2f GB | GPU=no aplica",
        hardware["logical_cpus"],
        hardware["physical_cpus"],
        hardware["ram_total_gb"],
    )

    parquet_info, schema_df = parquet_metadata(input_path, run_dir)
    validate_columns(schema_df, args.customer_col, args.date_col)
    available_columns = set(schema_df["column"])
    logger.info(
        "Parquet: %s filas | %s columnas | %s row groups | %.2f MB",
        f"{parquet_info['metadata_rows']:,}",
        parquet_info["columns"],
        parquet_info["row_groups"],
        parquet_info["file_size_mb"],
    )
    logger.info("Columnas: %s", ", ".join(schema_df["column"]))

    con = configure_duckdb(args, run_dir, logger)
    parquet_sql = quote_string(str(input_path))
    con.execute(f"CREATE VIEW tx AS SELECT * FROM read_parquet({parquet_sql})")

    customer = quote_identifier(args.customer_col)
    date_col = quote_identifier(args.date_col)
    invoice = quote_identifier(DEFAULT_COLUMNS["invoice"])
    item = quote_identifier(DEFAULT_COLUMNS["item"])
    subcategory = quote_identifier(DEFAULT_COLUMNS["subcategory"])
    category = quote_identifier(DEFAULT_COLUMNS["category"])
    store = quote_identifier(DEFAULT_COLUMNS["store"])
    net_sales = quote_identifier(DEFAULT_COLUMNS["net_sales"])
    quantity = quote_identifier(DEFAULT_COLUMNS["quantity"])

    overview = timed_df(
        con,
        logger,
        "resumen general",
        f"""
        SELECT
            COUNT(*)::BIGINT AS rows,
            MIN({date_col}) AS min_timestamp,
            MAX({date_col}) AS max_timestamp,
            COUNT(DISTINCT CAST({date_col} AS DATE))::BIGINT AS distinct_dates,
            COUNT(DISTINCT {customer})::BIGINT AS customers,
            COUNT(DISTINCT {invoice})::BIGINT AS invoices,
            COUNT(DISTINCT {item})::BIGINT AS items,
            COUNT(DISTINCT {subcategory})::BIGINT AS subcategories,
            COUNT(DISTINCT {category})::BIGINT AS categories,
            COUNT(DISTINCT {store})::BIGINT AS stores,
            SUM(COALESCE({net_sales}, 0))::DOUBLE AS net_sales,
            SUM(COALESCE({quantity}, 0))::DOUBLE AS units
        FROM tx
        """,
    )
    overview_record = records(overview)[0]
    logger.info(
        "Rango=%s a %s | clientes=%s | facturas=%s",
        overview_record["min_timestamp"],
        overview_record["max_timestamp"],
        f"{overview_record['customers']:,}",
        f"{overview_record['invoices']:,}",
    )

    null_profile = profile_nulls(con, logger, list(schema_df["column"]), int(overview_record["rows"]))
    save_csv(null_profile, run_dir / "null_profile.csv")

    numeric_profile = profile_numeric(con, logger, available_columns)
    save_csv(numeric_profile, run_dir / "numeric_profile.csv")

    identity_profile: dict[str, Any] | None = None
    identity_col = DEFAULT_COLUMNS["identity"]
    if identity_col in available_columns and DEFAULT_COLUMNS["customer"] in available_columns:
        identity = quote_identifier(identity_col)
        family = quote_identifier(DEFAULT_COLUMNS["customer"])
        identity_df = timed_df(
            con,
            logger,
            "consistencia IDENTIFICACION vs CODIGO_FAMILIA",
            f"""
            WITH pairs AS (
                SELECT DISTINCT {identity} AS identity_id, {family} AS family_id
                FROM tx
                WHERE {identity} IS NOT NULL AND {family} IS NOT NULL
            ), family_counts AS (
                SELECT family_id, COUNT(*) AS n FROM pairs GROUP BY family_id
            ), identity_counts AS (
                SELECT identity_id, COUNT(*) AS n FROM pairs GROUP BY identity_id
            )
            SELECT
                (SELECT COUNT(DISTINCT identity_id) FROM pairs)::BIGINT AS distinct_identifications,
                (SELECT COUNT(DISTINCT family_id) FROM pairs)::BIGINT AS distinct_families,
                (SELECT COUNT(*) FROM pairs)::BIGINT AS distinct_pairs,
                (SELECT COUNT(*) FROM pairs WHERE identity_id != family_id)::BIGINT AS unequal_pairs,
                (SELECT COUNT(*) FROM family_counts WHERE n > 1)::BIGINT AS families_with_multiple_ids,
                (SELECT COUNT(*) FROM identity_counts WHERE n > 1)::BIGINT AS ids_with_multiple_families
            """,
        )
        identity_profile = records(identity_df)[0]
        save_csv(identity_df, run_dir / "identity_consistency.csv")
        logger.info(
            "Claves cliente: IDENTIFICACION=%s | CODIGO_FAMILIA=%s | pares=%s | familias con >1 identificación=%s",
            f"{identity_profile['distinct_identifications']:,}",
            f"{identity_profile['distinct_families']:,}",
            f"{identity_profile['distinct_pairs']:,}",
            f"{identity_profile['families_with_multiple_ids']:,}",
        )

    monthly = timed_df(
        con,
        logger,
        "actividad mensual",
        f"""
        SELECT
            CAST(date_trunc('month', {date_col}) AS DATE) AS month,
            COUNT(*)::BIGINT AS rows,
            COUNT(DISTINCT {customer})::BIGINT AS customers,
            COUNT(DISTINCT {invoice})::BIGINT AS invoices,
            SUM(COALESCE({net_sales}, 0))::DOUBLE AS net_sales,
            SUM(COALESCE({quantity}, 0))::DOUBLE AS units,
            COUNT(*)::DOUBLE / NULLIF(COUNT(DISTINCT {invoice}), 0) AS rows_per_invoice,
            SUM(COALESCE({net_sales}, 0))::DOUBLE / NULLIF(COUNT(DISTINCT {invoice}), 0) AS net_sales_per_invoice,
            SUM(COALESCE({quantity}, 0))::DOUBLE / NULLIF(COUNT(DISTINCT {invoice}), 0) AS units_per_invoice
        FROM tx
        WHERE {date_col} IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """,
    )
    save_csv(monthly, run_dir / "monthly_activity.csv")
    monthly_changes = monthly.copy()
    for metric in ["rows", "customers", "invoices", "rows_per_invoice"]:
        monthly_changes[f"{metric}_mom_pct"] = 100.0 * monthly_changes[metric].pct_change()
    monthly_flags = monthly_changes[
        (monthly_changes["rows_mom_pct"].abs() >= 40.0)
        | (monthly_changes["rows_per_invoice_mom_pct"].abs() >= 40.0)
    ].copy()
    save_csv(monthly_flags, run_dir / "monthly_discontinuity_flags.csv")
    for row in monthly_flags.itertuples(index=False):
        logger.warning(
            "Discontinuidad mensual %s | filas=%+.1f%% | filas/factura=%+.1f%% | clientes=%+.1f%%",
            row.month,
            row.rows_mom_pct,
            row.rows_per_invoice_mom_pct,
            row.customers_mom_pct,
        )

    timed_sql(
        con,
        logger,
        "tabla temporal cliente-fecha",
        f"""
        CREATE TEMP TABLE customer_days AS
        SELECT
            {customer} AS customer_id,
            CAST({date_col} AS DATE) AS purchase_date,
            COUNT(*)::BIGINT AS rows_on_day,
            COUNT(DISTINCT {invoice})::BIGINT AS invoices_on_day,
            SUM(COALESCE({net_sales}, 0))::DOUBLE AS net_sales_on_day,
            SUM(COALESCE({quantity}, 0))::DOUBLE AS units_on_day
        FROM tx
        WHERE {customer} IS NOT NULL AND {date_col} IS NOT NULL
        GROUP BY 1, 2
        """,
    )

    timed_sql(
        con,
        logger,
        "tabla temporal de estadísticas por cliente",
        """
        CREATE TEMP TABLE customer_stats AS
        SELECT
            customer_id,
            COUNT(*)::BIGINT AS distinct_purchase_days,
            SUM(invoices_on_day)::BIGINT AS invoices,
            SUM(rows_on_day)::BIGINT AS transaction_rows,
            MIN(purchase_date) AS first_purchase_date,
            MAX(purchase_date) AS last_purchase_date,
            date_diff('day', MIN(purchase_date), MAX(purchase_date))::BIGINT AS activity_span_days,
            SUM(net_sales_on_day)::DOUBLE AS net_sales,
            SUM(units_on_day)::DOUBLE AS units
        FROM customer_days
        GROUP BY customer_id
        """,
    )

    timed_sql(
        con,
        logger,
        "tabla temporal de intervalos completos",
        """
        CREATE TEMP TABLE customer_gaps AS
        SELECT
            customer_id,
            date_diff('day', previous_date, purchase_date)::BIGINT AS gap_days
        FROM (
            SELECT
                customer_id,
                purchase_date,
                LAG(purchase_date) OVER (
                    PARTITION BY customer_id ORDER BY purchase_date
                ) AS previous_date
            FROM customer_days
        ) ordered_days
        WHERE previous_date IS NOT NULL AND purchase_date > previous_date
        """,
    )

    customer_quantiles = build_quantile_profile(
        con,
        logger,
        "customer_stats",
        ["distinct_purchase_days", "invoices", "transaction_rows", "activity_span_days", "net_sales", "units"],
        run_dir / "customer_profile_quantiles.csv",
        "perfil de clientes",
    )
    gap_quantiles = build_quantile_profile(
        con,
        logger,
        "customer_gaps",
        ["gap_days"],
        run_dir / "gap_profile_quantiles.csv",
        "perfil de intervalos",
    )

    eligibility = timed_df(
        con,
        logger,
        "cobertura por mínimo de compras",
        """
        SELECT
            threshold AS min_purchase_days,
            COUNT(*) FILTER (WHERE distinct_purchase_days >= threshold)::BIGINT AS eligible_customers,
            ROUND(
                100.0 * COUNT(*) FILTER (WHERE distinct_purchase_days >= threshold) / COUNT(*),
                4
            ) AS eligible_pct
        FROM customer_stats
        CROSS JOIN (VALUES (1), (2), (3), (4), (5), (6), (10), (12)) limits(threshold)
        GROUP BY threshold
        ORDER BY threshold
        """,
    )
    save_csv(eligibility, run_dir / "eligibility_by_purchase_days.csv")

    gap_row = gap_quantiles.iloc[0]
    logger.info(
        "Intervalos preliminares | n=%s | P50=%s días | P90=%s | P95=%s | P99=%s",
        f"{int(gap_row['n']):,}",
        gap_row["p50"],
        gap_row["p90"],
        gap_row["p95"],
        gap_row["p99"],
    )
    for threshold in [2, 3, 4]:
        row = eligibility.loc[eligibility["min_purchase_days"] == threshold].iloc[0]
        logger.info(
            "Elegibilidad >=%s fechas: %s clientes (%.2f%%)",
            threshold,
            f"{int(row['eligible_customers']):,}",
            row["eligible_pct"],
        )

    purchase_day_bins = timed_df(
        con,
        logger,
        "distribución de compras por cliente",
        """
        SELECT
            CASE
                WHEN distinct_purchase_days = 1 THEN '1'
                WHEN distinct_purchase_days = 2 THEN '2'
                WHEN distinct_purchase_days = 3 THEN '3'
                WHEN distinct_purchase_days = 4 THEN '4'
                WHEN distinct_purchase_days = 5 THEN '5'
                WHEN distinct_purchase_days BETWEEN 6 AND 10 THEN '6-10'
                WHEN distinct_purchase_days BETWEEN 11 AND 20 THEN '11-20'
                WHEN distinct_purchase_days BETWEEN 21 AND 50 THEN '21-50'
                WHEN distinct_purchase_days BETWEEN 51 AND 100 THEN '51-100'
                ELSE '101+'
            END AS bin,
            COUNT(*)::BIGINT AS customers
        FROM customer_stats
        GROUP BY 1
        ORDER BY MIN(distinct_purchase_days)
        """,
    )
    save_csv(purchase_day_bins, run_dir / "purchase_days_distribution.csv")

    gap_bins = timed_df(
        con,
        logger,
        "distribución agrupada de intervalos",
        """
        SELECT
            CASE
                WHEN gap_days BETWEEN 1 AND 7 THEN '1-7'
                WHEN gap_days BETWEEN 8 AND 14 THEN '8-14'
                WHEN gap_days BETWEEN 15 AND 30 THEN '15-30'
                WHEN gap_days BETWEEN 31 AND 60 THEN '31-60'
                WHEN gap_days BETWEEN 61 AND 90 THEN '61-90'
                WHEN gap_days BETWEEN 91 AND 120 THEN '91-120'
                WHEN gap_days BETWEEN 121 AND 180 THEN '121-180'
                WHEN gap_days BETWEEN 181 AND 365 THEN '181-365'
                WHEN gap_days BETWEEN 366 AND 730 THEN '366-730'
                ELSE '731+'
            END AS bin,
            COUNT(*)::BIGINT AS intervals
        FROM customer_gaps
        GROUP BY 1
        ORDER BY MIN(gap_days)
        """,
    )
    save_csv(gap_bins, run_dir / "gap_days_distribution.csv")

    category_name = quote_identifier(DEFAULT_COLUMNS["category_name"])
    top_categories = timed_df(
        con,
        logger,
        "principales categorías",
        f"""
        SELECT
            CAST({category} AS VARCHAR) AS category_code,
            COALESCE(ANY_VALUE({category_name}), 'SIN NOMBRE') AS category_name,
            CAST({category} AS VARCHAR) || ' - ' || COALESCE(ANY_VALUE({category_name}), 'SIN NOMBRE') AS category_label,
            COUNT(*)::BIGINT AS rows,
            COUNT(DISTINCT {customer})::BIGINT AS customers,
            SUM(COALESCE({net_sales}, 0))::DOUBLE AS net_sales
        FROM tx
        GROUP BY {category}
        ORDER BY net_sales DESC NULLS LAST
        LIMIT {args.top_n}
        """,
    )
    save_csv(top_categories, run_dir / "top_categories.csv")

    subcategory_name = quote_identifier(DEFAULT_COLUMNS["subcategory_name"])
    top_subcategories = timed_df(
        con,
        logger,
        "principales subcategorías",
        f"""
        SELECT
            CAST({subcategory} AS VARCHAR) AS subcategory_code,
            COALESCE(ANY_VALUE({subcategory_name}), 'SIN NOMBRE') AS subcategory_name,
            COUNT(*)::BIGINT AS rows,
            COUNT(DISTINCT {customer})::BIGINT AS customers,
            SUM(COALESCE({net_sales}, 0))::DOUBLE AS net_sales
        FROM tx
        GROUP BY {subcategory}
        ORDER BY net_sales DESC NULLS LAST
        LIMIT {args.top_n}
        """,
    )
    save_csv(top_subcategories, run_dir / "top_subcategories.csv")

    type_profile: pd.DataFrame | None = None
    if DEFAULT_COLUMNS["type"] in available_columns:
        type_col = quote_identifier(DEFAULT_COLUMNS["type"])
        type_profile = timed_df(
            con,
            logger,
            "distribución de TIPO",
            f"""
            SELECT
                COALESCE(CAST({type_col} AS VARCHAR), 'NULL') AS transaction_type,
                COUNT(*)::BIGINT AS rows,
                COUNT(DISTINCT {customer})::BIGINT AS customers,
                SUM(COALESCE({net_sales}, 0))::DOUBLE AS net_sales
            FROM tx
            GROUP BY {type_col}
            ORDER BY rows DESC
            """,
        )
        save_csv(type_profile, run_dir / "transaction_type_profile.csv")

    logger.info("RESUMEN DE CALIDAD NUMÉRICA")
    for row in numeric_profile.itertuples(index=False):
        logger.info(
            "%s | nulos=%s | ceros=%s | negativos=%s | P50=%s | P99=%s",
            row.column,
            f"{row.null_count:,}",
            f"{row.zero_count:,}",
            f"{row.negative_count:,}",
            row.p50,
            row.p99,
        )

    logger.info("INICIO | generación de gráficas")
    plot_started = time.perf_counter()
    make_plots(run_dir, monthly, purchase_day_bins, gap_bins, top_categories, null_profile)
    logger.info("FIN    | generación de gráficas | %.2f s", time.perf_counter() - plot_started)

    duration = time.perf_counter() - started
    output_files = sorted(path.name for path in run_dir.iterdir() if path.is_file())
    summary = {
        "run_name": run_name,
        "created_at": datetime.now(),
        "duration_seconds": round(duration, 2),
        "input_path": input_path,
        "output_dir": run_dir,
        "hardware": hardware,
        "parquet": parquet_info,
        "overview": overview_record,
        "identity_consistency": identity_profile,
        "null_profile": records(null_profile),
        "numeric_profile": records(numeric_profile),
        "eligibility": records(eligibility),
        "customer_quantiles": records(customer_quantiles),
        "gap_quantiles": records(gap_quantiles),
        "transaction_types": records(type_profile) if type_profile is not None else None,
        "monthly_discontinuity_flags": records(monthly_flags),
        "output_files": output_files + ["resumen.json"],
    }
    with open(run_dir / "resumen.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=json_default)

    con.close()
    logger.info("=" * 78)
    logger.info("DIAGNÓSTICO COMPLETADO EN %.2f s", duration)
    logger.info("Artefactos: %s", run_dir)
    logger.info("Siguiente decisión: limpieza, ventana histórica y población elegible.")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()

