from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from churn_features import (
    calcular_compras_ultimos_180d,
    calcular_delta_frecuencia_180d,
    calcular_dias_desde_ultima_compra,
    calcular_gasto_total,
    calcular_intervalo_cv,
    calcular_intervalo_maximo,
    calcular_intervalo_promedio,
    calcular_longitud_relacion_dias,
    calcular_recencia_relativa,
    calcular_subcategorias_distintas,
    calcular_ticket_promedio,
    calcular_total_compras,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "historico_con_demograficos.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "churn" / "caracteristicas_churn.parquet"

REQUIRED_COLUMNS = [
    "IDENTIFICACION",
    "DIM_PERIODO",
    "DIM_FACTURA",
    "VENTA_NETA",
    "COD_SUBCATEGORIA",
    "EDAD",
    "SEXO",
    "EDAD_IMPUTADA",
]

FEATURE_COLUMNS = [
    "dias_desde_ultima_compra",
    "total_compras_24m",
    "gasto_total_24m",
    "ticket_promedio_24m",
    "longitud_relacion_dias",
    "intervalo_promedio",
    "intervalo_maximo",
    "intervalo_cv",
    "recencia_relativa",
    "compras_ultimos_180d",
    "delta_frecuencia_180d",
    "subcategorias_distintas_24m",
    "EDAD",
    "SEXO",
    "EDAD_IMPUTADA",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calculo de caracteristicas para churn")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def validar_columnas(df: pd.DataFrame) -> None:
    faltantes = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")


def cargar_dataset(path: Path) -> pd.DataFrame:
    print(f"\n[1/5] Leyendo {path} ...")
    t0 = time.time()
    df = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    validar_columnas(df)

    df["IDENTIFICACION"] = df["IDENTIFICACION"].astype("string").str.strip()
    df["DIM_PERIODO"] = pd.to_datetime(df["DIM_PERIODO"], errors="coerce")
    df["VENTA_NETA"] = pd.to_numeric(df["VENTA_NETA"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["IDENTIFICACION", "DIM_PERIODO"]).copy()

    print(f"   filas: {len(df):,}")
    print(f"   clientes: {df['IDENTIFICACION'].nunique():,}")
    print(f"   tiempo lectura: {time.time() - t0:.1f}s")
    return df


def calcular_features_basicas(df: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.DataFrame:
    print("\n[2/5] Calculando features basicas por cliente ...")
    t0 = time.time()

    agrupado = df.groupby("IDENTIFICACION", sort=False)

    features = agrupado.agg(
        total_compras_24m=("DIM_FACTURA", "nunique"),
        gasto_total_24m=("VENTA_NETA", "sum"),
        primera_compra=("DIM_PERIODO", "min"),
        ultima_compra=("DIM_PERIODO", "max"),
        subcategorias_distintas_24m=("COD_SUBCATEGORIA", "nunique"),
        EDAD=("EDAD", "first"),
        SEXO=("SEXO", "first"),
        EDAD_IMPUTADA=("EDAD_IMPUTADA", "first"),
    ).reset_index()

    features["dias_desde_ultima_compra"] = (
        fecha_corte - features["ultima_compra"]
    ).dt.days.astype("float64")

    features["longitud_relacion_dias"] = (
        fecha_corte - features["primera_compra"]
    ).dt.days.astype("float64")

    denominador = features["total_compras_24m"].replace(0, np.nan)
    features["ticket_promedio_24m"] = (
        features["gasto_total_24m"] / denominador
    ).fillna(0.0)

    features = features.drop(columns=["primera_compra", "ultima_compra"])

    print(f"   clientes calculados: {len(features):,}")
    print(f"   tiempo: {time.time() - t0:.1f}s")
    return features


def calcular_features_temporales(df: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.DataFrame:
    print("\n[3/5] Calculando intervalos, recencia relativa y ventanas de 180 dias ...")
    t0 = time.time()

    resultados = []

    for identificacion, grupo in df.groupby("IDENTIFICACION", sort=False):
        fechas = grupo["DIM_PERIODO"]
        facturas = grupo["DIM_FACTURA"]

        dias_ultima = calcular_dias_desde_ultima_compra(fechas, fecha_corte)
        intervalo_promedio = calcular_intervalo_promedio(fechas)

        resultados.append({
            "IDENTIFICACION": identificacion,
            "intervalo_promedio": intervalo_promedio,
            "intervalo_maximo": calcular_intervalo_maximo(fechas),
            "intervalo_cv": calcular_intervalo_cv(fechas),
            "recencia_relativa": calcular_recencia_relativa(dias_ultima, intervalo_promedio),
            "compras_ultimos_180d": calcular_compras_ultimos_180d(
                fechas,
                facturas,
                fecha_corte,
            ),
            "delta_frecuencia_180d": calcular_delta_frecuencia_180d(
                fechas,
                facturas,
                fecha_corte,
            ),
        })

    out = pd.DataFrame(resultados)
    print(f"   clientes calculados: {len(out):,}")
    print(f"   tiempo: {time.time() - t0:.1f}s")
    return out


def validar_resultado(features: pd.DataFrame, clientes_esperados: int) -> None:
    print("\n[4/5] Validando dataset de caracteristicas ...")

    if len(features) != clientes_esperados:
        raise AssertionError(
            f"Se esperaban {clientes_esperados:,} clientes y se obtuvieron {len(features):,}"
        )

    if features["IDENTIFICACION"].duplicated().any():
        raise AssertionError("El dataset final contiene IDENTIFICACION duplicada")

    faltantes = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if faltantes:
        raise AssertionError(f"Faltan features esperadas: {faltantes}")

    numericas = [c for c in FEATURE_COLUMNS if c != "SEXO"]
    inf_total = int(np.isinf(features[numericas].select_dtypes(include=[np.number])).sum().sum())

    print(f"   filas finales: {len(features):,}")
    print(f"   clientes unicos: {features['IDENTIFICACION'].nunique():,}")
    print(f"   identificaciones duplicadas: {features['IDENTIFICACION'].duplicated().sum():,}")
    print(f"   valores infinitos: {inf_total:,}")
    print("\n   NULOS POR FEATURE:")
    print(features[FEATURE_COLUMNS].isna().sum().to_string())

    print("\n   RESUMEN FEATURES NUMERICAS:")
    print(features[numericas].describe().T.to_string())

    print("\n   DISTRIBUCION SEXO:")
    print(features["SEXO"].value_counts(dropna=False).to_string())


def main() -> None:
    args = parse_args()

    df = cargar_dataset(args.input)
    fecha_corte = df["DIM_PERIODO"].max()
    clientes_esperados = df["IDENTIFICACION"].nunique()

    print("\n   === CONFIGURACION ===")
    print(f"   fecha de corte T: {fecha_corte.date()}")
    print(f"   clientes esperados: {clientes_esperados:,}")

    basicas = calcular_features_basicas(df, fecha_corte)
    temporales = calcular_features_temporales(df, fecha_corte)

    features = basicas.merge(
        temporales,
        on="IDENTIFICACION",
        how="inner",
        validate="one_to_one",
    )

    columnas_finales = ["IDENTIFICACION"] + FEATURE_COLUMNS
    features = features[columnas_finales].copy()

    validar_resultado(features, clientes_esperados)

    print("\n[5/5] Guardando resultado ...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)

    print(f"   archivo: {args.output}")
    print(f"   tamanio: {args.output.stat().st_size / 1e6:.1f} MB")
    print("   dataset listo para integrar posteriormente el target de churn")


if __name__ == "__main__":
    main()
