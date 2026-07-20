"""Analisis de importancia de caracteristicas para churn mediante LightGBM.

Continuidad del flujo:
    03_join_cc_demograficos.py
        -> data/raw/historico_con_demograficos.parquet
    04_calculo_caracteristicas.py
        -> data/churn/caracteristicas_churn.parquet
    05_generar_target_final.py
        -> data/churn/dataset_churn_final.parquet
    06_importancia_caracteristicas.py
        -> data/churn/importancia_caracteristicas_churn.csv

El script entrena un LightGBM sencillo para cuantificar la importancia de las 15
caracteristicas de churn mediante GAIN y SPLIT. No corresponde al entrenamiento
del modelo final ni realiza ajuste de hiperparametros.

Ejemplo por defecto:
    python app/churn/06_importancia_caracteristicas.py

Ejemplo con rutas personalizadas:
    python app/churn/06_importancia_caracteristicas.py \
        --input data/churn/dataset_churn_final.parquet \
        --output data/churn/importancia_caracteristicas_churn.csv
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "churn" / "dataset_churn_final.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "churn" / "importancia_caracteristicas_churn.csv"
DEFAULT_LOG = REPO_ROOT / "data" / "logs" / "06_importancia_caracteristicas_churn.log"

ID_COL = "IDENTIFICACION"
DEFAULT_TARGET_COL = "target"
CATEGORICAL_COLUMNS = ["SEXO"]

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
    p = argparse.ArgumentParser(
        description="Importancia de caracteristicas de churn mediante LightGBM"
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Dataset generado por 05_generar_target_final.py",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV de salida con el ranking de importancia",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help="Archivo de log de la ejecucion",
    )
    p.add_argument("--target-col", type=str, default=DEFAULT_TARGET_COL)
    p.add_argument("--num-boost-round", type=int, default=300)
    p.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="0 = usar todos los CPU disponibles",
    )
    p.add_argument(
        "--gain-threshold",
        type=float,
        default=95.0,
        help="Porcentaje acumulado de gain para definir el subconjunto principal",
    )
    return p.parse_args()


def crear_logger(path: Path) -> Callable[[str], None]:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "=== 06_importancia_caracteristicas.py | "
            f"{datetime.now().isoformat(timespec='seconds')} ===\n"
        ),
        encoding="utf-8",
    )

    def log(mensaje: str = "") -> None:
        print(mensaje, flush=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(mensaje + "\n")

    return log


def validar_dataset(df: pd.DataFrame, target_col: str) -> None:
    requeridas = [ID_COL, target_col] + FEATURE_COLUMNS
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    if df[ID_COL].isna().any():
        raise ValueError(f"La columna {ID_COL} contiene valores nulos")

    if df[ID_COL].duplicated().any():
        raise ValueError(
            "El dataset debe contener una sola fila por IDENTIFICACION; "
            "se encontraron duplicados"
        )

    if df[target_col].isna().any():
        raise ValueError(f"La columna objetivo {target_col!r} contiene valores nulos")

    target_numerico = pd.to_numeric(df[target_col], errors="raise")
    valores_target = set(target_numerico.astype(int).unique().tolist())
    if not valores_target.issubset({0, 1}):
        raise ValueError(
            f"La columna objetivo {target_col!r} debe ser binaria 0/1. "
            f"Valores encontrados: {sorted(valores_target)}"
        )


def preparar_datos(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[FEATURE_COLUMNS].copy()

    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_COLUMNS:
            X[col] = (
                X[col]
                .astype("string")
                .fillna("DESCONOCIDO")
                .astype("category")
            )
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    y = pd.to_numeric(df[target_col], errors="raise").astype("int8").to_numpy()
    return X, y


def callback_progreso(log: Callable[[str], None], cada: int = 50):
    def _callback(env: lgb.callback.CallbackEnv) -> None:
        actual = env.iteration + 1
        total = env.end_iteration
        if actual == 1 or actual % cada == 0 or actual == total:
            log(f"   arboles entrenados: {actual}/{total}")

    _callback.order = 10
    _callback.before_iteration = False
    return _callback


def seleccionar_hasta_umbral(
    resultado: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    alcanzado = resultado["gain_acumulado_pct"] >= threshold
    if not alcanzado.any():
        return resultado.copy()
    ultimo = alcanzado.idxmax()
    return resultado.loc[:ultimo].copy()


def main() -> None:
    args = parse_args()

    if args.num_boost_round <= 0:
        raise ValueError("--num-boost-round debe ser mayor que 0")
    if not 0 < args.gain_threshold <= 100:
        raise ValueError("--gain-threshold debe estar en el intervalo (0, 100]")

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    log_path = args.log.expanduser().resolve()

    log = crear_logger(log_path)
    inicio = time.time()

    log("\n=== CONFIGURACION ===")
    log(f"input: {input_path}")
    log(f"output: {output_path}")
    log(f"log: {log_path}")
    log(f"target: {args.target_col}")
    log(f"num_boost_round: {args.num_boost_round}")
    log(f"gain_threshold: {args.gain_threshold:.2f}%")

    log(f"\n[1/5] Leyendo dataset: {input_path}")
    t0 = time.time()
    if not input_path.exists():
        raise FileNotFoundError(
            f"No existe {input_path}. Ejecute primero 05_generar_target_final.py "
            "o indique otro archivo mediante --input."
        )

    df = pd.read_parquet(input_path)
    validar_dataset(df, args.target_col)

    n_clientes = df[ID_COL].nunique()
    target_num = pd.to_numeric(df[args.target_col], errors="raise").astype(int)
    n_pos = int((target_num == 1).sum())
    n_neg = int((target_num == 0).sum())
    prevalencia = n_pos / len(df) if len(df) else 0.0

    log(f"   filas: {len(df):,}")
    log(f"   clientes unicos: {n_clientes:,}")
    log(f"   target 0: {n_neg:,}")
    log(f"   target 1: {n_pos:,}")
    log(f"   churn: {prevalencia * 100:.2f}%")
    log(f"   tiempo lectura: {time.time() - t0:.1f}s")

    if n_pos == 0 or n_neg == 0:
        raise ValueError("El dataset debe contener ambas clases para entrenar LightGBM")

    log("\n[2/5] Preparando caracteristicas ...")
    t0 = time.time()
    X, y = preparar_datos(df, args.target_col)

    nulos_numericos = int(
        X.select_dtypes(include=[np.number]).isna().sum().sum()
    )
    log(f"   features utilizadas: {len(FEATURE_COLUMNS)}")
    log(f"   categoricas: {', '.join(CATEGORICAL_COLUMNS)}")
    log(f"   nulos numericos enviados a LightGBM: {nulos_numericos:,}")
    log("   distribucion SEXO:")
    for categoria, cantidad in X["SEXO"].value_counts(dropna=False).items():
        log(f"      {categoria}: {cantidad:,}")
    log(f"   tiempo preparacion: {time.time() - t0:.1f}s")

    num_threads = args.num_threads if args.num_threads > 0 else (os.cpu_count() or 1)
    scale_pos_weight = n_neg / n_pos

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.01,
        "lambda_l2": 0.01,
        "scale_pos_weight": scale_pos_weight,
        "num_threads": num_threads,
        "seed": 42,
        "feature_fraction_seed": 42,
        "bagging_seed": 42,
        "data_random_seed": 42,
        "force_col_wise": True,
        "verbosity": -1,
    }

    log("\n[3/5] Entrenando LightGBM para estimar importancia ...")
    log(f"   num_threads: {num_threads}")
    log(f"   scale_pos_weight: {scale_pos_weight:.4f}")
    log("   progreso: se informa cada 50 arboles")
    t0 = time.time()

    dataset = lgb.Dataset(
        X,
        label=y,
        feature_name=FEATURE_COLUMNS,
        categorical_feature=CATEGORICAL_COLUMNS,
        free_raw_data=False,
    )

    booster = lgb.train(
        params,
        dataset,
        num_boost_round=args.num_boost_round,
        callbacks=[callback_progreso(log, cada=50)],
    )
    log(f"   tiempo entrenamiento: {time.time() - t0:.1f}s")

    log("\n[4/5] Calculando importancia por GAIN y SPLIT ...")
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")

    gain_total = float(gain.sum())
    split_total = float(split.sum())

    resultado = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "gain_abs": gain,
            "gain_pct": gain / gain_total * 100 if gain_total > 0 else 0.0,
            "split_abs": split,
            "split_pct": split / split_total * 100 if split_total > 0 else 0.0,
        }
    ).sort_values("gain_pct", ascending=False).reset_index(drop=True)

    resultado["rank"] = resultado.index + 1
    resultado["gain_acumulado_pct"] = resultado["gain_pct"].cumsum()
    resultado = resultado[
        [
            "rank",
            "feature",
            "gain_abs",
            "gain_pct",
            "gain_acumulado_pct",
            "split_abs",
            "split_pct",
        ]
    ]

    top_gain = seleccionar_hasta_umbral(resultado, args.gain_threshold)

    log("\n   RANKING COMPLETO DE CARACTERISTICAS:")
    log(f"   {'Rank':>4}  {'Feature':<32} {'Gain %':>9} {'Acum %':>9} {'Split %':>9}")
    log("   " + "-" * 70)
    for _, row in resultado.iterrows():
        log(
            f"   {int(row['rank']):>4}  {row['feature']:<32} "
            f"{row['gain_pct']:>9.2f} {row['gain_acumulado_pct']:>9.2f} "
            f"{row['split_pct']:>9.2f}"
        )

    log(
        f"\n   FEATURES NECESARIAS PARA ALCANZAR "
        f"{args.gain_threshold:.0f}% DEL GAIN: {len(top_gain)}"
    )
    for _, row in top_gain.iterrows():
        log(
            f"      {int(row['rank']):>2}. {row['feature']} "
            f"(gain={row['gain_pct']:.2f}%, acumulado={row['gain_acumulado_pct']:.2f}%)"
        )

    log("\n[5/5] Guardando resultado ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(output_path, index=False)

    log(f"   archivo: {output_path}")
    log(f"   log: {log_path}")
    log(f"   tiempo total: {time.time() - inicio:.1f}s")
    log("   proceso finalizado")


if __name__ == "__main__":
    main()
