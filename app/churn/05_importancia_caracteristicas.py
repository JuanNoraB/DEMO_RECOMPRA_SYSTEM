"""Analisis de importancia de caracteristicas para el modelo de churn.

Este script entrena un LightGBM sencillo sobre el conjunto de TRAINING y calcula
la importancia de cada variable mediante GAIN y SPLIT.

Importante:
- El archivo de entrada debe contener una fila por IDENTIFICACION.
- Debe incluir las 15 caracteristicas generadas en 04_calculo_caracteristicas.py.
- Debe incluir la columna objetivo (por defecto: target).
- Debe ejecutarse unicamente sobre training. Test no debe participar en la
  seleccion de variables.

Ejemplo:
    python app/churn/05_importancia_caracteristicas.py \
        --input data/churn/dataset_churn_train.parquet

Salidas por defecto:
    data/churn/importancia_caracteristicas_churn.csv
    data/logs/05_importancia_caracteristicas_churn.log
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
DEFAULT_INPUT = REPO_ROOT / "data" / "churn" / "dataset_churn_train.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "churn" / "importancia_caracteristicas_churn.csv"
DEFAULT_LOG = REPO_ROOT / "data" / "logs" / "05_importancia_caracteristicas_churn.log"

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
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
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
        help="Porcentaje acumulado de gain usado para reportar el subconjunto principal",
    )
    return p.parse_args()


def crear_logger(path: Path) -> Callable[[str], None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"=== 05_importancia_caracteristicas.py | {datetime.now().isoformat(timespec='seconds')} ===\n",
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

    if df[ID_COL].duplicated().any():
        raise ValueError(
            "El dataset debe contener una sola fila por IDENTIFICACION; se encontraron duplicados"
        )

    if df[target_col].isna().any():
        raise ValueError(f"La columna objetivo {target_col!r} contiene valores nulos")

    valores_target = set(pd.Series(df[target_col]).dropna().astype(int).unique().tolist())
    if not valores_target.issubset({0, 1}):
        raise ValueError(
            f"La columna objetivo {target_col!r} debe ser binaria 0/1. Valores: {sorted(valores_target)}"
        )


def preparar_datos(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[FEATURE_COLUMNS].copy()

    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_COLUMNS:
            X[col] = X[col].astype("string").fillna("DESCONOCIDO").astype("category")
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


def seleccionar_hasta_umbral(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    alcanzado = df["gain_acumulado_pct"] >= threshold
    if not alcanzado.any():
        return df.copy()
    ultimo = alcanzado.idxmax()
    return df.loc[:ultimo].copy()


def main() -> None:
    args = parse_args()
    log = crear_logger(args.log)
    inicio = time.time()

    if args.num_boost_round <= 0:
        raise ValueError("--num-boost-round debe ser mayor que 0")
    if not 0 < args.gain_threshold <= 100:
        raise ValueError("--gain-threshold debe estar en el intervalo (0, 100]")

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    log(f"\n[1/5] Leyendo {input_path} ...")
    t0 = time.time()
    if not input_path.exists():
        raise FileNotFoundError(
            f"No existe {input_path}. Este script necesita el dataset de training ya integrado con el target."
        )

    df = pd.read_parquet(input_path)
    validar_dataset(df, args.target_col)

    n_clientes = df[ID_COL].nunique()
    n_pos = int(pd.to_numeric(df[args.target_col]).sum())
    n_neg = len(df) - n_pos
    prevalencia = n_pos / len(df) if len(df) else 0.0

    log(f"   filas: {len(df):,}")
    log(f"   clientes: {n_clientes:,}")
    log(f"   churn=1: {n_pos:,} ({prevalencia * 100:.2f}%)")
    log(f"   churn=0: {n_neg:,}")
    log(f"   tiempo lectura: {time.time() - t0:.1f}s")

    if n_pos == 0 or n_neg == 0:
        raise ValueError("El dataset de training debe contener ambas clases")

    log("\n[2/5] Preparando matriz de caracteristicas ...")
    t0 = time.time()
    X, y = preparar_datos(df, args.target_col)
    log(f"   features: {len(FEATURE_COLUMNS)}")
    log(f"   categoricas: {CATEGORICAL_COLUMNS}")
    log(f"   tiempo: {time.time() - t0:.1f}s")

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
        "verbosity": -1,
    }

    log("\n[3/5] Entrenando LightGBM para estimar importancia ...")
    log(f"   num_boost_round: {args.num_boost_round}")
    log(f"   num_threads: {num_threads}")
    log(f"   scale_pos_weight: {scale_pos_weight:.4f}")
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

    log(
        f"\n   FEATURES HASTA ALCANZAR {args.gain_threshold:.0f}% DEL GAIN "
        f"({len(top_gain)} de {len(resultado)}):"
    )
    log(f"   {'Rank':>4}  {'Feature':<32} {'Gain %':>9} {'Acum %':>9} {'Split %':>9}")
    log("   " + "-" * 70)
    for _, row in top_gain.iterrows():
        log(
            f"   {int(row['rank']):>4}  {row['feature']:<32} "
            f"{row['gain_pct']:>9.2f} {row['gain_acumulado_pct']:>9.2f} "
            f"{row['split_pct']:>9.2f}"
        )

    log("\n[5/5] Guardando resultado ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(output_path, index=False)

    log(f"   archivo: {output_path}")
    log(f"   log: {args.log.expanduser().resolve()}")
    log(f"   tiempo total: {time.time() - inicio:.1f}s")
    log("   proceso finalizado")


if __name__ == "__main__":
    main()
