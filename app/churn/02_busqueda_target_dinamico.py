"""Busqueda de horizonte hibrido para definir churn en un entorno no contractual.

Se usan dos anios de historico anteriores a una fecha de corte T.
La unidad de analisis es IDENTIFICACION.

Solo se incluyen clientes con al menos cuatro fechas distintas de compra en el
periodo historico, de modo que la mediana individual se estime con al menos tres
intervalos entre compras.

Para cada cliente elegible se calcula su mediana individual de intervalos entre
compras. La referencia global B es el promedio de las medianas individuales, por
lo que cada cliente aporta el mismo peso al comportamiento central del negocio.

Para cada combinacion de alfa y gamma se define:

    H_i = alfa * B + gamma * mediana_i

Donde:
- alfa controla el componente global del horizonte.
- gamma controla el peso adicional del ciclo individual del cliente.
- B es el promedio global de las medianas individuales.
- mediana_i es la mediana de los intervalos historicos del cliente i.

Clasificacion:
- no_churn: compra dentro de (T, T + H_i]
- churn_provisional: no compra dentro de H_i
- churn_reactivado: churn provisional que compra antes de T + H_i + xB*B
- churn_persistente: churn provisional que no compra hasta T + H_i + xB*B

La ventana de validacion posterior utiliza xB veces la referencia global B.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    REPO_ROOT / "data" / "raw" / "HISTORICO_300K_FULL.parquet"
)

DEFAULT_OUTPUT = (
    REPO_ROOT / "data" / "churn" / "target_search"
)

COLS = [
    "IDENTIFICACION",
    "DIM_PERIODO",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Busqueda de horizontes hibridos de churn"
    )

    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    p.add_argument(
        "--alpha-min",
        type=float,
        default=1.5,
    )

    p.add_argument(
        "--alpha-max",
        type=float,
        default=2.5,
    )

    p.add_argument(
        "--alpha-step",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--gamma-min",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--gamma-max",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--gamma-step",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--xb",
        type=float,
        default=2.0,
    )

    p.add_argument(
        "--min-compras",
        type=int,
        default=4,
        help=(
            "Numero minimo de fechas distintas de compra requeridas "
            "en el periodo historico. Default: 4."
        ),
    )

    return p.parse_args()


def parameter_grid(
    start: float,
    stop: float,
    step: float,
    name: str,
) -> np.ndarray:

    if step <= 0 or stop < start:
        raise ValueError(
            f"Parametros de {name} invalidos"
        )

    n = int(
        round(
            (stop - start) / step
        )
    )

    return np.round(
        start + np.arange(n + 1) * step,
        10,
    )


def main() -> None:

    args = parse_args()

    if args.alpha_min <= 0:
        raise ValueError(
            "--alpha-min debe ser mayor que 0"
        )

    if args.gamma_min < 0:
        raise ValueError(
            "--gamma-min no puede ser negativo"
        )

    if args.xb <= 0:
        raise ValueError(
            "--xb debe ser mayor que 0"
        )

    if args.min_compras < 2:
        raise ValueError(
            "--min-compras debe ser al menos 2"
        )

    started = time.perf_counter()

    input_path = (
        args.input
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[TARGET] Leyendo: {input_path}"
    )

    # ---------------------------------------------------------
    # 1. LECTURA DE DATOS
    # ---------------------------------------------------------

    df = pd.read_parquet(
        input_path,
        columns=COLS,
    )

    df["DIM_PERIODO"] = pd.to_datetime(
        df["DIM_PERIODO"],
        errors="coerce",
    )

    # Una misma persona comprando varias lineas/productos el mismo dia
    # cuenta como una unica fecha de compra para calcular intervalos.
    df = (
        df
        .dropna(
            subset=COLS
        )
        .drop_duplicates(
            subset=[
                "IDENTIFICACION",
                "DIM_PERIODO",
            ]
        )
        .sort_values(
            [
                "IDENTIFICACION",
                "DIM_PERIODO",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ---------------------------------------------------------
    # 2. DEFINICION DE VENTANAS TEMPORALES
    # ---------------------------------------------------------

    fecha_max = df[
        "DIM_PERIODO"
    ].max()

    # Se reserva el ultimo anio para observar comportamiento futuro.
    T = (
        fecha_max
        - pd.DateOffset(
            years=1
        )
    )

    # Dos anios de historia anteriores a T.
    hist_start = (
        T
        - pd.DateOffset(
            years=2
        )
        + pd.Timedelta(
            days=1
        )
    )

    hist = df[
        df[
            "DIM_PERIODO"
        ].between(
            hist_start,
            T,
        )
    ].copy()

    future = df[
        df[
            "DIM_PERIODO"
        ] > T
    ].copy()

    # ---------------------------------------------------------
    # 3. FILTRO DE CLIENTES CON HISTORIAL SUFICIENTE
    # ---------------------------------------------------------

    clientes_hist_antes = (
        hist[
            "IDENTIFICACION"
        ].nunique()
    )

    # Como previamente se elimino IDENTIFICACION-fecha duplicada,
    # el size corresponde al numero de fechas distintas de compra.
    conteo_compras = (
        hist
        .groupby(
            "IDENTIFICACION",
            sort=False,
        )
        .size()
        .rename(
            "n_compras_historicas"
        )
    )

    ids_elegibles = (
        conteo_compras[
            conteo_compras
            >= args.min_compras
        ]
        .index
    )

    hist = (
        hist[
            hist[
                "IDENTIFICACION"
            ].isin(
                ids_elegibles
            )
        ]
        .copy()
    )

    # ---------------------------------------------------------
    # 4. INTERVALOS HISTORICOS POR CLIENTE
    # ---------------------------------------------------------

    hist["gap_dias"] = (
        hist
        .groupby(
            "IDENTIFICACION",
            sort=False,
        )[
            "DIM_PERIODO"
        ]
        .diff()
        .dt
        .days
    )

    perfil = (
        hist
        .groupby(
            "IDENTIFICACION",
            sort=False,
        )
        .agg(
            n_compras_historicas=(
                "DIM_PERIODO",
                "size",
            ),
            mediana_cliente=(
                "gap_dias",
                "median",
            ),
        )
        .reset_index()
    )

    # Seguridad adicional:
    # mantener unicamente clientes con el minimo requerido.
    perfil = (
        perfil[
            perfil[
                "n_compras_historicas"
            ]
            >= args.min_compras
        ]
        .dropna(
            subset=[
                "mediana_cliente"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ---------------------------------------------------------
    # 5. BASE GLOBAL
    # ---------------------------------------------------------

    # Promedio de las medianas individuales.
    # Cada cliente pesa una sola vez.
    B = float(
        perfil[
            "mediana_cliente"
        ].mean()
    )

    # ---------------------------------------------------------
    # 6. PRIMERA COMPRA FUTURA
    # ---------------------------------------------------------

    primera_futura = (
        future
        .groupby(
            "IDENTIFICACION",
            sort=False,
        )[
            "DIM_PERIODO"
        ]
        .min()
        .rename(
            "primera_compra_futura"
        )
        .reset_index()
    )

    clientes = (
        perfil
        .merge(
            primera_futura,
            on="IDENTIFICACION",
            how="left",
        )
    )

    primera = clientes[
        "primera_compra_futura"
    ]

    tiene_futura = (
        primera.notna()
    )

    n_total = len(
        clientes
    )

    # ---------------------------------------------------------
    # 7. GRID DE ALFA Y GAMMA
    # ---------------------------------------------------------

    alphas = parameter_grid(
        args.alpha_min,
        args.alpha_max,
        args.alpha_step,
        "alfa",
    )

    gammas = parameter_grid(
        args.gamma_min,
        args.gamma_max,
        args.gamma_step,
        "gamma",
    )

    resultados = []

    # ---------------------------------------------------------
    # 8. BUSQUEDA DEL HORIZONTE HIBRIDO
    # ---------------------------------------------------------

    for alfa in alphas:

        for gamma in gammas:

            # Horizonte individual:
            #
            # H_i = alfa * B + gamma * mediana_i
            #
            # Ejemplo:
            # alfa = 1.75
            # gamma = 1
            #
            # H_i = 1.75 * B + mediana_i

            H = (
                alfa * B
                + gamma
                * clientes[
                    "mediana_cliente"
                ]
            )

            H = H.clip(
                lower=1.0
            )

            dias_h = (
                np
                .ceil(H)
                .astype(int)
            )

            limite_h = (
                T
                + pd.to_timedelta(
                    dias_h,
                    unit="D",
                )
            )

            # -------------------------------------------------
            # VALIDACION POSTERIOR
            #
            # Se observa xB veces la base global B despues
            # del horizonte individual H_i.
            #
            # Ejemplo:
            # xb = 1 -> H_i + 1B
            # xb = 2 -> H_i + 2B
            # xb = 3 -> H_i + 3B
            # -------------------------------------------------

            dias_validacion = int(
                np.ceil(
                    args.xb * B
                )
            )

            limite_check = (
                limite_h
                + pd.to_timedelta(
                    dias_validacion,
                    unit="D",
                )
            )

            # Compra dentro del horizonte:
            # no se considera churn.
            compra_en_h = (
                tiene_futura
                & (
                    primera
                    <= limite_h
                )
            )

            # No compro dentro del horizonte.
            churn_provisional = (
                ~compra_en_h
            )

            # Fue churn provisional,
            # pero regreso durante la ventana de validacion.
            churn_reactivado = (
                churn_provisional
                & tiene_futura
                & (
                    primera
                    <= limite_check
                )
            )

            # No compro dentro de H_i
            # ni durante la ventana adicional de validacion.
            churn_persistente = (
                churn_provisional
                & ~churn_reactivado
            )

            n_prov = int(
                churn_provisional.sum()
            )

            n_reac = int(
                churn_reactivado.sum()
            )

            n_pers = int(
                churn_persistente.sum()
            )

            resultados.append(
                {
                    "alfa": float(
                        alfa
                    ),
                    "gamma": float(
                        gamma
                    ),
                    "H_media_dias": float(
                        H.mean()
                    ),
                    "H_mediana_dias": float(
                        H.median()
                    ),
                    "churn_provisional": n_prov,
                    "churn_reactivado": n_reac,
                    "churn_persistente": n_pers,
                    "pct_churn_provisional": (
                        n_prov
                        / n_total
                        * 100.0
                    ),
                    "pct_reactivacion": (
                        n_reac
                        / n_prov
                        * 100.0
                        if n_prov
                        else np.nan
                    ),
                    "pct_persistencia": (
                        n_pers
                        / n_prov
                        * 100.0
                        if n_prov
                        else np.nan
                    ),
                }
            )

    # ---------------------------------------------------------
    # 9. SALIDA
    # ---------------------------------------------------------

    tabla = pd.DataFrame(
        resultados
    )

    csv_path = (
        output_dir
        / "busqueda_horizonte_hibrido.csv"
    )

    log_path = (
        output_dir
        / "busqueda_horizonte_hibrido.log"
    )

    tabla.to_csv(
        csv_path,
        index=False,
    )

    resumen = (
        "\n=== CONFIGURACION ===\n"
        f"inicio_historico: {hist_start.date()}\n"
        f"fecha_corte_T: {T.date()}\n"
        f"fecha_max_dataset: {fecha_max.date()}\n"
        f"unidad_analisis: IDENTIFICACION\n"
        f"clientes_historicos_antes_filtro: {clientes_hist_antes}\n"
        f"min_compras_historicas: {args.min_compras}\n"
        f"min_intervalos_historicos: {args.min_compras - 1}\n"
        f"clientes_elegibles: {n_total}\n"
        f"B_global_mediana_clientes_dias: {B:.2f}\n"
        f"xB_validacion: {args.xb:.2f}\n"
        f"alpha_min: {args.alpha_min:.2f}\n"
        f"alpha_max: {args.alpha_max:.2f}\n"
        f"alpha_step: {args.alpha_step:.2f}\n"
        f"gamma_min: {args.gamma_min:.2f}\n"
        f"gamma_max: {args.gamma_max:.2f}\n"
        f"gamma_step: {args.gamma_step:.2f}\n"
        f"numero_combinaciones: {len(alphas) * len(gammas)}\n"
        "\n=== RESULTADOS ===\n"
        + tabla.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
        + (
            f"\n\nTiempo total: "
            f"{time.perf_counter() - started:.2f} s\n"
        )
    )

    print(
        resumen
    )

    log_path.write_text(
        resumen,
        encoding="utf-8",
    )

    print(
        f"[TARGET] Tabla: {csv_path}"
    )

    print(
        f"[TARGET] Log: {log_path}"
    )


if __name__ == "__main__":
    main()