"""Busqueda de horizonte hibrido para definir churn en un entorno no contractual.

Se usan dos anios de historico anteriores a una fecha de corte T.

Para cada cliente se calcula su mediana individual de intervalos entre compras.
La referencia global B es el promedio de las medianas individuales, de modo que
cada cliente aporta el mismo peso al comportamiento central del negocio.

Para cada combinacion de alfa y gamma se define:

    H_i = alfa * B + gamma * (mediana_i - B)

Donde:
- alfa controla el horizonte operativo global.
- gamma controla cuanto se personaliza el horizonte segun el ciclo individual.
- gamma = 0 produce un horizonte completamente global.
- gamma = 1 incorpora completamente la desviacion del cliente respecto de B.

Clasificacion:
- no_churn: compra dentro de (T, T + H_i]
- churn_provisional: no compra dentro de H_i
- churn_reactivado: churn provisional que compra antes de T + H_i + xB*B
- churn_persistente: churn provisional que no compra hasta T + H_i + xB*B

La ventana de validacion posterior conserva la referencia global B para que la
comparacion entre combinaciones de alfa y gamma utilice el mismo criterio temporal.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "HISTORICO_300K_FULL.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "churn" / "target_search"
COLS = ["CODIGO_FAMILIA", "DIM_PERIODO"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Busqueda de horizontes hibridos de churn")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--alpha-min", type=float, default=1.5)
    p.add_argument("--alpha-max", type=float, default=2.5)
    p.add_argument("--alpha-step", type=float, default=0.25)
    p.add_argument("--gamma-min", type=float, default=0.0)
    p.add_argument("--gamma-max", type=float, default=1.0)
    p.add_argument("--gamma-step", type=float, default=0.25)
    p.add_argument("--xb", type=float, default=2.0)
    return p.parse_args()


def parameter_grid(start: float, stop: float, step: float, name: str) -> np.ndarray:
    if step <= 0 or stop < start:
        raise ValueError(f"Parametros de {name} invalidos")
    n = int(round((stop - start) / step))
    return np.round(start + np.arange(n + 1) * step, 10)


def main() -> None:
    args = parse_args()

    if args.alpha_min <= 0:
        raise ValueError("--alpha-min debe ser mayor que 0")
    if args.gamma_min < 0:
        raise ValueError("--gamma-min no puede ser negativo")
    if args.xb <= 0:
        raise ValueError("--xb debe ser mayor que 0")

    started = time.perf_counter()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TARGET] Leyendo: {input_path}")
    df = pd.read_parquet(input_path, columns=COLS)
    df["DIM_PERIODO"] = pd.to_datetime(df["DIM_PERIODO"], errors="coerce")
    df = df.dropna(subset=COLS).drop_duplicates().sort_values(COLS).reset_index(drop=True)

    fecha_max = df["DIM_PERIODO"].max()
    T = fecha_max - pd.DateOffset(years=1)
    hist_start = T - pd.DateOffset(years=2) + pd.Timedelta(days=1)

    hist = df[df["DIM_PERIODO"].between(hist_start, T)].copy()
    future = df[df["DIM_PERIODO"] > T].copy()

    hist["gap_dias"] = hist.groupby("CODIGO_FAMILIA", sort=False)["DIM_PERIODO"].diff().dt.days

    perfil = (
        hist.groupby("CODIGO_FAMILIA", sort=False)
        .agg(mediana_cliente=("gap_dias", "median"))
        .dropna()
        .reset_index()
    )

    # Base global: promedio de las medianas individuales; cada cliente pesa una vez.
    B = float(perfil["mediana_cliente"].mean())

    primera_futura = (
        future.groupby("CODIGO_FAMILIA", sort=False)["DIM_PERIODO"]
        .min()
        .rename("primera_compra_futura")
        .reset_index()
    )

    # Solo se evalúan clientes con al menos un intervalo histórico, porque H_i
    # necesita una mediana individual definida.
    clientes = perfil.merge(primera_futura, on="CODIGO_FAMILIA", how="left")
    primera = clientes["primera_compra_futura"]
    tiene_futura = primera.notna()
    n_total = len(clientes)

    alphas = parameter_grid(args.alpha_min, args.alpha_max, args.alpha_step, "alfa")
    gammas = parameter_grid(args.gamma_min, args.gamma_max, args.gamma_step, "gamma")

    resultados = []

    for alfa in alphas:
        for gamma in gammas:
            H = alfa * B + gamma * (clientes["mediana_cliente"] - B)
            H = H.clip(lower=1.0)

            dias_h = np.ceil(H).astype(int)
            limite_h = T + pd.to_timedelta(dias_h, unit="D")

            # La validación usa xB veces la misma base global B para todas las
            # combinaciones, evitando cambiar simultáneamente H_i y el criterio
            # posterior de reactivación.
            dias_validacion = int(np.ceil(args.xb * B))
            limite_check = limite_h + pd.to_timedelta(dias_validacion, unit="D")

            compra_en_h = tiene_futura & (primera <= limite_h)
            churn_provisional = ~compra_en_h
            churn_reactivado = churn_provisional & tiene_futura & (primera <= limite_check)
            churn_persistente = churn_provisional & ~churn_reactivado

            n_prov = int(churn_provisional.sum())
            n_reac = int(churn_reactivado.sum())
            n_pers = int(churn_persistente.sum())

            resultados.append({
                "alfa": float(alfa),
                "gamma": float(gamma),
                "H_media_dias": float(H.mean()),
                "H_mediana_dias": float(H.median()),
                "churn_provisional": n_prov,
                "churn_reactivado": n_reac,
                "churn_persistente": n_pers,
                "pct_churn_provisional": n_prov / n_total * 100.0,
                "pct_reactivacion": n_reac / n_prov * 100.0 if n_prov else np.nan,
                "pct_persistencia": n_pers / n_prov * 100.0 if n_prov else np.nan,
            })

    tabla = pd.DataFrame(resultados)
    csv_path = output_dir / "busqueda_horizonte_hibrido.csv"
    log_path = output_dir / "busqueda_horizonte_hibrido.log"
    tabla.to_csv(csv_path, index=False)

    resumen = (
        "\n=== CONFIGURACION ===\n"
        f"inicio_historico: {hist_start.date()}\n"
        f"fecha_corte_T: {T.date()}\n"
        f"fecha_max_dataset: {fecha_max.date()}\n"
        f"clientes_historicos_con_intervalos: {n_total}\n"
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
        + tabla.to_string(index=False, float_format=lambda x: f"{x:,.2f}")
        + f"\n\nTiempo total: {time.perf_counter() - started:.2f} s\n"
    )

    print(resumen)
    log_path.write_text(resumen, encoding="utf-8")
    print(f"[TARGET] Tabla: {csv_path}")
    print(f"[TARGET] Log: {log_path}")


if __name__ == "__main__":
    main()
