"""Busqueda simple de un horizonte global para churn.

Se usan dos anios de historico anteriores a una fecha de corte T.
La base global B es la mediana de TODOS los intervalos historicos observados
entre compras consecutivas de los clientes.

Para cada alfa:
    H = alfa * B

Clasificacion:
- no_churn: compra dentro de (T, T + H]
- churn_provisional: no compra dentro de H
- churn_reactivado: churn provisional que compra antes de T + H + xB*B
- churn_persistente: churn provisional que no compra hasta T + H + xB*B
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
    p = argparse.ArgumentParser(description="Busqueda de horizonte global de churn")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--alpha-min", type=float, default=1.0)
    p.add_argument("--alpha-max", type=float, default=3.0)
    p.add_argument("--alpha-step", type=float, default=0.05)
    p.add_argument("--xb", type=float, default=2.0)
    return p.parse_args()


def alpha_grid(start: float, stop: float, step: float) -> np.ndarray:
    if start <= 0 or step <= 0 or stop < start:
        raise ValueError("Parametros de alfa invalidos")
    n = int(round((stop - start) / step))
    return np.round(start + np.arange(n + 1) * step, 10)


def main() -> None:
    args = parse_args()
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
    gaps = hist["gap_dias"].dropna()
    B = float(gaps.median())

    clientes_hist = hist["CODIGO_FAMILIA"].drop_duplicates().to_frame()
    primera_futura = (
        future.groupby("CODIGO_FAMILIA", sort=False)["DIM_PERIODO"]
        .min()
        .rename("primera_compra_futura")
        .reset_index()
    )
    clientes = clientes_hist.merge(primera_futura, on="CODIGO_FAMILIA", how="left")

    primera = clientes["primera_compra_futura"]
    tiene_futura = primera.notna()
    n_total = len(clientes)
    alphas = alpha_grid(args.alpha_min, args.alpha_max, args.alpha_step)

    resultados = []
    for alfa in alphas:
        H = float(alfa * B)
        limite_h = T + pd.Timedelta(days=int(np.ceil(H)))
        limite_check = T + pd.Timedelta(days=int(np.ceil(H + args.xb * B)))

        compra_en_h = tiene_futura & (primera <= limite_h)
        churn_provisional = ~compra_en_h
        churn_reactivado = churn_provisional & tiene_futura & (primera <= limite_check)
        churn_persistente = churn_provisional & ~churn_reactivado

        n_no = int(compra_en_h.sum())
        n_prov = int(churn_provisional.sum())
        n_reac = int(churn_reactivado.sum())
        n_pers = int(churn_persistente.sum())

        resultados.append({
            "alfa": float(alfa),
            "H_dias": H,
            "no_churn": n_no,
            "churn_provisional": n_prov,
            "churn_reactivado": n_reac,
            "churn_persistente": n_pers,
            "pct_no_churn": n_no / n_total * 100.0,
            "pct_churn_provisional": n_prov / n_total * 100.0,
            "pct_reactivacion": n_reac / n_prov * 100.0 if n_prov else np.nan,
            "pct_persistencia": n_pers / n_prov * 100.0 if n_prov else np.nan,
        })

    tabla = pd.DataFrame(resultados)
    csv_path = output_dir / "busqueda_horizonte_global.csv"
    log_path = output_dir / "busqueda_horizonte_global.log"
    tabla.to_csv(csv_path, index=False)

    resumen = (
        "\n=== CONFIGURACION ===\n"
        f"inicio_historico: {hist_start.date()}\n"
        f"fecha_corte_T: {T.date()}\n"
        f"fecha_max_dataset: {fecha_max.date()}\n"
        f"clientes_historicos: {n_total}\n"
        f"intervalos_historicos: {len(gaps)}\n"
        f"B_global_mediana_dias: {B:.2f}\n"
        f"xB_validacion: {args.xb:.2f}\n"
        f"alpha_min: {args.alpha_min:.2f}\n"
        f"alpha_max: {args.alpha_max:.2f}\n"
        f"alpha_step: {args.alpha_step:.2f}\n"
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
