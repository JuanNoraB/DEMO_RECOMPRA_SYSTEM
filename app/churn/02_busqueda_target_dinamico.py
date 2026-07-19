"""Busqueda de horizonte hibrido para definir churn en entorno no contractual.

H_i = alfa * B + gamma * mediana_i
B = media de las medianas individuales de intervalos.
La validacion posterior usa xB * B dias.
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
COLS = ["IDENTIFICACION", "DIM_PERIODO"]


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
    p.add_argument("--min-compras", type=int, default=4)
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
    if args.min_compras < 2:
        raise ValueError("--min-compras debe ser al menos 2")

    started = time.perf_counter()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[TARGET] Leyendo: {input_path}")

    df = pd.read_parquet(input_path, columns=COLS)
    df["DIM_PERIODO"] = pd.to_datetime(df["DIM_PERIODO"], errors="coerce")
    df = (
        df.dropna(subset=COLS)
        .drop_duplicates(subset=["IDENTIFICACION", "DIM_PERIODO"])
        .sort_values(["IDENTIFICACION", "DIM_PERIODO"])
        .reset_index(drop=True)
    )

    fecha_max = df["DIM_PERIODO"].max()
    T = fecha_max - pd.DateOffset(years=1)
    hist_start = T - pd.DateOffset(years=2) + pd.Timedelta(days=1)

    hist = df[df["DIM_PERIODO"].between(hist_start, T)].copy()
    future = df[df["DIM_PERIODO"] > T].copy()

    clientes_hist_antes = hist["IDENTIFICACION"].nunique()
    conteo_compras = hist.groupby("IDENTIFICACION", sort=False).size()
    ids_elegibles = conteo_compras[conteo_compras >= args.min_compras].index
    hist = hist[hist["IDENTIFICACION"].isin(ids_elegibles)].copy()

    hist["gap_dias"] = (
        hist.groupby("IDENTIFICACION", sort=False)["DIM_PERIODO"].diff().dt.days
    )

    perfil = (
        hist.groupby("IDENTIFICACION", sort=False)
        .agg(
            n_compras_historicas=("DIM_PERIODO", "size"),
            mediana_cliente=("gap_dias", "median"),
        )
        .reset_index()
    )
    perfil = (
        perfil[perfil["n_compras_historicas"] >= args.min_compras]
        .dropna(subset=["mediana_cliente"])
        .reset_index(drop=True)
    )

    # Referencia global: media de las medianas individuales.
    B = float(perfil["mediana_cliente"].mean())

    primera_futura = (
        future.groupby("IDENTIFICACION", sort=False)["DIM_PERIODO"]
        .min()
        .rename("primera_compra_futura")
        .reset_index()
    )
    clientes = perfil.merge(primera_futura, on="IDENTIFICACION", how="left")
    primera = clientes["primera_compra_futura"]
    tiene_futura = primera.notna()
    n_total = len(clientes)

    alphas = parameter_grid(args.alpha_min, args.alpha_max, args.alpha_step, "alfa")
    gammas = parameter_grid(args.gamma_min, args.gamma_max, args.gamma_step, "gamma")
    dias_validacion = int(np.ceil(args.xb * B))

    resultados: list[dict[str, float | int]] = []

    for alfa in alphas:
        for gamma in gammas:
            H = alfa * B + gamma * clientes["mediana_cliente"]
            H = H.clip(lower=1.0)
            dias_h = np.ceil(H).astype(int)

            limite_h = T + pd.to_timedelta(dias_h, unit="D")
            limite_check = limite_h + pd.to_timedelta(dias_validacion, unit="D")

            compra_en_h = tiene_futura & (primera <= limite_h)
            c_provisional = ~compra_en_h
            c_reactivado = c_provisional & tiene_futura & (primera <= limite_check)
            c_persistente = c_provisional & ~c_reactivado

            n_prov = int(c_provisional.sum())
            n_reac = int(c_reactivado.sum())
            n_pers = int(c_persistente.sum())

            # Porcentaje de clientes cuya ventana completa H_i + xB*B supera Dmax.
            pct_dmax = float((limite_check > fecha_max).mean() * 100.0)

            resultados.append(
                {
                    "a": float(alfa),
                    "g": float(gamma),
                    "B": B,
                    "H50": float(H.quantile(0.50)),
                    "H90": float(H.quantile(0.90)),
                    "C_prov": n_prov,
                    "C_reac": n_reac,
                    "C_pers": n_pers,
                    "C_prov%": (n_prov / n_total * 100.0) if n_total else np.nan,
                    "React%": (n_reac / n_prov * 100.0) if n_prov else np.nan,
                    "Pers%": (n_pers / n_prov * 100.0) if n_prov else np.nan,
                    "Dmax%": pct_dmax,
                }
            )

    tabla = pd.DataFrame(resultados)
    csv_path = output_dir / "busqueda_horizonte_hibrido.csv"
    log_path = output_dir / "busqueda_horizonte_hibrido.log"
    tabla.to_csv(csv_path, index=False)

    resumen = (
        "\n=== CONFIGURACION ===\n"
        f"hist: {hist_start.date()} -> {T.date()}\n"
        f"Dmax: {fecha_max.date()}\n"
        "unidad: IDENTIFICACION\n"
        f"clientes_pre: {clientes_hist_antes}\n"
        f"min_compras: {args.min_compras}\n"
        f"clientes_elegibles: {n_total}\n"
        f"B_media_medianas_dias: {B:.2f}\n"
        f"xB: {args.xb:.2f}\n"
        f"V_dias: {dias_validacion}\n"
        f"comb: {len(alphas) * len(gammas)}\n"
        "\n=== RESULTADOS ===\n"
        + tabla.to_string(index=False, float_format=lambda x: f"{x:.2f}")
        + f"\n\nTiempo total: {time.perf_counter() - started:.2f} s\n"
    )

    print(resumen)
    log_path.write_text(resumen, encoding="utf-8")
    print(f"[TARGET] Tabla: {csv_path}")
    print(f"[TARGET] Log: {log_path}")


if __name__ == "__main__":
    main()
