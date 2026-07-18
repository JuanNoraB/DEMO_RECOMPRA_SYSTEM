"""Calibracion empirica de un horizonte dinamico para churn a nivel cliente.

Usa dos anios de historico anteriores a una fecha de corte T y el periodo
posterior disponible como seguimiento.

Para cada cliente con al menos un intervalo se calcula una base B (media o
mediana). Para cada alfa se define H_i = alfa * B_i. La comprobacion posterior
usa xB * B_i, donde xB es configurable.

No se filtran clientes al inicio por seguimiento futuro. Cuando una combinacion
H_i + xB*B_i excede el final del dataset, el caso se reporta como seguimiento
incompleto y no se cuenta falsamente como churn persistente.
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
DEFAULT_OUTPUT = REPO_ROOT / "data" / "churn" / "target_search"
COLS = ["CODIGO_FAMILIA", "DIM_PERIODO"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Busqueda de horizonte dinamico de churn")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--alpha-min", type=float, default=1.0)
    p.add_argument("--alpha-max", type=float, default=3.0)
    p.add_argument("--alpha-step", type=float, default=0.05)
    p.add_argument(
        "--xb",
        type=float,
        default=3.0,
        help="Numero de ciclos B usados para comprobar persistencia despues de H",
    )
    return p.parse_args()


def alpha_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0 or stop < start:
        raise ValueError("Rango de alfa invalido")
    if start <= 0:
        raise ValueError("alpha-min debe ser mayor que 0")
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

    if not input_path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    alphas = alpha_grid(args.alpha_min, args.alpha_max, args.alpha_step)

    print(f"[TARGET] Leyendo: {input_path}")
    df = pd.read_parquet(input_path, columns=COLS)
    df["DIM_PERIODO"] = pd.to_datetime(df["DIM_PERIODO"], errors="coerce")
    df = (
        df.dropna(subset=COLS)
        .drop_duplicates()
        .sort_values(COLS)
        .reset_index(drop=True)
    )

    fecha_min = df["DIM_PERIODO"].min()
    fecha_max = df["DIM_PERIODO"].max()

    # Dos anios de historico y aproximadamente el ultimo anio para seguimiento.
    T = fecha_max - pd.DateOffset(years=1)
    hist_start = T - pd.DateOffset(years=2) + pd.Timedelta(days=1)

    hist = df[df["DIM_PERIODO"].between(hist_start, T)].copy()
    future = df[df["DIM_PERIODO"] > T].copy()

    hist["gap_dias"] = (
        hist.groupby("CODIGO_FAMILIA", sort=False)["DIM_PERIODO"].diff().dt.days
    )

    perfil = (
        hist.groupby("CODIGO_FAMILIA", sort=False)
        .agg(
            n_fechas=("DIM_PERIODO", "size"),
            media=("gap_dias", "mean"),
            mediana=("gap_dias", "median"),
        )
        .reset_index()
    )
    perfil["n_intervalos"] = perfil["n_fechas"] - 1

    # Para calcular B se necesita al menos un intervalo historico.
    perfil = perfil.dropna(subset=["media", "mediana"]).copy()

    primera_futura = (
        future.groupby("CODIGO_FAMILIA", sort=False)["DIM_PERIODO"]
        .min()
        .rename("primera_compra_futura")
        .reset_index()
    )
    perfil = perfil.merge(primera_futura, on="CODIGO_FAMILIA", how="left")

    resultados: list[dict[str, float | int | str]] = []
    T_np = np.datetime64(T.to_datetime64())
    fecha_max_np = np.datetime64(fecha_max.to_datetime64())

    for base in ["mediana", "media"]:
        B = perfil[base].to_numpy(dtype=float)
        primera = perfil["primera_compra_futura"].to_numpy(dtype="datetime64[ns]")
        tiene_futura = ~pd.isna(perfil["primera_compra_futura"]).to_numpy()

        for alfa in alphas:
            H = alfa * B
            dias_h = np.ceil(H).astype(int)
            dias_check = np.ceil(H + args.xb * B).astype(int)

            limite_h = T_np + dias_h.astype("timedelta64[D]")
            limite_check = T_np + dias_check.astype("timedelta64[D]")

            compra_en_h = tiene_futura & (primera <= limite_h)

            # Si H termina despues del dataset, solo sabemos que no es churn si ya vimos compra.
            h_completo = limite_h <= fecha_max_np
            churn_provisional = h_completo & ~compra_en_h

            # Reactivacion se conoce si la primera compra ocurre antes del limite de comprobacion,
            # aunque el limite completo exceda el final del dataset.
            reactivado = churn_provisional & tiene_futura & (primera <= limite_check)
            check_completo = limite_check <= fecha_max_np
            persistente = churn_provisional & check_completo & ~reactivado
            seguimiento_incompleto = churn_provisional & ~check_completo & ~reactivado

            n_total = len(perfil)
            n_no_churn = int(compra_en_h.sum())
            n_prov = int(churn_provisional.sum())
            n_reac = int(reactivado.sum())
            n_pers = int(persistente.sum())
            n_incomp = int(seguimiento_incompleto.sum())
            n_eval = n_reac + n_pers

            resultados.append(
                {
                    "base": base,
                    "alfa": float(alfa),
                    "xB": float(args.xb),
                    "clientes_con_intervalos": n_total,
                    "no_churn": n_no_churn,
                    "churn_provisional": n_prov,
                    "reactivados_en_xB": n_reac,
                    "churn_persistente": n_pers,
                    "seguimiento_incompleto": n_incomp,
                    "churn_provisional_evaluable": n_eval,
                    "pct_reactivacion": (n_reac / n_eval * 100.0) if n_eval else np.nan,
                    "pct_persistencia": (n_pers / n_eval * 100.0) if n_eval else np.nan,
                    "H_mediano_dias": float(np.median(H)),
                    "H_p90_dias": float(np.quantile(H, 0.90)),
                }
            )

    tabla = pd.DataFrame(resultados)
    csv_path = output_dir / "busqueda_horizonte_dinamico.csv"
    log_path = output_dir / "busqueda_horizonte_dinamico.log"
    graph_path = output_dir / "persistencia_por_alfa.png"
    tabla.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    for base, grupo in tabla.groupby("base"):
        ax.plot(grupo["alfa"], grupo["pct_persistencia"], marker="o", label=base)
    ax.set_xlabel("Alfa")
    ax.set_ylabel("Persistencia de churn provisional evaluable (%)")
    ax.set_title(f"Persistencia del churn segun alfa y base temporal (check = {args.xb:g}B)")
    ax.grid(alpha=0.25)
    ax.legend(title="Base")
    fig.tight_layout()
    fig.savefig(graph_path, dpi=160)
    plt.close(fig)

    resumen = pd.DataFrame(
        {
            "metrica": [
                "fecha_min_dataset",
                "fecha_max_dataset",
                "inicio_historico",
                "fecha_corte_T",
                "dias_seguimiento_disponible",
                "clientes_con_intervalos",
                "alpha_min",
                "alpha_max",
                "alpha_step",
                "xB_validacion",
                "combinaciones_evaluadas",
            ],
            "valor": [
                str(fecha_min.date()),
                str(fecha_max.date()),
                str(hist_start.date()),
                str(T.date()),
                int((fecha_max - T).days),
                len(perfil),
                args.alpha_min,
                args.alpha_max,
                args.alpha_step,
                args.xb,
                len(tabla),
            ],
        }
    )

    output_text = (
        "\n=== CONFIGURACION DEL EXPERIMENTO ===\n"
        + resumen.to_string(index=False)
        + "\n\n=== RESULTADOS POR BASE Y ALFA ===\n"
        + tabla.to_string(index=False, float_format=lambda x: f"{x:,.2f}")
        + f"\n\nTiempo total: {time.perf_counter() - started:.2f} s\n"
    )

    print(output_text)
    log_path.write_text(output_text, encoding="utf-8")

    print(f"[TARGET] Tabla: {csv_path}")
    print(f"[TARGET] Grafico: {graph_path}")
    print(f"[TARGET] Log: {log_path}")


if __name__ == "__main__":
    main()
