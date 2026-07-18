"""Calibracion empirica de un horizonte dinamico para churn a nivel cliente.

Se usan dos anios de historico y el ultimo anio disponible como seguimiento.
Para cada cliente se calcula una base B (media o mediana de sus intervalos).
Para cada alfa: H_i = alfa * B_i.

Clasificacion:
- no_churn: compra dentro de (T, T + H_i]
- reactivado: no compra en H_i, pero compra antes de T + H_i + 2*B_i
- churn_persistente: no compra hasta T + H_i + 2*B_i

La poblacion elegible es comun para todas las combinaciones base/alfa: se exige
seguimiento completo incluso para alfa_max y para la mayor base individual
entre media y mediana. No se construye todavia el dataset final del modelo.
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
    p.add_argument("--alpha-max", type=float, default=2.0)
    p.add_argument("--alpha-step", type=float, default=0.05)
    return p.parse_args()


def alpha_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0 or stop < start:
        raise ValueError("Rango de alfa invalido")
    n = int(round((stop - start) / step))
    return np.round(start + np.arange(n + 1) * step, 10)


def main() -> None:
    args = parse_args()
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
    df = df.dropna(subset=COLS).drop_duplicates().sort_values(COLS).reset_index(drop=True)

    fecha_min = df["DIM_PERIODO"].min()
    fecha_max = df["DIM_PERIODO"].max()

    # Reservamos el ultimo anio completo para comprobar H + 2B.
    T = fecha_max - pd.DateOffset(years=1)
    hist_start = T - pd.DateOffset(years=2) + pd.Timedelta(days=1)

    hist = df[df["DIM_PERIODO"].between(hist_start, T)].copy()
    future = df[df["DIM_PERIODO"] > T].copy()

    hist["gap_dias"] = hist.groupby("CODIGO_FAMILIA", sort=False)["DIM_PERIODO"].diff().dt.days
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

    primera_futura = (
        future.groupby("CODIGO_FAMILIA", sort=False)["DIM_PERIODO"]
        .min()
        .rename("primera_compra_futura")
        .reset_index()
    )
    perfil = perfil.merge(primera_futura, on="CODIGO_FAMILIA", how="left")
    perfil = perfil.dropna(subset=["media", "mediana"]).copy()

    # Misma poblacion para media/mediana y todos los alfas.
    perfil["base_max"] = perfil[["media", "mediana"]].max(axis=1)
    dias_seguimiento = (fecha_max - T).days
    perfil["seguimiento_max_requerido"] = (args.alpha_max + 2.0) * perfil["base_max"]
    elegibles = perfil[perfil["seguimiento_max_requerido"] <= dias_seguimiento].copy()

    resultados: list[dict[str, float | int | str]] = []
    for base in ["mediana", "media"]:
        B = elegibles[base].to_numpy(dtype=float)
        primera = elegibles["primera_compra_futura"].to_numpy(dtype="datetime64[ns]")
        tiene_futura = ~pd.isna(elegibles["primera_compra_futura"]).to_numpy()

        for alfa in alphas:
            H = alfa * B
            limite_h = np.array(T.to_datetime64()) + H.astype("timedelta64[D]")
            limite_check = np.array(T.to_datetime64()) + (H + 2.0 * B).astype("timedelta64[D]")

            compra_en_h = tiene_futura & (primera <= limite_h)
            churn_provisional = ~compra_en_h
            reactivado = churn_provisional & tiene_futura & (primera <= limite_check)
            persistente = churn_provisional & ~reactivado

            n_total = len(elegibles)
            n_prov = int(churn_provisional.sum())
            n_reac = int(reactivado.sum())
            n_pers = int(persistente.sum())

            resultados.append(
                {
                    "base": base,
                    "alfa": float(alfa),
                    "clientes_elegibles": n_total,
                    "no_churn": int(compra_en_h.sum()),
                    "churn_provisional": n_prov,
                    "reactivados_en_2B": n_reac,
                    "churn_persistente": n_pers,
                    "pct_reactivacion": (n_reac / n_prov * 100.0) if n_prov else np.nan,
                    "pct_persistencia": (n_pers / n_prov * 100.0) if n_prov else np.nan,
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
    ax.set_ylabel("Persistencia de churn provisional (%)")
    ax.set_title("Persistencia del churn segun alfa y base temporal")
    ax.grid(alpha=0.25)
    ax.legend(title="Base")
    fig.tight_layout()
    fig.savefig(graph_path, dpi=160)
    plt.close(fig)

    resumen = pd.DataFrame(
        {
            "metrica": [
                "fecha_min_dataset", "fecha_max_dataset", "inicio_historico", "fecha_corte_T",
                "dias_seguimiento", "clientes_con_intervalos", "clientes_elegibles_comunes",
                "alpha_min", "alpha_max", "alpha_step", "combinaciones_evaluadas",
            ],
            "valor": [
                str(fecha_min.date()), str(fecha_max.date()), str(hist_start.date()), str(T.date()),
                dias_seguimiento, len(perfil), len(elegibles), args.alpha_min, args.alpha_max,
                args.alpha_step, len(tabla),
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
