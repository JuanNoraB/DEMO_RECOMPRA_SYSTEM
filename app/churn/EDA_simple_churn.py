"""EDA simple de intervalos de compra para churn a nivel cliente.

Lee solo IDENTIFICACION y DIM_PERIODO, conserva los dos anios mas
recientes del historico y genera:
1. Una tabla estadistica general.
2. Una tabla por terciles excluyentes.
3. Tres graficos de distribucion.
4. Correlacion de Spearman entre cuatro variables de ciclo.
5. PCA de dos componentes sobre esas cuatro variables.

No construye el target ni entrena modelos.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "HISTORICO_300K_FULL.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "churn" / "logs"
REQUIRED_COLUMNS = ["IDENTIFICACION", "DIM_PERIODO"]
PCA_COLUMNS = [
    "n_intervalos",
    "media_intervalos",
    "mediana_intervalos",
    "desviacion_intervalos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA simple de churn con pandas")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def summarize(series: pd.Series, variable: str) -> dict[str, float | int | str]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "variable": variable,
        "n": int(values.size),
        "media": float(values.mean()),
        "mediana": float(values.median()),
        "desviacion_std": float(values.std(ddof=0)),
        "varianza": float(values.var(ddof=0)),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p90": float(values.quantile(0.90)),
    }


def summarize_terciles(series: pd.Series, variable: str) -> list[dict[str, float | int | str]]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    p33 = float(values.quantile(1 / 3))
    p66 = float(values.quantile(2 / 3))

    grupos = [
        (f"P0-P33 (<= {p33:.2f})", values <= p33),
        (f"P33-P66 ({p33:.2f}, {p66:.2f}]", (values > p33) & (values <= p66)),
        (f"P66-P100 (> {p66:.2f})", values > p66),
    ]

    rows = []
    for tramo, mask in grupos:
        group = values.loc[mask]
        rows.append(
            {
                "variable": variable,
                "tramo": tramo,
                "clientes": int(group.size),
                "media": float(group.mean()),
                "desviacion_std": float(group.std(ddof=0)),
                "varianza": float(group.var(ddof=0)),
                "cv": float(group.std(ddof=0) / group.mean()) if group.mean() != 0 else np.nan,
            }
        )
    return rows


def save_distribution_plot(
    series: pd.Series,
    title: str,
    xlabel: str,
    output_path: Path,
    bin_width: int = 5,
) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    upper = float(values.quantile(0.98))
    visible = values[values <= upper]

    max_value = max(bin_width, int(np.ceil(upper / bin_width) * bin_width))
    bins = np.arange(0, max_value + bin_width, bin_width)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(visible, bins=bins)
    ax.set_title(f"{title} (hasta P98 = {upper:.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Numero de clientes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_pca(perfil: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = perfil[["IDENTIFICACION"] + PCA_COLUMNS].dropna().copy()

    transformed = data[PCA_COLUMNS].copy()
    for column in PCA_COLUMNS:
        transformed[column] = np.log1p(transformed[column].clip(lower=0))

    scaled = StandardScaler().fit_transform(transformed)
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)

    pca_scores = pd.DataFrame(
        {
            "IDENTIFICACION": data["IDENTIFICACION"].to_numpy(),
            "PC1": components[:, 0],
            "PC2": components[:, 1],
        }
    )

    pca_summary = pd.DataFrame(
        {
            "componente": ["PC1", "PC2", "PC1+PC2"],
            "varianza_explicada": [
                float(pca.explained_variance_ratio_[0]),
                float(pca.explained_variance_ratio_[1]),
                float(pca.explained_variance_ratio_.sum()),
            ],
        }
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=PCA_COLUMNS,
        columns=["PC1", "PC2"],
    ).reset_index(names="variable")

    sample = pca_scores.sample(n=min(30000, len(pca_scores)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(sample["PC1"], sample["PC2"], s=7, alpha=0.25)
    ax.set_title(
        "PCA de perfiles de compra\n"
        f"PC1 + PC2 explican {pca.explained_variance_ratio_.sum() * 100:.2f}%"
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)")
    ax.grid(alpha=0.20)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_perfiles_clientes.png", dpi=160)
    plt.close(fig)

    pca_scores.to_csv(output_dir / "pca_scores_clientes.csv", index=False)
    loadings.to_csv(output_dir / "pca_cargas.csv", index=False)
    pca_summary.to_csv(output_dir / "pca_varianza_explicada.csv", index=False)

    return pca_summary, loadings


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print(f"[EDA] Leyendo: {input_path}")
    df = pd.read_parquet(input_path, columns=REQUIRED_COLUMNS)
    df["DIM_PERIODO"] = pd.to_datetime(df["DIM_PERIODO"], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS)

    fecha_max = df["DIM_PERIODO"].max()
    fecha_inicio = fecha_max - pd.DateOffset(years=2) + pd.Timedelta(days=1)
    df = df[df["DIM_PERIODO"].between(fecha_inicio, fecha_max)].copy()

    compras = (
        df[REQUIRED_COLUMNS]
        .drop_duplicates()
        .sort_values(["IDENTIFICACION", "DIM_PERIODO"])
        .reset_index(drop=True)
    )

    compras["intervalo_dias"] = (
        compras.groupby("IDENTIFICACION", sort=False)["DIM_PERIODO"]
        .diff()
        .dt.days
    )

    perfil = (
        compras.groupby("IDENTIFICACION", sort=False)
        .agg(
            n_fechas_compra=("DIM_PERIODO", "size"),
            primera_compra=("DIM_PERIODO", "min"),
            ultima_compra=("DIM_PERIODO", "max"),
            media_intervalos=("intervalo_dias", "mean"),
            mediana_intervalos=("intervalo_dias", "median"),
            desviacion_intervalos=("intervalo_dias", lambda x: x.std(ddof=0)),
            varianza_intervalos=("intervalo_dias", lambda x: x.var(ddof=0)),
        )
        .reset_index()
    )

    perfil["n_intervalos"] = perfil["n_fechas_compra"] - 1
    perfil["duracion_relacion_dias"] = (
        perfil["ultima_compra"] - perfil["primera_compra"]
    ).dt.days

    variables = [
        "n_fechas_compra",
        "n_intervalos",
        "duracion_relacion_dias",
        "media_intervalos",
        "mediana_intervalos",
        "desviacion_intervalos",
        "varianza_intervalos",
    ]
    tabla = pd.DataFrame([summarize(perfil[col], col) for col in variables])

    tabla_terciles = pd.DataFrame(
        [
            row
            for variable in PCA_COLUMNS
            for row in summarize_terciles(perfil[variable], variable)
        ]
    )

    correlacion_spearman = perfil[PCA_COLUMNS].corr(method="spearman")
    pca_summary, pca_loadings = run_pca(perfil, output_dir)

    resumen_general = pd.DataFrame(
        {
            "metrica": [
                "fecha_inicio",
                "fecha_fin",
                "filas_transaccionales",
                "clientes",
                "fechas_cliente_unicas",
                "clientes_con_intervalos",
            ],
            "valor": [
                str(fecha_inicio.date()),
                str(fecha_max.date()),
                int(len(df)),
                int(perfil["IDENTIFICACION"].nunique()),
                int(len(compras)),
                int((perfil["n_intervalos"] >= 1).sum()),
            ],
        }
    )

    tabla.to_csv(output_dir / "eda_simple_churn.csv", index=False)
    tabla_terciles.to_csv(output_dir / "eda_simple_churn_terciles.csv", index=False)
    correlacion_spearman.to_csv(output_dir / "correlacion_spearman.csv")

    graficos = [
        (
            "n_intervalos",
            "Distribucion del numero de intervalos por cliente",
            "Numero de intervalos",
            output_dir / "distribucion_n_intervalos.png",
        ),
        (
            "mediana_intervalos",
            "Distribucion de la mediana de intervalos por cliente",
            "Mediana de intervalos (dias)",
            output_dir / "distribucion_mediana_intervalos.png",
        ),
        (
            "desviacion_intervalos",
            "Distribucion de la desviacion de intervalos por cliente",
            "Desviacion de intervalos (dias)",
            output_dir / "distribucion_desviacion_intervalos.png",
        ),
    ]
    for variable, title, xlabel, path in graficos:
        save_distribution_plot(perfil[variable], title, xlabel, path)

    output_text = (
        "\n=== RESUMEN GENERAL ===\n"
        + resumen_general.to_string(index=False)
        + "\n\n=== TABLA ESTADISTICA ===\n"
        + tabla.to_string(index=False, float_format=lambda x: f"{x:,.2f}")
        + "\n\n=== TABLA POR TERCILES EXCLUYENTES ===\n"
        + tabla_terciles.to_string(index=False, float_format=lambda x: f"{x:,.3f}")
        + "\n\n=== CORRELACION DE SPEARMAN ===\n"
        + correlacion_spearman.to_string(float_format=lambda x: f"{x:.3f}")
        + "\n\n=== VARIANZA EXPLICADA POR PCA ===\n"
        + pca_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}")
        + "\n\n=== CARGAS DEL PCA ===\n"
        + pca_loadings.to_string(index=False, float_format=lambda x: f"{x:.4f}")
        + f"\n\nTiempo total: {time.perf_counter() - started:.2f} s\n"
    )

    print(output_text)
    (output_dir / "eda_simple_churn.log").write_text(output_text, encoding="utf-8")

    print(f"[EDA] Resultados guardados en: {output_dir}")
    print(f"[EDA] PCA: {output_dir / 'pca_perfiles_clientes.png'}")
    print(f"[EDA] Spearman: {output_dir / 'correlacion_spearman.csv'}")


if __name__ == "__main__":
    main()