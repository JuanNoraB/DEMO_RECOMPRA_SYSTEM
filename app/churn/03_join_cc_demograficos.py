from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CC = REPO_ROOT / "data" / "raw" / "CC.csv"
DEFAULT_HIST = REPO_ROOT / "data" / "raw" / "HISTORICO_300K_FULL.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "historico_con_demograficos.parquet"
MIN_COMPRAS = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Join demografico para historico elegible de churn")
    p.add_argument("--cc", type=Path, default=DEFAULT_CC)
    p.add_argument("--historico", type=Path, default=DEFAULT_HIST)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--min-compras", type=int, default=MIN_COMPRAS)
    return p.parse_args()


def cargar_cc(path: Path) -> pd.DataFrame:
    print(f"\n[1/6] Leyendo {path} ...")
    t0 = time.time()
    cc = pd.read_csv(path, dtype=str, engine="python", on_bad_lines="warn")
    print(f"   filas leidas: {len(cc):,} ({time.time() - t0:.1f}s)")
    return cc


def limpiar_identificacion(cc: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/6] Limpiando IDENTIFICACION ...")
    n0 = len(cc)
    cc = cc.dropna(subset=["IDENTIFICACION"]).copy()
    cc["IDENTIFICACION"] = cc["IDENTIFICACION"].astype("string").str.strip()
    cc = cc[cc["IDENTIFICACION"].notna() & (cc["IDENTIFICACION"] != "")].copy()
    print(f"   filas con IDENTIFICACION nula/vacia descartadas: {n0 - len(cc):,}")

    es_numerica = cc["IDENTIFICACION"].str.fullmatch(r"\d+", na=False)
    n_no_numericas = int((~es_numerica).sum())
    print(f"   identificaciones no numericas conservadas: {n_no_numericas:,}")
    print(f"   filas duplicadas por IDENTIFICACION: {cc['IDENTIFICACION'].duplicated().sum():,}")
    return cc


def calcular_edad(cc: pd.DataFrame, referencia: pd.Timestamp) -> pd.DataFrame:
    print(f"\n[3/6] Calculando EDAD (ref. {referencia.date()}) ...")
    ext = cc["FECHA_NACIMIENTO"].str.extract(r"^(\d{2})-([A-Za-z]{3})-(\d{2})$")
    ext.columns = ["dd", "mon", "yy"]
    yy = pd.to_numeric(ext["yy"], errors="coerce")
    anio = np.where(yy <= referencia.year % 100, 2000 + yy, 1900 + yy)
    fecha_txt = (
        pd.Series(anio, index=cc.index).astype("Int64").astype(str)
        + "-" + ext["mon"] + "-" + ext["dd"]
    )
    fecha_nac = pd.to_datetime(fecha_txt, format="%Y-%b-%d", errors="coerce")
    edad = ((referencia - fecha_nac).dt.days / 365.25).astype("float64")
    edad = edad.where(edad.notna() & (edad >= 0) & (edad <= 110))

    cc["FECHA_NACIMIENTO_PARSEADA"] = fecha_nac
    cc["EDAD"] = edad
    print(f"   EDAD nula: {cc['EDAD'].isna().sum():,} de {len(cc):,}")
    if cc["EDAD"].notna().any():
        print(f"   media EDAD conocida: {cc['EDAD'].mean():.2f}")
        print(f"   mediana EDAD conocida: {cc['EDAD'].median():.2f}")
    return cc


def revisar_sexo(cc: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/6] Revisando SEXO ...")
    cc["SEXO"] = cc["SEXO"].astype("string").str.strip().str.upper()
    cc["SEXO"] = cc["SEXO"].where(cc["SEXO"].isin(["M", "F"]))
    print(f"   SEXO nulo/invalido: {cc['SEXO'].isna().sum():,}")
    print(cc["SEXO"].value_counts(dropna=False).to_string())
    return cc


def deduplicar_cc(cc: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/6] Deduplicando CC por IDENTIFICACION ...")
    n_ids = cc["IDENTIFICACION"].nunique()
    cc = cc.assign(
        _tiene_fecha=cc["FECHA_NACIMIENTO_PARSEADA"].notna(),
        _tiene_sexo=cc["SEXO"].notna(),
    ).sort_values(
        ["IDENTIFICACION", "_tiene_fecha", "_tiene_sexo"],
        ascending=[True, False, False],
    )
    out = cc.drop_duplicates("IDENTIFICACION", keep="first")[["IDENTIFICACION", "EDAD", "SEXO"]].copy()
    print(f"   ids unicos: {n_ids:,} -> filas tras dedup: {len(out):,}")
    assert len(out) == n_ids
    return out


def hacer_join(hist_path: Path, cc: pd.DataFrame, output: Path, min_compras: int) -> None:
    print(f"\n[6/6] Leyendo, filtrando elegibilidad y haciendo LEFT JOIN con {hist_path.name} ...")
    hist = pd.read_parquet(hist_path)
    hist["DIM_PERIODO"] = pd.to_datetime(hist["DIM_PERIODO"], errors="coerce")
    hist = hist.dropna(subset=["IDENTIFICACION", "DIM_PERIODO"]).copy()
    hist["IDENTIFICACION"] = hist["IDENTIFICACION"].astype("string").str.strip()

    fecha_max = hist["DIM_PERIODO"].max()
    T = fecha_max - pd.DateOffset(years=1)
    hist_start = T - pd.DateOffset(years=2) + pd.Timedelta(days=1)

    print("\n   === PERIODO UTILIZADO PARA CHURN ===")
    print(f"   fecha maxima dataset : {fecha_max.date()}")
    print(f"   fecha de corte T     : {T.date()}")
    print(f"   inicio historico     : {hist_start.date()}")
    print(f"   fin historico        : {T.date()}")

    hist = hist[hist["DIM_PERIODO"].between(hist_start, T)].copy()
    filas_periodo = len(hist)
    clientes_periodo = hist["IDENTIFICACION"].nunique()

    # Misma logica del script 02 para definir elegibilidad:
    # una compra por IDENTIFICACION y fecha; luego minimo de 4 compras.
    hist_fechas = hist[["IDENTIFICACION", "DIM_PERIODO"]].drop_duplicates()
    duplicados_cliente_fecha = filas_periodo - len(hist_fechas)
    conteo_compras = hist_fechas.groupby("IDENTIFICACION", sort=False).size()
    ids_elegibles = conteo_compras[conteo_compras >= min_compras].index

    # Conservamos todas las filas transaccionales de los clientes elegibles.
    # No deduplicamos el historico final porque se necesitan ventas y subcategorias
    # para calcular posteriormente las caracteristicas monetarias y de diversidad.
    hist = hist[hist["IDENTIFICACION"].isin(ids_elegibles)].copy()

    print("\n   === FILTRO DE ELEGIBILIDAD ===")
    print(f"   filas originales en periodo            : {filas_periodo:,}")
    print(f"   clientes originales en periodo         : {clientes_periodo:,}")
    print(f"   filas repetidas IDENTIFICACION-PERIODO : {duplicados_cliente_fecha:,}")
    print(f"   compras unicas cliente-fecha           : {len(hist_fechas):,}")
    print(f"   minimo compras distintas requerido     : {min_compras}")
    print(f"   clientes elegibles                     : {len(ids_elegibles):,}")
    print(f"   clientes excluidos                     : {clientes_periodo - len(ids_elegibles):,}")
    print(f"   filas transaccionales elegibles        : {len(hist):,}")

    cc = cc.copy()
    cc["IDENTIFICACION"] = cc["IDENTIFICACION"].astype("string").str.strip()

    ids_hist = set(hist["IDENTIFICACION"].dropna().unique())
    ids_cc = set(cc["IDENTIFICACION"].dropna().unique())

    print("\n   === COMPROBACION IDENTIFICACION ===")
    print(f"   historico IDs comienzan con 0: {hist['IDENTIFICACION'].str.startswith('0', na=False).any()}")
    print(f"   CC IDs comienzan con 0       : {cc['IDENTIFICACION'].str.startswith('0', na=False).any()}")

    merged = hist.merge(cc, on="IDENTIFICACION", how="left", validate="many_to_one")
    assert len(merged) == len(hist), "El LEFT JOIN no debe cambiar el numero de filas"

    match = ids_hist & ids_cc
    no_match = ids_hist - ids_cc

    print("\n   === RESULTADO DEL LEFT JOIN EN CLIENTES ELEGIBLES ===")
    print(f"   clientes con match : {len(match):,} ({len(match) / len(ids_hist) * 100:.2f}%)")
    print(f"   clientes sin match : {len(no_match):,} ({len(no_match) / len(ids_hist) * 100:.2f}%)")

    dem = merged[["IDENTIFICACION", "EDAD", "SEXO"]].drop_duplicates("IDENTIFICACION")
    print(f"   clientes sin EDAD   : {dem['EDAD'].isna().sum():,} ({dem['EDAD'].isna().mean() * 100:.2f}%)")
    print(f"   clientes sin SEXO   : {dem['SEXO'].isna().sum():,} ({dem['SEXO'].isna().mean() * 100:.2f}%)")
    if dem["EDAD"].notna().any():
        print(f"   media EDAD disponible   : {dem['EDAD'].mean():.2f}")
        print(f"   mediana EDAD disponible : {dem['EDAD'].median():.2f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output, index=False)
    print(f"\n   parquet guardado en: {output}")
    print(f"   tamanio: {output.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    args = parse_args()
    if args.min_compras < 1:
        raise ValueError("--min-compras debe ser al menos 1")

    fechas = pd.read_parquet(args.historico, columns=["DIM_PERIODO"])
    fechas["DIM_PERIODO"] = pd.to_datetime(fechas["DIM_PERIODO"], errors="coerce")
    T = fechas["DIM_PERIODO"].max() - pd.DateOffset(years=1)
    del fechas

    cc = cargar_cc(args.cc)
    cc = limpiar_identificacion(cc)
    cc = calcular_edad(cc, T)
    cc = revisar_sexo(cc)
    cc = deduplicar_cc(cc)
    hacer_join(args.historico, cc, args.output, args.min_compras)


if __name__ == "__main__":
    main()
