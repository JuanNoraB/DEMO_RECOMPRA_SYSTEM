"""Join de datos demograficos (CC.csv) con el historico de compras.

Pasos:
1. Lee CC.csv (IDENTIFICACION, FECHA_NACIMIENTO, SEXO), tolera lineas corruptas.
2. Calcula EDAD a partir de FECHA_NACIMIENTO (formato DD-MON-YY) usando como
   fecha de referencia 2026-01-01. Infiere el siglo del anio de 2 digitos:
   yy <= 25 -> 20yy, yy > 25 -> 19yy (ya que la referencia es 2026).
3. Imputa EDAD faltante con la media de las edades conocidas.
4. Deduplica IDENTIFICACION (puede haber mas de una fila por cliente).
5. Left join del historico (HISTORICO_300K_FULL.parquet) con CC por
   IDENTIFICACION, sin perder filas del historico.
6. Reporta cuantos clientes matchearon / no matchearon y cuantos nulos quedan.

Ejecutar:
    python app/churn/03_join_cc_demograficos.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CC = REPO_ROOT / "data" / "raw" / "CC.csv"
DEFAULT_HIST = REPO_ROOT / "data" / "raw" / "HISTORICO_300K_FULL.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "churn" / "join_demograficos"
FECHA_REFERENCIA = pd.Timestamp("2026-01-01")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Join CC.csv con historico de compras")
    p.add_argument("--cc", type=Path, default=DEFAULT_CC)
    p.add_argument("--historico", type=Path, default=DEFAULT_HIST)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def cargar_cc(path: Path) -> pd.DataFrame:
    print(f"\n[1/6] Leyendo {path} ...")
    t0 = time.time()
    cc = pd.read_csv(path, dtype=str, engine="python", on_bad_lines="warn")
    print(f"   filas leidas: {len(cc):,}  ({time.time() - t0:.1f}s)")
    return cc


def limpiar_identificacion(cc: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/6] Limpiando IDENTIFICACION ...")
    n0 = len(cc)
    cc = cc.dropna(subset=["IDENTIFICACION"]).copy()
    cc["IDENTIFICACION"] = cc["IDENTIFICACION"].str.strip()
    print(f"   filas con IDENTIFICACION nula descartadas: {n0 - len(cc):,}")

    es_numerica = cc["IDENTIFICACION"].str.fullmatch(r"\d+")
    n_no_numerica = int((~es_numerica).sum())
    print(f"   IDENTIFICACION no numerica (no puede matchear con historico int64): {n_no_numerica:,}")
    cc = cc[es_numerica].copy()
    cc["ID_INT"] = cc["IDENTIFICACION"].astype("int64")

    n_dup = int(cc["ID_INT"].duplicated().sum())
    print(f"   IDENTIFICACION duplicadas (filas extra a colapsar): {n_dup:,}")
    return cc


def calcular_edad(cc: pd.DataFrame) -> pd.DataFrame:
    print("\n[3/6] Calculando EDAD desde FECHA_NACIMIENTO (ref. 2026-01-01) ...")
    ext = cc["FECHA_NACIMIENTO"].str.extract(r"^(\d{2})-([A-Za-z]{3})-(\d{2})$")
    ext.columns = ["dd", "mon", "yy"]
    n_formato_raro = int(cc["FECHA_NACIMIENTO"].notna().sum() - ext.dropna().shape[0])
    if n_formato_raro:
        print(f"   fechas con formato inesperado (quedan como nulas): {n_formato_raro:,}")

    yy = pd.to_numeric(ext["yy"], errors="coerce")
    siglo = np.where(yy <= 25, 2000, 1900)
    anio_completo = siglo + yy
    fecha_str = anio_completo.astype("Int64").astype(str) + "-" + ext["mon"] + "-" + ext["dd"]
    fecha_nac = pd.to_datetime(fecha_str, format="%Y-%b-%d", errors="coerce")

    edad = ((FECHA_REFERENCIA - fecha_nac).dt.days / 365.25).astype("float64")
    edad = edad.where(edad.notna() & (edad >= 0))
    cc["EDAD"] = edad

    n_nulos = int(cc["EDAD"].isna().sum())
    print(f"   EDAD nula (fecha faltante o invalida): {n_nulos:,} de {len(cc):,} ({n_nulos / len(cc) * 100:.2f}%)")

    media_edad = float(cc["EDAD"].mean())
    print(f"   media de EDAD (sobre no-nulos): {media_edad:.2f} anios")
    cc["EDAD"] = cc["EDAD"].fillna(media_edad)
    cc["EDAD_IMPUTADA"] = edad.isna()
    return cc


def revisar_sexo(cc: pd.DataFrame) -> pd.DataFrame:
    print("\n[4/6] Revisando SEXO ...")
    n_nulos = int(cc["SEXO"].isna().sum())
    print(f"   SEXO nulo: {n_nulos:,}")
    print(f"   distribucion SEXO:\n{cc['SEXO'].value_counts(dropna=False).to_string()}")
    if n_nulos:
        moda = cc["SEXO"].mode(dropna=True).iloc[0]
        print(f"   imputando {n_nulos:,} nulos de SEXO con la moda ('{moda}') (son muy pocos)")
        cc["SEXO"] = cc["SEXO"].fillna(moda)
    return cc


def deduplicar(cc: pd.DataFrame) -> pd.DataFrame:
    print("\n[5/6] Deduplicando por ID_INT ...")
    n0 = cc["ID_INT"].nunique()
    cc_sorted = cc.sort_values(by=["EDAD_IMPUTADA"], ascending=True)
    cc_dedup = cc_sorted.drop_duplicates(subset="ID_INT", keep="first")
    print(f"   ids unicos: {n0:,} -> filas tras dedup: {len(cc_dedup):,} (debe coincidir)")
    return cc_dedup[["ID_INT", "EDAD", "SEXO"]].rename(columns={"ID_INT": "IDENTIFICACION"})


def hacer_join(hist_path: Path, cc_dedup: pd.DataFrame, output_dir: Path) -> None:
    print(f"\n[6/6] Left join con {hist_path.name} ...")
    hist = pd.read_parquet(hist_path)
    n_filas_hist = len(hist)
    ids_hist = hist["IDENTIFICACION"].unique()
    print(f"   filas historico: {n_filas_hist:,}  |  clientes unicos historico: {len(ids_hist):,}")

    merged = hist.merge(cc_dedup, on="IDENTIFICACION", how="left")
    assert len(merged) == n_filas_hist, "El left join no debe cambiar el numero de filas"

    ids_hist_set = set(ids_hist)
    ids_cc_set = set(cc_dedup["IDENTIFICACION"])
    ids_match = ids_hist_set & ids_cc_set
    ids_no_match = ids_hist_set - ids_cc_set

    print(f"\n   === RESULTADO DEL JOIN ===")
    print(f"   clientes historico que matchearon con CC   : {len(ids_match):,} ({len(ids_match)/len(ids_hist_set)*100:.2f}%)")
    print(f"   clientes historico que NO matchearon con CC: {len(ids_no_match):,} ({len(ids_no_match)/len(ids_hist_set)*100:.2f}%)")

    n_nulos_edad = int(merged["EDAD"].isna().sum())
    n_nulos_sexo = int(merged["SEXO"].isna().sum())
    print(f"\n   filas del historico con EDAD nula tras el join: {n_nulos_edad:,} ({n_nulos_edad/n_filas_hist*100:.2f}%)")
    print(f"   filas del historico con SEXO nula tras el join: {n_nulos_sexo:,} ({n_nulos_sexo/n_filas_hist*100:.2f}%)")

    output_dir.mkdir(parents=True, exist_ok=True)
    resumen = {
        "filas_historico": n_filas_hist,
        "clientes_unicos_historico": len(ids_hist_set),
        "clientes_match_cc": len(ids_match),
        "clientes_no_match_cc": len(ids_no_match),
        "pct_match": round(len(ids_match) / len(ids_hist_set) * 100, 2),
        "filas_edad_nula_tras_join": n_nulos_edad,
        "filas_sexo_nula_tras_join": n_nulos_sexo,
    }
    with open(output_dir / "resumen_join.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"\n   resumen guardado en: {output_dir / 'resumen_join.json'}")

    out_parquet = output_dir / "historico_con_demograficos.parquet"
    merged.to_parquet(out_parquet, index=False)
    print(f"   parquet guardado en: {out_parquet}  ({out_parquet.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    args = parse_args()
    cc = cargar_cc(args.cc)
    cc = limpiar_identificacion(cc)
    cc = calcular_edad(cc)
    cc = revisar_sexo(cc)
    cc_dedup = deduplicar(cc)
    hacer_join(args.historico, cc_dedup, args.output_dir)


if __name__ == "__main__":
    main()
