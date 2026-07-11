"""
sliced_windows.py — Genera ventanas deslizantes del historico para augmentar datos de entrenamiento.

Estrategia para reducir desbalance (5.8% positivos):
  Ventana 0: slice completo [inicio, fecha_corte + target_dias].
             Contiene todos los pares (target 0 y 1) → conjunto base.
  Ventana i (i > 0): mismo tamaño pero desplazado i * target_dias hacia adelante.
             FILTRADO: solo pares (CODIGO_FAMILIA, COD_SUBCATEGORIA) que compraron
             en (fecha_corte_i, fecha_corte_i + target_dias].
             → Todos tienen target=1 al pasar por feature_engineering.

Cada CSV generado es un historico válido para feature_engineering.py con su
correspondiente fecha_corte (= inicio_ventana + historico_dias).

Uso:
  python sliced_windows.py --fecha-inicio 2022-01-01 --historico-dias 365 --n-ventanas 5
  python sliced_windows.py --fecha-inicio 2021-06-01 --historico-dias 730 --n-ventanas 4

Naming convention:
  {window_inicio_YYYYMMDD}_{fecha_corte_YYYYMMDD}_{tag}.csv
  tag identifica el archivo fuente (C, D, ...); las fechas distinguen cada ventana.

Outputs en data/raw/:
  20220101_20230101_C.csv   ← ventana 0: todos los pares (target 0 y 1)
  20220122_20230122_C.csv   ← ventana 1: solo compradores
  20220212_20230212_C.csv   ← ventana 2: solo compradores
  ...
  sliced_summary_C.json     ← metadata de cada ventana

Siguiente paso:
  Por cada CSV generado, ejecutar feature_engineering.py con su fecha_corte.
  El summary.json incluye el comando exacto por ventana.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, HISTORICO_FILE, PARSE_DATES, RAW_DTYPES, PREDICTION_WINDOW_DAYS

DATE_COL   = "DIM_PERIODO"
KEY_COLS   = ["CODIGO_FAMILIA", "COD_SUBCATEGORIA"]   # claves en el historico crudo
OUTPUT_DIR = DATA_DIR / "raw"


def _print_range(df: pd.DataFrame, tag: str) -> None:
    print(f"  [{tag}] {len(df):,} filas | "
          f"{df[DATE_COL].min().date()} → {df[DATE_COL].max().date()}")


def generate_windows(
    df: pd.DataFrame,
    fecha_inicio: pd.Timestamp,
    historico_dias: int,
    target_dias: int,
    n_ventanas: int,
    output_dir: Path,
    tag: str = "C",
) -> list[dict]:
    """
    Genera N ventanas deslizadas y guarda un CSV por ventana.

    Parámetros
    ----------
    df             : historico crudo completo
    fecha_inicio   : inicio de la primera ventana
    historico_dias : días de lookback para features
    target_dias    : días de ventana target (predicción)
    n_ventanas     : cuántas ventanas generar
    output_dir     : directorio de salida

    Retorna lista de metadata por ventana.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for i in range(n_ventanas):
        window_inicio = fecha_inicio + timedelta(days=i * target_dias)
        fecha_corte   = window_inicio + timedelta(days=historico_dias)
        window_fin    = fecha_corte   + timedelta(days=target_dias)

        # Slice del periodo completo (historico + target window)
        mask_slice = (df[DATE_COL] >= window_inicio) & (df[DATE_COL] <= window_fin)
        slice_df   = df[mask_slice].copy()

        if len(slice_df) == 0:
            print(f"[Window {i:02d}] Sin datos en [{window_inicio.date()} → {window_fin.date()}], omitiendo.")
            continue

        # Compradores: pares con al menos 1 tx en la ventana target
        mask_target  = (slice_df[DATE_COL] > fecha_corte) & (slice_df[DATE_COL] <= window_fin)
        buyers       = slice_df[mask_target][KEY_COLS].drop_duplicates()
        n_compradores = len(buyers)

        if i == 0:
            # Ventana base: todos los pares (balance natural)
            out_df      = slice_df
            tipo        = "base_todos_pares"
        else:
            # Ventanas siguientes: solo compradores con su historico completo
            if n_compradores == 0:
                print(f"[Window {i:02d}] 0 compradores en target [{fecha_corte.date()} → {window_fin.date()}], omitiendo.")
                continue
            out_df = slice_df.merge(buyers, on=KEY_COLS, how="inner")
            tipo   = "solo_compradores"

        filename = f"{window_inicio.strftime('%Y%m%d')}_{fecha_corte.strftime('%Y%m%d')}_{tag}.csv"
        filepath = output_dir / filename
        out_df.to_csv(filepath, index=False, sep=";")

        n_pares = out_df[KEY_COLS].drop_duplicates().shape[0]
        meta = {
            "ventana":        i,
            "tipo":           tipo,
            "window_inicio":  str(window_inicio.date()),
            "fecha_corte":    str(fecha_corte.date()),
            "window_fin":     str(window_fin.date()),
            "n_filas":        len(out_df),
            "n_pares":        n_pares,
            "n_compradores":  n_compradores,
            "archivo":        filename,
            "feature_eng_cmd": (
                f"python feature_engineering.py "
                f"--historico data/raw/{filename} "
                f"--fecha-corte {fecha_corte.date()}"
            ),
        }
        summary.append(meta)
        print(f"[Window {i:02d}] {tipo:<22} | "
              f"fecha_corte={fecha_corte.date()} | "
              f"{n_pares:>6,} pares | {n_compradores:>5,} compradores | "
              f"{len(out_df):>9,} filas → {filename}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Ventanas deslizantes del historico")
    parser.add_argument("--historico",      type=str, default=str(HISTORICO_FILE),
                        help="Path al CSV de transacciones crudas")
    parser.add_argument("--fecha-inicio",   type=str, required=True,
                        help="Inicio de la primera ventana (YYYY-MM-DD)")
    parser.add_argument("--historico-dias", type=int, default=365,
                        help="Días de lookback para features (default: 365)")
    parser.add_argument("--target-dias",    type=int, default=PREDICTION_WINDOW_DAYS,
                        help=f"Días de ventana target (default: {PREDICTION_WINDOW_DAYS})")
    parser.add_argument("--n-ventanas",     type=int, default=5,
                        help="Número de ventanas (default: 5)")
    parser.add_argument("--tag",            type=str, default="C",
                        help="Identificador del archivo fuente en el nombre (default: C)")
    parser.add_argument("--output-dir",     type=str, default=str(OUTPUT_DIR),
                        help="Directorio de salida (default: data/raw/)")
    args = parser.parse_args()

    historico_path = Path(args.historico)
    if not historico_path.exists():
        print(f"[ERROR] No se encontró: {historico_path}")
        return

    fecha_inicio    = pd.Timestamp(args.fecha_inicio)
    output_dir      = Path(args.output_dir)
    tag             = args.tag.upper()
    datos_necesarios = args.historico_dias + args.n_ventanas * args.target_dias
    fecha_fin_estimada = fecha_inicio + timedelta(days=datos_necesarios)

    print("[SlicedWindows] Configuración:")
    print(f"  historico      = {historico_path.name}")
    print(f"  fecha_inicio   = {fecha_inicio.date()}")
    print(f"  historico_dias = {args.historico_dias}")
    print(f"  target_dias    = {args.target_dias}")
    print(f"  n_ventanas     = {args.n_ventanas}")
    print(f"  tag            = {tag}")
    print(f"  datos_necesarios = {datos_necesarios} días "
          f"(hasta ~{fecha_fin_estimada.date()})")
    print()

    print(f"[SlicedWindows] Cargando {historico_path.name} ...")
    df = pd.read_csv(historico_path, encoding="utf-8", sep=";", low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%d-%b-%y", errors="coerce")
    _print_range(df, "historico completo")

    max_date = df[DATE_COL].max()
    if max_date < fecha_fin_estimada:
        print(f"\n[AVISO] El historico llega hasta {max_date.date()} pero se necesitan "
              f"datos hasta ~{fecha_fin_estimada.date()}. "
              f"Algunas ventanas pueden quedar vacías.\n")

    print()
    summary = generate_windows(
        df             = df,
        fecha_inicio   = fecha_inicio,
        historico_dias = args.historico_dias,
        target_dias    = args.target_dias,
        n_ventanas     = args.n_ventanas,
        output_dir     = output_dir,
        tag            = tag,
    )

    summary_path = output_dir / f"sliced_summary_{tag}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[SlicedWindows] {len(summary)} ventanas generadas en {output_dir}")
    print(f"[SlicedWindows] Resumen: {summary_path}")
    print("\nSiguiente paso — ejecutar feature_engineering por cada ventana:")
    for meta in summary:
        print(f"  {meta['feature_eng_cmd']}")


if __name__ == "__main__":
    main()
