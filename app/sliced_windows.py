"""
sliced_windows.py — Genera ventanas deslizantes del historico para augmentar datos de entrenamiento.

Estrategia para reducir desbalance (5.8% positivos):
  Ventana 0: slice completo [inicio, fecha_corte + target_dias].
             Contiene todos los pares (target 0 y 1) → conjunto base.
  Ventana i (i > 0): mismo tamaño pero desplazado i * target_dias hacia adelante.
             compradores + muestra aleatoria de no_compradores (neg_ratio:1).

Cada CSV generado es un historico válido para feature_engineering.py con su
correspondiente fecha_corte (= inicio_ventana + historico_dias).

Uso:
  python sliced_windows.py --fecha-inicio 2023-01-01 --historico-dias 1074 --n-ventanas 5 --neg-ratio 4

Naming convention:
  {window_inicio_YYYYMMDD}_{fecha_corte_YYYYMMDD}_{tag}.csv

Outputs en data/raw/:
  20230101_20251210_C.csv   ← ventana 0: todos los pares (target 0 y 1)
  20230122_20251231_C.csv   ← ventana 1: compradores + no_compradores 4x
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
from config import DATA_DIR, HISTORICO_FILE, PREDICTION_WINDOW_DAYS

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
    neg_ratio: float = 4.0,
    seed: int = 42,
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
            n_no_compradores = None
        else:
            # Ventanas siguientes: compradores + muestra aleatoria de no-compradores (neg_ratio:1)
            if n_compradores == 0:
                print(f"[Window {i:02d}] 0 compradores en target [{fecha_corte.date()} → {window_fin.date()}], omitiendo.")
                continue

            # Paso 1: universo de pares con historia hasta fecha_corte (unico lugar donde
            #         feature_engineering puede calcular features reales; sin historia previa no hay features).
            mask_historico    = slice_df[DATE_COL] <= fecha_corte
            df_pares_totales  = slice_df[mask_historico][KEY_COLS].drop_duplicates()

            # Paso 2: compradores = los que ya calculamos arriba (compraron en el target)
            df_compradores = buyers

            # Paso 3: no_compradores = pares_totales - compradores (diferencia de conjuntos)
            es_comprador = df_pares_totales.set_index(KEY_COLS).index.isin(
                df_compradores.set_index(KEY_COLS).index
            )
            df_no_compradores_pool = df_pares_totales[~es_comprador]

            # Paso 4: muestreo aleatorio de no_compradores, tamano = neg_ratio * n_compradores
            a = n_compradores
            b = len(df_no_compradores_pool)
            n_muestra = min(int(round(neg_ratio * a)), b)
            df_no_compradores_muestra = (
                df_no_compradores_pool.sample(n=n_muestra, random_state=seed)
                if n_muestra > 0 else df_no_compradores_pool.iloc[0:0]
            )
            n_no_compradores = len(df_no_compradores_muestra)

            # Paso 5: union final = compradores + no_compradores muestreados
            df_pares_finales = pd.concat([df_compradores, df_no_compradores_muestra], ignore_index=True)
            out_df = slice_df.merge(df_pares_finales, on=KEY_COLS, how="inner")
            tipo   = f"compradores_mas_no_compradores_{neg_ratio:g}x"

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
            "n_no_compradores": n_no_compradores,
            "archivo":        filename,
            "feature_eng_cmd": (
                f"python feature_engineering.py "
                f"--historico data/raw/{filename} "
                f"--fecha-corte {fecha_corte.date()}"
            ),
        }
        summary.append(meta)
        print(f"[Window {i:02d}] {tipo:<30} | "
              f"fecha_corte={fecha_corte.date()} | "
              f"{n_pares:>6,} pares | {n_compradores:>5,} compradores | "
              f"{n_no_compradores if n_no_compradores is not None else '-':>6} no_compradores | "
              f"{len(out_df):>9,} filas → {filename}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Ventanas deslizantes del historico")
    parser.add_argument("--historico",      type=str, default=str(HISTORICO_FILE),
                        help="Path al CSV de transacciones crudas")
    parser.add_argument("--fecha-inicio",   type=str, required=True,
                        help="Inicio de la primera ventana (YYYY-MM-DD)")
    parser.add_argument("--historico-dias", type=int, default=1074,
                        help="Días de lookback para features (default: 1074)")
    parser.add_argument("--target-dias",    type=int, default=PREDICTION_WINDOW_DAYS,
                        help=f"Días de ventana target (default: {PREDICTION_WINDOW_DAYS})")
    parser.add_argument("--n-ventanas",     type=int, default=5,
                        help="Número de ventanas (default: 5)")
    parser.add_argument("--tag",            type=str, default="C",
                        help="Identificador del archivo fuente en el nombre (default: C)")
    parser.add_argument("--output-dir",     type=str, default=str(OUTPUT_DIR),
                        help="Directorio de salida (default: data/raw/)")
    parser.add_argument("--neg-ratio",      type=float, default=4.0,
                        help="Ratio no_compradores:compradores en ventanas i>0 (default: 4.0)")
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
        neg_ratio      = args.neg_ratio,
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
