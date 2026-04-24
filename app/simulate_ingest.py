"""
simulate_ingest.py — Simula la llegada de nuevas transacciones.

Toma N días del archivo de simulación y los inserta en el histórico base.
Genera:
  - nuevas_series.csv: (CODIGO_FAMILIA;COD_SUBCATEGORIA) de las series modificadas
  - tx_counter.json: contador acumulado de nuevas transacciones

Uso CLI (ejecutar desde terminal):
  python simulate_ingest.py --days 1
  python simulate_ingest.py --days 3
  python simulate_ingest.py --status           # ver estado actual
  python simulate_ingest.py --reset            # resetear todo al estado inicial
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import DATA_DIR, RAW_DIR


SIM_FILE = RAW_DIR / "historico_1000_sim.csv"
BASE_FILE = RAW_DIR / "historico_1000_base.csv"
COUNTER_FILE = DATA_DIR / "tx_counter.json"
SERIES_FILE = DATA_DIR / "nuevas_series.csv"
CURSOR_FILE = DATA_DIR / "ingest_cursor.json"


def _load_counter() -> dict:
    if COUNTER_FILE.exists():
        with open(COUNTER_FILE) as f:
            return json.load(f)
    return {"total_new": 0, "last_ingest": None}


def _save_counter(counter: dict):
    with open(COUNTER_FILE, "w") as f:
        json.dump(counter, f, indent=2)


def _load_cursor() -> dict:
    """Cursor: lleva control de hasta qué fecha ya se inyectó."""
    if CURSOR_FILE.exists():
        with open(CURSOR_FILE) as f:
            return json.load(f)
    return {"last_date_ingested": None}


def _save_cursor(cursor: dict):
    with open(CURSOR_FILE, "w") as f:
        json.dump(cursor, f, indent=2)


def ingest_days(n_days: int):
    """Inserta los próximos N días de simulación al histórico base."""
    if not SIM_FILE.exists():
        print(f"[Error] No existe {SIM_FILE}")
        return

    df_sim = pd.read_csv(SIM_FILE, sep=";", parse_dates=["DIM_PERIODO"])
    cursor = _load_cursor()

    if cursor["last_date_ingested"]:
        last_date = pd.Timestamp(cursor["last_date_ingested"])
        df_remaining = df_sim[df_sim["DIM_PERIODO"] > last_date]
    else:
        df_remaining = df_sim

    if df_remaining.empty:
        print("[Ingest] No quedan más datos de simulación por inyectar.")
        return

    available_dates = sorted(df_remaining["DIM_PERIODO"].dt.date.unique())
    dates_to_ingest = available_dates[:n_days]

    if not dates_to_ingest:
        print("[Ingest] No hay más fechas disponibles.")
        return

    mask = df_remaining["DIM_PERIODO"].dt.date.isin(dates_to_ingest)
    df_new = df_remaining[mask].copy()

    # Insertar en base
    df_base = pd.read_csv(BASE_FILE, sep=";")
    df_updated = pd.concat([df_base, df_new], ignore_index=True)
    df_updated.to_csv(BASE_FILE, index=False, sep=";")

    # Series modificadas (acumuladas)
    new_series = df_new[["CODIGO_FAMILIA", "COD_SUBCATEGORIA"]].drop_duplicates()
    if SERIES_FILE.exists():
        old_series = pd.read_csv(SERIES_FILE, sep=";")
        all_series = pd.concat([old_series, new_series]).drop_duplicates()
    else:
        all_series = new_series
    all_series.to_csv(SERIES_FILE, index=False, sep=";")

    # Actualizar counter
    counter = _load_counter()
    counter["total_new"] += len(df_new)
    counter["last_ingest"] = str(dates_to_ingest[-1])
    _save_counter(counter)

    # Actualizar cursor
    cursor["last_date_ingested"] = str(dates_to_ingest[-1])
    _save_cursor(cursor)

    # Publicar a Kafka
    from kafka_helper import publish
    publish("tx.nuevas", {
        "registros_insertados": len(df_new),
        "dias_insertados": len(dates_to_ingest),
        "fecha_inicio": str(dates_to_ingest[0]),
        "fecha_fin": str(dates_to_ingest[-1]),
        "series_afectadas": len(new_series),
        "tx_acumuladas": counter["total_new"],
    })

    print(f"[Ingest] Insertados {len(df_new)} registros de {len(dates_to_ingest)} días")
    print(f"  Fechas: {dates_to_ingest[0]} → {dates_to_ingest[-1]}")
    print(f"  Series afectadas (esta vez): {len(new_series)}")
    print(f"  Series acumuladas total: {len(all_series)}")
    print(f"  Nuevas transacciones acumuladas: {counter['total_new']}")
    print(f"  Base total: {len(df_updated)} registros")

    remaining_dates = len(available_dates) - len(dates_to_ingest)
    remaining_records = len(df_remaining[~mask])
    print(f"  Sim restante: {remaining_dates} días, {remaining_records} registros")


def show_status():
    """Muestra estado actual del sistema."""
    counter = _load_counter()
    cursor = _load_cursor()

    print("=" * 50)
    print("ESTADO DEL SISTEMA")
    print("=" * 50)
    print(f"  Tx nuevas acumuladas: {counter['total_new']}")
    print(f"  Última ingesta: {counter['last_ingest']}")
    print(f"  Último día inyectado: {cursor.get('last_date_ingested', 'ninguno')}")

    if BASE_FILE.exists():
        df = pd.read_csv(BASE_FILE, sep=";")
        df["DIM_PERIODO"] = pd.to_datetime(df["DIM_PERIODO"], format="mixed")
        print(f"  Base: {len(df)} registros | {df['DIM_PERIODO'].min().date()} → {df['DIM_PERIODO'].max().date()}")

    if SERIES_FILE.exists():
        series = pd.read_csv(SERIES_FILE, sep=";")
        print(f"  Series pendientes de reentrenar: {len(series)}")

    if SIM_FILE.exists():
        df_sim = pd.read_csv(SIM_FILE, sep=";", parse_dates=["DIM_PERIODO"])
        if cursor["last_date_ingested"]:
            remaining = df_sim[df_sim["DIM_PERIODO"] > pd.Timestamp(cursor["last_date_ingested"])]
        else:
            remaining = df_sim
        print(f"  Sim restante: {len(remaining)} registros, {remaining['DIM_PERIODO'].dt.date.nunique()} días")

    print("=" * 50)


def reset():
    """Resetea el sistema al estado inicial (reconstruye base sin sim)."""
    import shutil

    # Reconstruir base desde el historico_1000.csv original
    original = RAW_DIR / "historico_1000.csv"
    if not original.exists():
        print(f"[Error] No existe {original} para reconstruir")
        return

    # Releer y re-splitear
    from simulation import split_historico
    split_historico(original, trim_months=2)

    # Limpiar archivos de estado
    for f in (COUNTER_FILE, SERIES_FILE, CURSOR_FILE):
        if f.exists():
            f.unlink()

    print("[Reset] Sistema reseteado al estado inicial.")
    show_status()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulación de ingesta de transacciones")
    parser.add_argument("--days", type=int, default=0, help="Número de días a inyectar")
    parser.add_argument("--status", action="store_true", help="Ver estado actual")
    parser.add_argument("--reset", action="store_true", help="Resetear al estado inicial")
    args = parser.parse_args()

    if args.reset:
        reset()
    elif args.status:
        show_status()
    elif args.days > 0:
        ingest_days(args.days)
    else:
        parser.print_help()
