from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _serie_fechas(values: Iterable) -> pd.Series:
    """Convierte valores a fecha, elimina nulos, deduplica y ordena."""
    s = pd.Series(values)
    s = pd.to_datetime(s, errors="coerce").dropna().drop_duplicates().sort_values()
    return s.reset_index(drop=True)


def calcular_dias_desde_ultima_compra(
    fechas: Iterable,
    fecha_corte: pd.Timestamp,
) -> float:
    """Dias entre la ultima fecha distinta de compra y la fecha de corte."""
    s = _serie_fechas(fechas)
    if s.empty:
        return np.nan
    return float((fecha_corte - s.iloc[-1]).days)


def calcular_total_compras(facturas: Iterable) -> int:
    """Numero de facturas distintas observadas."""
    return int(pd.Series(facturas).dropna().nunique())


def calcular_gasto_total(venta_neta: Iterable) -> float:
    """Suma de VENTA_NETA en todas las filas transaccionales."""
    valores = pd.to_numeric(pd.Series(venta_neta), errors="coerce").fillna(0.0)
    return float(valores.sum())


def calcular_ticket_promedio(venta_neta: Iterable, facturas: Iterable) -> float:
    """Venta neta total dividida para el numero de facturas distintas."""
    gasto_total = calcular_gasto_total(venta_neta)
    total_facturas = calcular_total_compras(facturas)
    if total_facturas == 0:
        return 0.0
    return float(gasto_total / total_facturas)


def calcular_longitud_relacion_dias(
    fechas: Iterable,
    fecha_corte: pd.Timestamp,
) -> float:
    """Dias entre la primera compra observada y la fecha de corte."""
    s = _serie_fechas(fechas)
    if s.empty:
        return np.nan
    return float((fecha_corte - s.iloc[0]).days)


def calcular_intervalos(fechas: Iterable) -> np.ndarray:
    """Vector de intervalos en dias entre fechas distintas consecutivas."""
    s = _serie_fechas(fechas)
    if len(s) < 2:
        return np.array([], dtype=float)
    valores = s.to_numpy(dtype="datetime64[ns]")
    return np.diff(valores).astype("timedelta64[D]").astype(float)


def calcular_intervalo_promedio(fechas: Iterable) -> float:
    """Media de los intervalos entre fechas distintas consecutivas."""
    intervalos = calcular_intervalos(fechas)
    if intervalos.size == 0:
        return np.nan
    return float(np.mean(intervalos))


def calcular_intervalo_maximo(fechas: Iterable) -> float:
    """Maximo intervalo observado entre fechas distintas consecutivas."""
    intervalos = calcular_intervalos(fechas)
    if intervalos.size == 0:
        return np.nan
    return float(np.max(intervalos))


def calcular_intervalo_cv(fechas: Iterable) -> float:
    """Coeficiente de variacion poblacional: std(ddof=0) / media."""
    intervalos = calcular_intervalos(fechas)
    if intervalos.size == 0:
        return np.nan
    media = float(np.mean(intervalos))
    if media <= 0:
        return np.nan
    return float(np.std(intervalos, ddof=0) / media)


def calcular_recencia_relativa(
    dias_desde_ultima_compra: float,
    intervalo_promedio: float,
) -> float:
    """Recencia dividida para el intervalo promedio individual."""
    if pd.isna(dias_desde_ultima_compra) or pd.isna(intervalo_promedio):
        return np.nan
    if intervalo_promedio <= 0:
        return np.nan
    return float(dias_desde_ultima_compra / intervalo_promedio)


def calcular_compras_en_ventana(
    fechas: Iterable,
    facturas: Iterable,
    inicio_exclusivo: pd.Timestamp,
    fin_inclusivo: pd.Timestamp,
) -> int:
    """Cuenta facturas distintas dentro de (inicio_exclusivo, fin_inclusivo]."""
    df = pd.DataFrame({
        "fecha": pd.to_datetime(pd.Series(fechas), errors="coerce"),
        "factura": pd.Series(facturas),
    })
    mask = (df["fecha"] > inicio_exclusivo) & (df["fecha"] <= fin_inclusivo)
    return int(df.loc[mask, "factura"].dropna().nunique())


def calcular_compras_ultimos_180d(
    fechas: Iterable,
    facturas: Iterable,
    fecha_corte: pd.Timestamp,
) -> int:
    """Facturas distintas observadas en (T-180 dias, T]."""
    inicio = fecha_corte - pd.Timedelta(days=180)
    return calcular_compras_en_ventana(fechas, facturas, inicio, fecha_corte)


def calcular_delta_frecuencia_180d(
    fechas: Iterable,
    facturas: Iterable,
    fecha_corte: pd.Timestamp,
) -> int:
    """Facturas recientes menos facturas de los 180 dias inmediatamente anteriores."""
    corte_180 = fecha_corte - pd.Timedelta(days=180)
    corte_360 = fecha_corte - pd.Timedelta(days=360)
    recientes = calcular_compras_en_ventana(fechas, facturas, corte_180, fecha_corte)
    anteriores = calcular_compras_en_ventana(fechas, facturas, corte_360, corte_180)
    return int(recientes - anteriores)


def calcular_subcategorias_distintas(subcategorias: Iterable) -> int:
    """Numero de COD_SUBCATEGORIA distintos observados."""
    return int(pd.Series(subcategorias).dropna().nunique())
