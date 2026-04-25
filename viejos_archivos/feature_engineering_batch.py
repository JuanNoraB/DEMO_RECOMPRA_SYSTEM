#%%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import math

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "Data"
HISTORICAL_FILE = DATA_DIR / "Historico_08122025.csv"
OUTPUT_FILE = BASE_DIR / "features_final_all_familias.csv"

RAW_DTYPES = {
    "COD_SUBCATEGORIA": "Int64",
    "COD_CATEGORIA": "Int64",
    "COD_UNIDAD_COMERCIAL": "Int64",
    "COD_ITEM": "Int64",
    "DIM_FACTURA": "Int64",
    "COD_LOCAL": "Int64",
    "CODIGO_FAMILIA": "Int64",
}

NUMERIC_COLUMNS = ["CANTIDAD_SUELTA", "PVP", "VENTA_NETA", "DESCUENTO"]
PARSE_DATES = ["DIM_PERIODO"]

CUTOFF_DATE = pd.Timestamp("2025-11-30")  # Actualizado para incluir datos de noviembre
RECENT_WINDOW_DAYS = 60
FREQUENCY_WINDOW_DAYS = 180
SOW_MONTHS_24 = 24
SOW_MONTHS_12 = 12


def load_historical_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo histórico en: {path}")

    df = pd.read_csv(
        path,
        encoding='utf-8',
        sep=';')
        

    # Convertir fechas
    df["DIM_PERIODO"] = pd.to_datetime(
        df["DIM_PERIODO"],
        format="%d-%b-%y", # ejemplo: 24-Nov-25
        errors="coerce",
    )

    # Convertir columnas numéricas
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    
    # Convertir columnas int con coercion (rellenar NaN antes de convertir)
    for col in RAW_DTYPES.keys():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df = df.dropna(subset=["DIM_PERIODO", "CODIGO_FAMILIA", "COD_SUBCATEGORIA"])
    df["CODIGO_FAMILIA"] = df["CODIGO_FAMILIA"].astype(int)
    df["COD_SUBCATEGORIA"] = df["COD_SUBCATEGORIA"].astype(int)

    #filtrar solo familia especifica 1719363028
    #ceduleas  = [1102088158,1719363028]
    #df = df[df["CODIGO_FAMILIA"].isin(ceduleas)]


    return df

def calcular_cv_normalizado(gaps_dias):
    """
    Normaliza gaps dividiendo por el mínimo y calcula el Coeficiente de Variación (CV).
    
    Returns:
        cv (float): Coeficiente de variación normalizado
        gaps_norm (array): Gaps normalizados
    """
    arr = np.asarray(gaps_dias, dtype=float)
    
    if arr.size < 2:
        return 999.0, arr  # CV infinito = muy irregular
    
    # Normalización: dividir por el mínimo
    min_gap = np.min(arr)
    if min_gap == 0:
        return 999.0, arr
    
    gaps_norm = arr / min_gap
  
    # Calcular CV normalizado
    mean_norm = np.mean(gaps_norm)
    std_norm = np.std(gaps_norm)
    
    cv = std_norm / mean_norm if mean_norm > 0 else 999.0
    
    # Clipear CV a rango [0, 1] para casos extremos
    if cv != 999.0:
        cv = min(cv, 1.0)
    
    return cv, gaps_norm

def calcular_ciclos(
    df_ventas,
    familia_id,
    subcat,
    meses_historico=12,
    periodo_dias=7,
    min_compras=5,
    max_compras_recientes=15,
    cv_threshold=0.6,
    today=pd.Timestamp.today(),
    tipo="corto",
    clase_de_calculo = 1
):
    '''
    RESTRICCIONES POR TIPO DE CICLO 
    corto:
        ciclo_dias >= 3 && ciclo_dias <= 30
        min_compras >= 5
        max_compras_recientes = 15
        meses_historico = 12
        cv = 1

    corto_medio:
        ciclo_dias >= 30 && ciclo_dias <= 75
        min_compras >= 5
        max_compras_recientes = 15
        meses_historico = 12
        cv = 0.95

    media:
        ciclo_dias >=75 && ciclo_dias <=150
        min_compras >= 4
        max_compras_recientes = 10
        meses_historico = 18
        cv = 0.6

    largo:
        ciclo_dias > 150
        min_compras >= 4
        max_compras_recientes = 10
        meses_historico = 36
        cv = 0.45
    '''

    '''
    Cambio del calculo de cv con para frecuencias dependiendo de tipo de ciclo calculos replica
   				CON 2 	CON 3
        0	30	C	1	60	90
        31	75	CM	0.95	150	225
        75	150	M	0.6	300	450
        150	300	L	0.45	600	900

    corto = 60/90  --30 
    medio-cortp = 150-225
    mediano = 300/450
    largo = 600/900
    '''
    if tipo == 'corto':
        dias_cv =90
    elif tipo == "corto_medio":
        dias_cv = 225
    elif tipo == "mediano":
        dias_cv = 450
    elif tipo == "largo":
        dias_cv = 900
    

    today = today.normalize()
    fecha_inicio = today - pd.DateOffset(months=meses_historico)

    fecha_inicio_cv = today - pd.DateOffset(days=dias_cv)
    
    df_sub = df_ventas[
        (df_ventas["CODIGO_FAMILIA"] == familia_id) &
        (df_ventas["COD_SUBCATEGORIA"] == subcat) &
        (pd.to_datetime(df_ventas["DIM_PERIODO"]) >= fecha_inicio)
    ].copy()

    df_cv = df_ventas[
        (df_ventas["CODIGO_FAMILIA"] == familia_id) &
        (df_ventas["COD_SUBCATEGORIA"] == subcat) &
        (pd.to_datetime(df_ventas["DIM_PERIODO"]) >= fecha_inicio_cv)
    ].copy()
    
    if df_sub.empty:
        return {"ciclo_dias": [0,0,0], "cv": 999, "tipo": "no_ciclico", "razon": f"sin_datos {tipo}"}
    
    # Calcular bloques
    #gaps para poder determinar


    dias_desde_inicio = (df_sub["DIM_PERIODO"] - fecha_inicio).dt.days
    df_sub["bloque"] = dias_desde_inicio // periodo_dias


    dias_desde_inicio_cv = (df_cv["DIM_PERIODO"] - fecha_inicio_cv).dt.days
    df_cv["bloque"] = dias_desde_inicio_cv // periodo_dias
    
    # Ordenar por fecha y tomar bloques únicos
    df_sub = df_sub.sort_values("DIM_PERIODO", ascending=False)
    bloques_con_compra = np.sort(df_sub["bloque"].unique())

    df_cv = df_cv.sort_values("DIM_PERIODO", ascending=False)
    bloques_con_compra_cv = np.sort(df_cv["bloque"].unique())

    #df_sub[['DIM_PERIODO','bloque']].sort_values(by='DIM_PERIODO',ascending=False)
    # Limitar a últimas N compras
    if len(bloques_con_compra) > max_compras_recientes:
        bloques_con_compra = bloques_con_compra[-max_compras_recientes:]
    
    # Verificar mínimo de compras
    if len(bloques_con_compra) < 2:
        return {"ciclo_dias": [0,0,0], "cv": 999, "tipo": "no_ciclico", "razon": f"pocas_compras 2 {tipo}","gaps_originales": [], "gaps_normalizados": [],'gaps_ciclos_bloques': []}
    

    # Calcular gaps de BLOQUES (para CV - suavizado)
    gaps_bloques = np.diff(bloques_con_compra)

    gapas_bloques_cv = np.diff(bloques_con_compra_cv)
    if len(gaps_bloques) == 0:
        # Solo 1 compra: no hay forma de calcular gaps
        return {"ciclo_dias": [0,0,0], "cv": 999, "tipo": "no_ciclico", "razon": f"sin_gaps {tipo}", "gaps_originales": [], "gaps_normalizados": [],'gaps_ciclos_bloques': []}
    
    gaps_dias_bloques = gaps_bloques * periodo_dias
    gaps_dias_bloques_cv = gapas_bloques_cv * periodo_dias
    
    # Calcular CV normalizado usando gaps de bloques (suavizados)
    _, gaps_norm = calcular_cv_normalizado(gaps_dias_bloques)
    cv,_ = calcular_cv_normalizado(gaps_dias_bloques_cv)

    

    # Calcular gaps REALES TENDIENDO A LA ALTA POR EL MAX
    # Para esto, tomamos las fechas únicas ordenadas
    fechas_unicas = df_sub.groupby('bloque')['DIM_PERIODO'].max().sort_values()
    # ver si sobrepasa las compras a maximo reciente para corto por ejemplo 15 
    if len(bloques_con_compra) > max_compras_recientes:
        fechas_unicas = fechas_unicas.iloc[-max_compras_recientes:]
        #POR ESO EL RODEN ASENDEDENTE PARA PODER OBTENER LAS ULITMAS COMPRAS 
    
    gaps_dias_reales = np.diff(fechas_unicas).astype('timedelta64[D]').astype(int)
    gaps_dias_reales = gaps_dias_reales.tolist()



    
    ciclo_dias = int(float(np.mean(gaps_dias_reales)))



    

    if tipo == "corto":
        limite_inferior = 3
        limite_superior = 30
        hacia_abajo = 0.25
        hacia_arriba = 2
    elif tipo == "corto_medio":
        limite_inferior = 30
        limite_superior = 75
        hacia_abajo = 0.25
        hacia_arriba = 1.25
    elif tipo == "mediano":
        limite_inferior = 75
        limite_superior = 150
        hacia_abajo = 0.25
        hacia_arriba = 0.75
    elif tipo == "largo":
        limite_inferior = 150
        limite_superior = 360
        hacia_abajo = 0.2
        hacia_arriba = 0.3
    
    #CAMBIO DE FRECUCIAS LARA LIMITE INFERIOR NO ACUMULATIVO
    limite_inferior = 3
    # Media correspondiente con el tipo del ciclo a analizar
    if len(bloques_con_compra) < min_compras:
        return {
            "ciclo_dias": [0,ciclo_dias,0],
            "cv": cv,
            "tipo": 'no_ciclico',
            "razon": f'min_compras {tipo}',
            "gaps_originales": gaps_dias_reales,
            "gaps_normalizados": gaps_dias_bloques_cv.tolist(),
            "gaps_ciclos_bloques":gaps_dias_bloques.tolist()
            }

    # Decidir si es cíclico
    if cv <= cv_threshold or clase_de_calculo == 0:
        # Ciclo promedio basado en gaps REALES
        
        if ciclo_dias > limite_inferior and ciclo_dias <= limite_superior:
            # ES CÍCLICO: retornar con intervalos
            return {
            "ciclo_dias": [ciclo_dias*(1-cv_threshold*hacia_abajo),ciclo_dias,ciclo_dias*(1+cv_threshold*hacia_arriba)],
            "cv": cv,
            "tipo": tipo,
            "razon": f"OK {tipo}",
            "gaps_originales": gaps_dias_reales,
            "gaps_normalizados": gaps_dias_bloques_cv.tolist(),
            "gaps_ciclos_bloques":gaps_dias_bloques.tolist()
            }
        # NO CÍCLICO (fuera de rango): retornar ciclo_dias calculados para DEBUG
        return {
            "ciclo_dias": [0, ciclo_dias, 0], 
            "cv": cv, 
            "tipo": "no_ciclico", 
            "razon": f"fuera_rango {tipo}",
            "gaps_originales": gaps_dias_reales,
            "gaps_normalizados": gaps_dias_bloques_cv.tolist(),
            "gaps_ciclos_bloques":gaps_dias_bloques.tolist()
            }
    else:
        # NO CÍCLICO (cv alto): retornar ciclo_dias calculados para DEBUG
        return {
            "ciclo_dias": [0, ciclo_dias, 0],
            "cv": cv,
            "tipo": "no_ciclico",
            "razon": f"cv_alto {tipo}",
            "gaps_originales": gaps_dias_reales,
            "gaps_normalizados": gaps_dias_bloques_cv.tolist(),
            "gaps_ciclos_bloques":gaps_dias_bloques.tolist()
        }



def calcular_ciclos_por_bloques(
    df_ventas,
    familia_id,
    today=pd.Timestamp.today(),
    clase_de_calculo = 1

):
    """
    Orquestador: intenta primero ciclos cortos, luego largos.
    Retorna DataFrame con resultados para todas las subcategorías.
    """
    today = today.normalize()
    
    # Obtener subcategorías de la familia
    df_fam = df_ventas[df_ventas["CODIGO_FAMILIA"] == familia_id].copy()
    if df_fam.empty:
        return pd.DataFrame()
    
    subcategorias = df_fam["COD_SUBCATEGORIA"].unique()
    resultados = []
    
    for subcat in subcategorias:
        ciclos_clase = [
            {"tipo": "corto", "cv_threshold": 1,"min_compras": 5,"max_compras_recientes": 15,"meses_historico": 12},
            {"tipo": "corto_medio", "cv_threshold": 0.95,"min_compras": 5,"max_compras_recientes": 15,"meses_historico": 12},
            {"tipo": "mediano", "cv_threshold": 0.6,"min_compras": 4,"max_compras_recientes": 10,"meses_historico": 18},
            {"tipo": "largo", "cv_threshold": 0.45,"min_compras": 4,"max_compras_recientes": 10,"meses_historico": 36}
        ]
        resultado = {"ciclo_dias": [0,0,0], "cv": 999, "tipo": "no_ciclico", "razon": "sin_gaps"}

        # FASE 1: Intentar ciclos cortos
        ciclo_encontrado = False
        for ciclo in ciclos_clase:
            resultado = calcular_ciclos(
            df_ventas=df_ventas,
            familia_id=familia_id,
            meses_historico=ciclo["meses_historico"],
            min_compras=ciclo["min_compras"],
            max_compras_recientes=ciclo["max_compras_recientes"],
            periodo_dias=5 if ciclo["tipo"] == "corto" else 7,
            subcat=subcat,
            today=today,
            cv_threshold=ciclo["cv_threshold"],
            tipo=ciclo["tipo"],
            clase_de_calculo = clase_de_calculo
            )
        
            if resultado["ciclo_dias"][1] > 0 and resultado["tipo"] != "no_ciclico":
                # Encontró ciclo
                ciclo_encontrado = True
                resultados.append({
                "CODIGO_FAMILIA": familia_id,
                "COD_SUBCATEGORIA": subcat,
                "ciclo_dias": resultado["ciclo_dias"],
                "cv": resultado["cv"],
                "tipo_ciclo": resultado["tipo"],
                "razon": resultado.get("razon", "NO_ENCONTRADO"),
                "gaps_originales_dias": resultado.get("gaps_originales", []),
                "gaps_normalizados": resultado.get("gaps_normalizados", []),
                "gaps_ciclos_bloques": resultado.get("gaps_ciclos_bloques", [])
                
                })
                break
        
        # Si no encontró ningún ciclo, agregar como no_ciclico
        if not ciclo_encontrado:
            resultados.append({
                "CODIGO_FAMILIA": familia_id,
                "COD_SUBCATEGORIA": subcat,
                "ciclo_dias": resultado["ciclo_dias"],
                "cv": resultado["cv"],
                "tipo_ciclo": resultado["tipo"],
                "razon": resultado.get("razon", "NO_ENCONTRADO"),
                "gaps_originales_dias": resultado.get("gaps_originales", []),
                "gaps_normalizados": resultado.get("gaps_normalizados", []),
                "gaps_ciclos_bloques": resultado.get("gaps_ciclos_bloques", [])
            })
      
    df_resultado = pd.DataFrame(resultados)
    
    # Ordenar: cíclicos primero, por CV ascendente
    if not df_resultado.empty:
        # Ordenar por ciclo_dias[1] y cv
        df_resultado['_ciclo_idx1'] = df_resultado['ciclo_dias'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else 0)
        df_resultado = df_resultado.sort_values(["_ciclo_idx1", "cv"], ascending=[False, True]).reset_index(drop=True)
        df_resultado = df_resultado.drop(columns=['_ciclo_idx1'])
    
    if clase_de_calculo == 0:
        columnas_debug = ["COD_SUBCATEGORIA","tipo_ciclo"]
        df_resultado = df_resultado[columnas_debug].copy()
        #rename tipo_ciclo a tipo_ciclo_b
        df_resultado.rename(columns={"tipo_ciclo": "tipo_ciclo_b"}, inplace=True)
        #agregar columna ciclo_binario
        df_resultado["ciclo_binario"] = df_resultado["tipo_ciclo_b"].apply(lambda x: 1 if x != "no_ciclico" else 0)
    if clase_de_calculo == 1:
        df_resultado["ciclo_binario_c"] = df_resultado["tipo_ciclo"].apply(lambda x: 1 if x != "no_ciclico" else 0)

    return df_resultado
    

def compute_recency_features(subcat_agg: pd.DataFrame,
                             ciclos_estacionales: pd.DataFrame,
                             fecha_corte: pd.Timestamp) -> pd.DataFrame:
    subcat_agg = subcat_agg.copy()

    # días desde última compra
    subcat_agg["dias_desde_ultima_compra"] = (
        fecha_corte - subcat_agg["ultima_compra"]
    ).dt.days.clip(lower=0)

    # pegar ciclo_dias y tipo_ciclo
    subcat_agg = subcat_agg.merge(ciclos_estacionales, on="COD_SUBCATEGORIA", how="left")

    # ---- Extraer [inferior, mu, superior] por fila (NO por índice global) ----
    ciclos = subcat_agg["ciclo_dias"].apply(
        lambda v: v if isinstance(v, (list, tuple)) and len(v) == 3 else [0.0, 0.0, 0.0]
    )
    ciclos_df = pd.DataFrame(ciclos.tolist(), columns=["ciclo_inf", "ciclo_mu", "ciclo_sup"], index=subcat_agg.index)
    subcat_agg = pd.concat([subcat_agg, ciclos_df], axis=1)

    # ---- Parámetros (ajústalos si quieres) ----
    t_edge = 0.95         # f(inferior)=f(superior)=0.95 (>=0.9)
    y_left_far = 0.20     # f(inferior/2)=0.2  (izq)
    y_right_far = 0.10    # f(mu+5*(sup-mu))=0.1 (der) -> cola que sí cae
    k_right = 5.0

    # ---- Construir puntos extra derivados (no hardcode de 30/200) ----
    left_far  = subcat_agg["ciclo_inf"] / 2.0
    right_far = subcat_agg["ciclo_mu"] + k_right * (subcat_agg["ciclo_sup"] - subcat_agg["ciclo_mu"])

    # ---- Distancias al centro ----
    mu = subcat_agg["ciclo_mu"].to_numpy(dtype=float)
    x1 = subcat_agg["ciclo_inf"].to_numpy(dtype=float)
    x2 = subcat_agg["ciclo_sup"].to_numpy(dtype=float)
    lf = left_far.to_numpy(dtype=float)
    rf = right_far.to_numpy(dtype=float)

    dL_edge = mu - x1
    dL_far  = mu - lf
    dR_edge = x2 - mu
    dR_far  = rf - mu

    # ---- Máscara de casos válidos: CICLOS y geometría correcta ----
    # IMPORTANTE: Usar tipo_ciclo para filtrar no_ciclicos
    is_ciclico = subcat_agg["tipo_ciclo"] != "no_ciclico"
    valid = is_ciclico & (mu > 0) & (dL_edge > 0) & (dL_far > dL_edge) & (dR_edge > 0) & (dR_far > dR_edge)

    # ---- Calibración vectorizada de p y s por lado ----
    # Modelo: y = exp(-(d/s)^p)
    # p = ln( ln(1/t_edge)/ln(1/t_far) ) / ln(d_edge/d_far)
    # s = d_edge / (ln(1/t_edge)^(1/p))
    pL = np.zeros_like(mu)
    sL = np.ones_like(mu)
    pR = np.zeros_like(mu)
    sR = np.ones_like(mu)

    # precomputar constantes
    ln1_te = np.log(1.0 / t_edge)
    ln1_tL = np.log(1.0 / y_left_far)
    ln1_tR = np.log(1.0 / y_right_far)

    # ojo: esto da p positivo porque numerador y denominador son negativos
    pL[valid] = np.log(ln1_te / ln1_tL) / np.log(dL_edge[valid] / dL_far[valid])
    sL[valid] = dL_edge[valid] / (ln1_te ** (1.0 / pL[valid]))

    pR[valid] = np.log(ln1_te / ln1_tR) / np.log(dR_edge[valid] / dR_far[valid])
    sR[valid] = dR_edge[valid] / (ln1_te ** (1.0 / pR[valid]))

    # ---- Evaluación del score recencia (x = dias_desde_ultima_compra) ----
    x = subcat_agg["dias_desde_ultima_compra"].to_numpy(dtype=float)

    score = np.zeros_like(x, dtype=float)
    left_side = x <= mu

    # evitar divisiones raras
    dist = np.zeros_like(x, dtype=float)
    p    = np.zeros_like(x, dtype=float)

    dist[valid & left_side]  = (mu[valid & left_side] - x[valid & left_side]) / sL[valid & left_side]
    p[valid & left_side]     = pL[valid & left_side]

    dist[valid & (~left_side)] = (x[valid & (~left_side)] - mu[valid & (~left_side)]) / sR[valid & (~left_side)]
    p[valid & (~left_side)]    = pR[valid & (~left_side)]

    # score = exp(-(dist^p))
    score[valid] = np.exp(-np.power(dist[valid], p[valid]))

    # ---- Guardar features compatibles con tu pipeline ----
    subcat_agg["l_compra_sobre_ciclo"] = 0.0
    subcat_agg.loc[valid, "l_compra_sobre_ciclo"] = x[valid] / mu[valid]

    subcat_agg["castigo_recencia"] = 1.0  # ya está “incluido” en la forma del score
    subcat_agg["recencia_hl"] = score     # tu recencia principal (0..1)

    # Si quieres mantener "recencia" por compatibilidad:
    subcat_agg["recencia"] = subcat_agg["recencia_hl"]

    return subcat_agg[[
        "COD_SUBCATEGORIA",
        "recencia_hl",
        "castigo_recencia",
        "l_compra_sobre_ciclo",
        "dias_desde_ultima_compra",
        "recencia"
    ]]


def compute_frequency_features(df_family: pd.DataFrame, ciclos_estacionales: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.DataFrame:
    """
    Calcula features de frecuencia binarias (0 o 1) en 3 niveles.
    Retorna [freq_baja, freq_media, freq_alta] basado en [ciclo_inf, ciclo_mu, ciclo_sup].
    
    Períodos de revisión por tipo:
    - corto: 90 días
    - corto_medio: 225 días
    - mediano: 450 días
    - largo: 900 días
    
    Lógica:
    - compras_esperadas = periodo_revision / ciclo_dias (redondeado)
    - Si compras_reales > compras_esperadas → 0 (sobrepasó, mala frecuencia)
    - Si compras_reales <= compras_esperadas → 1 (dentro del rango, buena frecuencia)
    """
    resultados = []
    
    # Mapeo de tipo de ciclo a período de revisión (días)
    periodos_revision = {
        "corto": 90,
        "corto_medio": 225,
        "mediano": 450,
        "largo": 900
    }
    
    for _, row in ciclos_estacionales.iterrows():
        subcat = row["COD_SUBCATEGORIA"]
        ciclo_dias = row["ciclo_dias"]
        tipo_ciclo = row["tipo_ciclo"]  # Si falla, hay un error arriba

        # No_ciclicos: poner valores en 0 (no hay frecuencia válida)
        if tipo_ciclo == "no_ciclico":
            resultados.append({
                "COD_SUBCATEGORIA": subcat,
                "freq_baja": 0,
                "freq_media": 0,
                "freq_alta": 0,
                "cv_invertido": 0.0,  # NO calcular para no_ciclico
                "compras_reales": 0,
                "periodo_revision": 0
            })
            continue
        
        # Extraer [ciclo_inf, ciclo_mu, ciclo_sup] - siempre es lista de 3
        ciclo_inf, ciclo_mu, ciclo_sup = ciclo_dias
        
        # Calcular CV invertido para cíclicos
        cv_original = row["cv"]
        cv_invertido = 1.0 - min(cv_original, 1.0)
        
        # Determinar período de revisión según tipo - debe existir
        periodo_dias = periodos_revision[tipo_ciclo]  # Si falla, tipo_ciclo inválido
        
        # Calcular ventana de tiempo
        ventana_inicio = fecha_corte - pd.Timedelta(days=periodo_dias)
        recientes = df_family[
            (df_family["COD_SUBCATEGORIA"] == subcat) &
            (df_family["DIM_PERIODO"] >= ventana_inicio)
        ].copy()
        
        compras_reales = recientes["DIM_PERIODO"].unique().size
        
        # Calcular frecuencia para cada nivel
        freq_scores = []
        for ciclo_val in [ciclo_inf, ciclo_mu, ciclo_sup]:
            # Compras esperadas (redondeado)
            compras_esperadas = round(periodo_dias / ciclo_val)
            
            # Binario: 1 si NO sobrepasa, 0 si sobrepasa
            freq_score = 1 if compras_reales <= compras_esperadas else 0
            freq_scores.append(freq_score)
        
        
        
        resultados.append({
            "COD_SUBCATEGORIA": subcat,
            "freq_baja": freq_scores[0],      # Binario: frecuencia con ciclo inferior
            "freq_media": freq_scores[1],     # Binario: frecuencia con ciclo medio
            "freq_alta": freq_scores[2],      # Binario: frecuencia con ciclo superior
            "cv_invertido": cv_invertido,     # 1-CV: mayor valor = más estable
            "compras_reales": compras_reales,
            "periodo_revision": periodo_dias
        })
    
    return pd.DataFrame(resultados)


def compute_sow_features(df_family: pd.DataFrame, ciclos_estacionales: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.DataFrame:
    """
    Calcula Share of Wallet con pesos adaptativos según tipo de ciclo.
    - Ciclos cortos: peso_12m=7, peso_24m=3 (sesgo reciente)
    - Ciclos largos: peso_12m=1, peso_24m=1 (sin sesgo)
    """
    ventana_12m_inicio = fecha_corte - pd.DateOffset(months=SOW_MONTHS_12)
    ventana_24m_inicio = fecha_corte - pd.DateOffset(months=SOW_MONTHS_24)

    # Últimos 12 meses
    historicos_12m = df_family[df_family["DIM_PERIODO"] >= ventana_12m_inicio]

    # De 24 a 12 meses atrás
    historicos_24m = df_family[
        (df_family["DIM_PERIODO"] >= ventana_24m_inicio)
        & (df_family["DIM_PERIODO"] < ventana_12m_inicio)
    ]

    # --- 12 meses: transacciones por subcategoría ---
    sow_agg_12m = (
        historicos_12m.groupby("COD_SUBCATEGORIA")
        .agg(transacciones_netas=("DIM_FACTURA", "count"))
        .reset_index()
    )

    # --- 24–12 meses: transacciones por subcategoría ---
    sow_agg_24m = (
        historicos_24m.groupby("COD_SUBCATEGORIA")
        .agg(transacciones_netas=("DIM_FACTURA", "count"))
        .reset_index()
    )
    
    # Crear mapa de tipo de ciclo
    tipo_ciclo_map = ciclos_estacionales.set_index("COD_SUBCATEGORIA")["tipo_ciclo"].to_dict()
    
    # Aplicar pesos según tipo de ciclo
    def aplicar_pesos(row, periodo):
        subcat = row["COD_SUBCATEGORIA"]
        tipo = tipo_ciclo_map.get(subcat, "no_ciclico")
        
        if tipo == "largo":
            # Pesos iguales para ciclos largos
            peso = 1
        else:
            # Sesgo reciente para ciclos cortos
            peso = 7 if periodo == "12m" else 3
        
        return row["transacciones_netas"] * peso
    
    sow_agg_12m["transacciones_netas"] = sow_agg_12m.apply(lambda r: aplicar_pesos(r, "12m"), axis=1)
    sow_agg_24m["transacciones_netas"] = sow_agg_24m.apply(lambda r: aplicar_pesos(r, "24m"), axis=1)

    # Unión vertical (no merge)
    sow_agg = pd.concat([sow_agg_12m, sow_agg_24m], ignore_index=True)

    if sow_agg.empty:
        # Si no hay datos en ninguna ventana, devolver todo en 0
        return pd.DataFrame(
            {
                "COD_SUBCATEGORIA": df_family["COD_SUBCATEGORIA"].unique(),
                "sow_24m": 0.0,
            }
        )

    # Sumar pesos por subcategoría
    sow_agg = (
        sow_agg.groupby("COD_SUBCATEGORIA")
        .agg(transacciones_netas=("transacciones_netas", "sum"))
        .reset_index()
    )

    total_transacciones = sow_agg["transacciones_netas"].sum()

    if total_transacciones <= 0:
        sow_agg["sow_24m"] = 0.0
    else:
        sow_agg["sow_24m"] = sow_agg["transacciones_netas"] / total_transacciones

    max_sow_24m = sow_agg["sow_24m"].max()
    mutiplicador = 1 / max_sow_24m
    sow_agg["sow_24m"] = sow_agg["sow_24m"] * mutiplicador

    return sow_agg[["COD_SUBCATEGORIA", "sow_24m","transacciones_netas"]]


def _discretize_series(series, umbral):
    """
    Convierte una serie de transacciones en una serie binaria (0 o 1)
    usando la suma de meses adyacentes.
    """
    discretized = np.zeros(len(series), dtype=int)
    for i in range(len(series) - 1):
        # Suma el mes actual y el siguiente
        if series[i] > 0 and series[i] + series[i+1] >= umbral:
            discretized[i] = 1
            #print(f"Mes {i}: {series[i]} + {series[i+1]} = {series[i] + series[i+1]} >= {umbral}")
    
    # El último punto se compara solo
    if series[-1] >= umbral:
        discretized[-1] = 1
        
    return discretized

def _clean_consecutive_ones(series):
    """
    """
    cleaned = np.copy(series)
    for i in range(len(cleaned) - 1):
        if cleaned[i] == 1 and cleaned[i+1] == 1:
            cleaned[i+1] = 0
    return cleaned

def _calculate_gaps(series,min_picos):
    indices_picos = []
    if np.sum(series == 1) >= min_picos: 
        contador = 0
        for index,i in enumerate(series):
            if i == 1:
                if index !=0:
                    indices_picos.append(contador)
                    contador = 0
            else:
                contador += 1
            
            if index == len(series)-1 and i== 0:
                indices_picos.append(contador)
    
    return indices_picos

def _detectar_estacionalidad(
    df_ventas: pd.DataFrame,
    historico_ventas: int = 12,
    today: pd.Timestamp = pd.Timestamp.today().normalize(),
    min_picos: int = 2
):
    """
    Detecta si una subcategoría tiene un patrón de compra estacional.
    Devuelve un puntaje de estacionalidad (0 a 1) y los gaps entre picos.
    """
    # 1. Crear la ventana de historico_vetas
    today = pd.to_datetime(today).normalize()
    # 13 meses hacia atrás desde el inicio del mes actual
    start_date_hist = (today.replace(day=1) - pd.DateOffset(months=historico_ventas))
    end_date_hist = today
    full_date_range = pd.date_range(start=start_date_hist, end=end_date_hist, freq='MS')
    df_ventas = df_ventas[
        (df_ventas["DIM_PERIODO"] >= start_date_hist) &
        (df_ventas["DIM_PERIODO"] <= end_date_hist)
    ]

    df_ventas["mes_anio"] = df_ventas["DIM_PERIODO"].dt.strftime("%Y-%m")

    df_vetas_g = df_ventas.groupby(["COD_SUBCATEGORIA","mes_anio"]).agg(
        ventas=("DIM_FACTURA", "count")
    ).reset_index()

    #build the serie 
    df_date_full = pd.DataFrame(
        {
            "mes_anio": full_date_range.strftime("%Y-%m"),
            "key": 1
        }
    )
    df_categorias_f = pd.DataFrame(df_vetas_g["COD_SUBCATEGORIA"].unique(), columns=["COD_SUBCATEGORIA"])
    df_categorias_f["key"] = 1
    df_date_full = df_date_full.merge(df_categorias_f, on="key", how="outer").drop("key", axis=1)
    
    df_date_full = df_date_full.merge(df_vetas_g, on=["COD_SUBCATEGORIA", "mes_anio"], how="left")
    df_date_full["ventas"] = df_date_full["ventas"].fillna(0)
    df_date_full = df_date_full[["COD_SUBCATEGORIA", "mes_anio", "ventas"]]

    # El umbral es la media de transacciones de los meses que SÍ tuvieron compras
    df_agg = df_date_full.groupby("COD_SUBCATEGORIA").agg(
        umbral=("ventas", lambda x: math.floor(x[x>0].mean()) if (x>0).any() else 0),
        serie = ("ventas",lambda x: x.tolist())
    ).reset_index()
    
    df_agg["serie_binaria"] = df_agg.apply(lambda x: _discretize_series(x["serie"], x["umbral"]), axis=1)
    df_agg["serie_limpia"] = df_agg.apply(lambda x: _clean_consecutive_ones(x["serie_binaria"]), axis=1)
    
    # 5. Calcular gaps y puntaje de estacionalidad
    df_agg["indices_picos"] = df_agg.apply(lambda x: _calculate_gaps(x["serie_limpia"],min_picos), axis=1)

    df_agg['CV'] = df_agg.apply(lambda x: np.std(x['indices_picos']) / np.mean(x['indices_picos']) if len(x['indices_picos']) > 1 else 0.85, axis=1)
    df_agg["mean_indices_picos"] = df_agg.apply(lambda x: np.mean(x['indices_picos']) if len(x['indices_picos']) > 1 else 0.85, axis=1)
    df_agg["std_indices_picos"] = df_agg.apply(lambda x: np.std(x['indices_picos']) if len(x['indices_picos']) > 1 else 0.85, axis=1)

    a = 7.85  # pendiente
    b = 0.6   # CV donde score = 0.5
    a = -1.85
    b = 2.5

    df_agg['puntaje'] = df_agg.apply(lambda x: np.exp(a * np.power(x['CV'],b)), axis=1)


    return df_agg[["COD_SUBCATEGORIA", "puntaje","umbral","serie","serie_binaria","serie_limpia","indices_picos","CV","mean_indices_picos","std_indices_picos"]]


def compute_seasonality_features(df_family: pd.DataFrame, ciclos_estacionales: pd.DataFrame, fecha_corte: pd.Timestamp) -> pd.DataFrame:
    """
    Calcula feature de comparación temporal según tipo de ciclo.
    
    Períodos por tipo:
    - corto: último 30d vs días 30-90 atrás
    - corto_medio: últimos 60d vs días 60-150 atrás
    - mediano: últimos 90d vs días 365-275 atrás (3 meses año pasado)
    - largo: últimos 90d vs días 730-640 atrás (3 meses hace 2 años)
    
    Función con meseta (plateau):
    - ratio [0.8, 1.2]: score alto (meseta)
    - fuera del rango: decae exponencial
    """
    
    # Configuración de períodos por tipo de ciclo
    config_periodos = {
        "corto": {
            "dias_actual": 30,
            "inicio_pasado": 30,
            "fin_pasado": 90
        },
        "corto_medio": {
            "dias_actual": 60,
            "inicio_pasado": 60,
            "fin_pasado": 150
        },
        "mediano": {
            "dias_actual": 90,
            "inicio_pasado": 275,
            "fin_pasado": 365
        },
        "largo": {
            "dias_actual": 90,
            "inicio_pasado": 640,
            "fin_pasado": 730
        }
    }
    
    tipo_ciclo_map = ciclos_estacionales.set_index("COD_SUBCATEGORIA")["tipo_ciclo"].to_dict()
    resultados = []
    
    for subcat in df_family["COD_SUBCATEGORIA"].unique():
        tipo = tipo_ciclo_map[subcat]  # Si falla, error en pipeline anterior
        
        # No_ciclicos: poner valores en 0 (no hay estacionalidad válida)
        if tipo == "no_ciclico":
            resultados.append({
                "COD_SUBCATEGORIA": subcat,
                "season_ratio": 0.0,
                "compras_actual": 0,
                "compras_pasado": 0,
                "ratio_temporal": 0.0
            })
            continue
        
        config = config_periodos[tipo]
        
        # Ventana actual
        inicio_actual = fecha_corte - pd.Timedelta(days=config["dias_actual"])
        mask_actual = (
            (df_family["COD_SUBCATEGORIA"] == subcat) &
            (df_family["DIM_PERIODO"] > inicio_actual) & 
            (df_family["DIM_PERIODO"] <= fecha_corte)
        )
        
        # Ventana pasada
        inicio_pasado = fecha_corte - pd.Timedelta(days=config["fin_pasado"])
        fin_pasado = fecha_corte - pd.Timedelta(days=config["inicio_pasado"])
        mask_pasado = (
            (df_family["COD_SUBCATEGORIA"] == subcat) &
            (df_family["DIM_PERIODO"] > inicio_pasado) & 
            (df_family["DIM_PERIODO"] <= fin_pasado)
        )
        
        # Contar facturas únicas
        compras_actual = df_family[mask_actual]["DIM_FACTURA"].nunique()
        compras_pasado = df_family[mask_pasado]["DIM_FACTURA"].nunique()
        
        # Calcular ratio
        if compras_pasado > 0:
            ratio = compras_actual / compras_pasado
        else:
            ratio = 999
        
        # Función con meseta (plateau)
        # Meseta alta entre 0.8-1.2, decae fuera del rango
        if 0.8 <= ratio <= 1.2:
            # Dentro del rango óptimo
            score = 0.95
        elif ratio < 0.8:
            # Compraste menos de lo esperado → score tiende a 1.0
            # Decaimiento suave: score aumenta cuando ratio baja
            score = 0.95 + (0.8 - ratio) * 0.05  # Sube hasta 1.0
            score = min(score, 1.0)
        else:  # ratio > 1.2
            # Compraste más de lo esperado → decae exponencial
            exceso = ratio - 1.2
            score = 0.95 * np.exp(-exceso * 0.5)  # Decae suave
        
        resultados.append({
            "COD_SUBCATEGORIA": subcat,
            "season_ratio": score,
            "compras_actual": compras_actual,
            "compras_pasado": compras_pasado,
            "ratio_temporal": ratio
        })
    
    return pd.DataFrame(resultados)


def compute_features_for_family(
    df_family: pd.DataFrame,   # es el dataset de ventas filtrado por la familia sin group by de nada
    family_code: int,
    fecha_corte: pd.Timestamp = None
) -> pd.DataFrame:
    """
    Calcula features para una familia.
    
    Args:
        df_family: DataFrame con ventas de la familia
        family_code: Código de la familia
        fecha_corte: Fecha de corte para cálculo de features (default: CUTOFF_DATE global)
    """
    if fecha_corte is None:
        fecha_corte = CUTOFF_DATE
    
    df_family = df_family[df_family["DIM_PERIODO"] <= fecha_corte].copy()
    if df_family.empty:
        return pd.DataFrame()

    subcat_agg = (
        df_family.groupby("COD_SUBCATEGORIA")
        .agg(
            total_cantidad=("CANTIDAD_SUELTA", "sum"),
            total_venta_neta=("VENTA_NETA", "sum"),
            total_descuento=("DESCUENTO", "sum"),
            total_pvp=("PVP", "sum"),
            promedio_pvp=("PVP", "mean"),
            facturas_unicas=("DIM_FACTURA", "nunique"),
            registros=("DIM_FACTURA", "count"),
            primera_compra=("DIM_PERIODO", "min"),
            ultima_compra=("DIM_PERIODO", "max"),
        )
        .reset_index()
    )

    if subcat_agg.empty:
        return pd.DataFrame()

    subcat_agg = subcat_agg.sort_values("total_venta_neta", ascending=False).reset_index(drop=True)
    
    # Calcular ciclos 
    ciclos_estacionales = calcular_ciclos_por_bloques(
        df_ventas=df_family,
        familia_id=family_code,
        today=fecha_corte,
        clase_de_calculo =1
    )

    # Calculo ciclos Debug
    
    ciclos_debug = calcular_ciclos_por_bloques(
        df_ventas=df_family,
        familia_id=family_code,
        today=fecha_corte,
        clase_de_calculo = 0
    )
    
    
    if ciclos_estacionales.empty:
        return pd.DataFrame()

    # Calcular features de recencia
    recency_features = compute_recency_features(
        subcat_agg=subcat_agg,
        ciclos_estacionales=ciclos_estacionales,
        fecha_corte=fecha_corte
    )
    
    # Calcular features de frecuencia (con ventanas adaptativas)
    freq_features = compute_frequency_features(
        df_family=df_family,
        ciclos_estacionales=ciclos_estacionales,
        fecha_corte=fecha_corte
    )
    features_final = recency_features.merge(freq_features, on="COD_SUBCATEGORIA", how="left")
    
    # Calcular SOW (con pesos adaptativos)
    sow_features = compute_sow_features(
        df_family=df_family,
        ciclos_estacionales=ciclos_estacionales,
        fecha_corte=fecha_corte
    )
    features_final = features_final.merge(sow_features, on="COD_SUBCATEGORIA", how="left")
    
    # Calcular estacionalidad (con ventanas adaptativas)
    seasonality_features = compute_seasonality_features(
        df_family=df_family,
        ciclos_estacionales=ciclos_estacionales,
        fecha_corte=fecha_corte
    )
    features_final = features_final.merge(seasonality_features, on="COD_SUBCATEGORIA", how="left")

    features_final = features_final.merge(ciclos_estacionales, on="COD_SUBCATEGORIA", how="left")
    features_final  = features_final.merge(ciclos_debug, on="COD_SUBCATEGORIA",how="left")
    features_final = features_final.fillna(0.0)

    # Score final (legacy - ahora el modelo FNN usará las features directamente)
    # Usando freq_media como representante de frecuencia para score_final
    features_final["score_final"] = (
        0.4 * features_final["recencia_hl"]
        + 0.3 * features_final["freq_media"]
        + 0.1 * features_final["sow_24m"]
        + 0.2 * features_final["season_ratio"]
    )

    score_columns = [
        "COD_SUBCATEGORIA",
        "recencia_hl",
        "freq_baja",
        "freq_media",
        "freq_alta",
        "cv_invertido",
        "sow_24m",
        "season_ratio",
        "score_final",
    ]

    # --- Renombrado de columnas por origen ---
    rename_map = {}

    def add_renames(cols, prefix):
        for col in cols:
            if col in features_final.columns and col not in score_columns:
                rename_map[col] = f"{prefix}_{col}"
    
    # Definir columnas por origen
    columnas_debug_ciclos = ['tipo_ciclo_b','ciclo_binario']
    columnas_ciclos = ["CODIGO_FAMILIA", "COD_SUBCATEGORIA", "ciclo_dias", "cv", "tipo_ciclo","razon", "gaps_originales_dias", "gaps_normalizados","gaps_ciclos_bloques","ciclo_binario_c"]
    columnas_recencia = ["COD_SUBCATEGORIA", "recencia_hl", "castigo_recencia", "l_compra_sobre_ciclo", "dias_desde_ultima_compra", "recencia"]
    columnas_freq = ["COD_SUBCATEGORIA", "freq_baja", "freq_media", "freq_alta", "cv_invertido", "compras_reales", "periodo_revision"]
    columnas_sow = ["COD_SUBCATEGORIA", "sow_24m", "transacciones_netas"]
    columnas_seasonality = ["COD_SUBCATEGORIA", "season_ratio", "compras_actual", "compras_pasado", "ratio_temporal"]

    add_renames(columnas_debug_ciclos, "Debug_ciclos")
    add_renames(columnas_ciclos, "Ciclos")
    add_renames(columnas_recencia, "Recencia")
    add_renames(columnas_freq, "Freq")
    add_renames(columnas_sow, "Sow")
    add_renames(columnas_seasonality, "Seasonality")

    features_final = features_final.rename(columns=rename_map)

    # Construir lista final de columnas: score_columns + las renombradas
    final_cols = list(score_columns) + list(rename_map.values())
    # Eliminar duplicados manteniendo orden
    final_cols = list(dict.fromkeys(final_cols))
    # Filtrar solo existentes
    final_cols = [c for c in final_cols if c in features_final.columns]

    features_final = features_final[final_cols]
    features_final["nucleo"] = family_code
    
    # Extraer valor medio del ciclo (optimizado con vectorización)
    if 'Ciclos_ciclo_dias' in features_final.columns:
        features_final['ciclo_dias_mu'] = features_final['Ciclos_ciclo_dias'].apply(
            lambda x: float(x[1]) if isinstance(x, (list, np.ndarray)) and len(x) >= 2 else 0.0
        )
    else:
        features_final['ciclo_dias_mu'] = 0.0

    return features_final

def process_all_families(
    df_raw: pd.DataFrame,
    families: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    if families is None:
        families_iter = sorted(df_raw["CODIGO_FAMILIA"].unique())
    else:
        family_set = {int(fam) for fam in families}
        families_iter = [fam for fam in sorted(df_raw["CODIGO_FAMILIA"].unique()) if fam in family_set]

    results: List[pd.DataFrame] = []
    for family_code in families_iter:
        df_family = df_raw[df_raw["CODIGO_FAMILIA"] == family_code].copy()
        features_family = compute_features_for_family(df_family, family_code)
        if features_family.empty:
            continue
        results.append(features_family)
        print(f"[FeatureEngineering] Núcleo {family_code}: {len(features_family)} subcategorías procesadas")

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def main() -> None:
    print("Cargando dataset histórico...")
    df_raw = load_historical_dataset(HISTORICAL_FILE)
    print(f"Registros totales: {len(df_raw)} | Núcleos únicos: {df_raw['CODIGO_FAMILIA'].nunique()}")



    features_all = process_all_families(df_raw)
    if features_all.empty:
        print("No se generaron features para ningún núcleo.")
        return

    features_all.to_csv(OUTPUT_FILE, index=False)
    print(f"Archivo consolidado guardado en: {OUTPUT_FILE}")

    return features_all,df_raw


if __name__ == "__main__":
    features_all,df_raw = main()
    print("finalizado proceso")
    
    # %%
    # Post-procesamiento - Comentado para permitir importación limpia del módulo
    """
    columnas_subcategrio = ["COD_SUBCATEGORIA","NOMBRE_SUBCATEGORIA"]
    df_raw_subcat = df_raw[columnas_subcategrio].drop_duplicates()
    
    features_all_subcat = features_all.merge(df_raw_subcat, on="COD_SUBCATEGORIA", how="left")
    columnas_finales = [
    'nucleo',                        # CODIGO FAMILIA
    'COD_SUBCATEGORIA',              # COD SUBCATEGORIA
    'NOMBRE_SUBCATEGORIA',           # NOMBRE SUBCATEGORIA
    'recencia_hl',                   # RECENCIA
    'freq_score',                    # FRECUENCIA
    'sow_24m',                       # SOW
    'season_ratio',                  # ESTACIONALIDAD
    'score_final',                   # SCORE SUBCATEGORIA
    'Ciclos_ciclo_dias',             # CICLO
    'Recencia_dias_desde_ultima_compra', # DIAS ULTIMA COMPRA
    'Seasonality_mean_indices_picos',# MEDIA ESTACION
    'Seasonality_std_indices_picos'  # STD DEV ESTACION
]
features_all_subcat = features_all_subcat[columnas_finales]

features_all_subcat = features_all_subcat.sort_values(by="score_final", ascending=False).reset_index(drop=True).reset_index()
#save csv 
#remane index to top
#reamne columns 
'''
    'nucleo',                        # CODIGO FAMILIA
    'COD_SUBCATEGORIA',              # COD SUBCATEGORIA
    'NOMBRE_SUBCATEGORIA',           # NOMBRE SUBCATEGORIA
    'recencia_hl',                   # RECENCIA
    'freq_score',                    # FRECUENCIA
    'sow_24m',                       # SOW
    'season_ratio',                  # ESTACIONALIDAD
    'score_final',                   # SCORE SUBCATEGORIA
    'Ciclos_ciclo_dias',             # CICLO
    'Recencia_dias_desde_ultima_compra', # DIAS ULTIMA COMPRA
    'Seasonality_mean_indices_picos',# MEDIA ESTACION
    'Seasonality_std_indices_picos'  # STD DEV ESTACION
'''
features_all_subcat.rename(columns={"index": "top",
"nucleo":"CODIGO_FAMILIA",
"COD_SUBCATEGORIA":"COD_SUBCATEGORIA",
"NOMBRE_SUBCATEGORIA":"NOMBRE_SUBCATEGORIA",
"recencia_hl":"RECENCIA",
"freq_score":"FRECUENCIA",
"sow_24m":"SOW",
"season_ratio":"ESTACIONALIDAD",
"score_final":"SCORE_SUBCATEGORIA",
"Ciclos_ciclo_dias":"CICLO",
"Recencia_dias_desde_ultima_compra":"DIAS_ULTIMA_COMPRA",
"Seasonality_mean_indices_picos":"MEDIA_ESTACION",
"Seasonality_std_indices_picos":"STD_DEV_ESTACION"}, inplace=True)



features_all_subcat.to_csv("features_with_subcat_names.csv", index=False)
#save in excel format
features_all_subcat.to_excel("features_with_subcat_names.xlsx", index=False)

#%%

#read item.xlsx
item = pd.read_excel("item.xlsx")
# %%
nombre_divicion = ["FARMA","CONSUMO"]
item = item[item['NOMBRE_DIVISION_COMERCIAL'].isin(nombre_divicion)]
#DROP DUPLICADOS 
colums_unicas = ['COD_ITEM', 'COD_SUBCATEGORIA','COD_DIVISION_COMERCIAL']
item = item.drop_duplicates(subset=colums_unicas).reset_index(drop=True)

columnas_uso = ['COD_SUBCATEGORIA','COD_DIVISION_COMERCIAL','NOMBRE_DIVISION_COMERCIAL']
item_merge = item[columnas_uso].drop_duplicates(subset=['COD_SUBCATEGORIA']).copy()
features_all_subcat = features_all_subcat.merge(item_merge, on="COD_SUBCATEGORIA", how="left")

# --- FARMA ---
feature_farma = features_all_subcat[features_all_subcat['NOMBRE_DIVISION_COMERCIAL'] == 'FARMA'].copy()
# Ordenar por nucleo y score para ranking correcto por familia
feature_farma = feature_farma.sort_values(by=["CODIGO_FAMILIA", "SCORE_SUBCATEGORIA"], ascending=[True, False])
# Ranking por nucleo
feature_farma['top_s'] = feature_farma.groupby("CODIGO_FAMILIA").cumcount() + 1

# --- CONSUMO ---
feature_consumo = features_all_subcat[features_all_subcat['NOMBRE_DIVISION_COMERCIAL'] == 'CONSUMO'].copy()
# Ordenar por nucleo y score para ranking correcto por familia
feature_consumo = feature_consumo.sort_values(by=["CODIGO_FAMILIA", "SCORE_SUBCATEGORIA"], ascending=[True, False])
# Ranking por nucleo
feature_consumo['top_s'] = feature_consumo.groupby("CODIGO_FAMILIA").cumcount() + 1

    feature_concat = pd.concat([feature_farma, feature_consumo], ignore_index=True)
    features_all_subcat = feature_concat.copy()
    features_all_subcat.to_excel("features_with_subcat_names.xlsx", index=False)
    """
