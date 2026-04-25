"""
=============================================================================
COMPARACIÓN FINAL: LINEAR vs FNN
=============================================================================

Compara modelo lineal (Excel) vs FNN entrenado

CONFIGURACIÓN:
- Cambia FECHA_MODELO para usar diferentes modelos FNN
- '2025-11-09': Modelo conservador (mejor en diciembre)
- '2025-11-30': Modelo con toda la info

El script:
1. Carga predictions del Linear (del Excel)
2. Calcula features hasta FECHA_MODELO para FNN
3. Carga modelo FNN entrenado
4. Compara ambos vs data_test.csv (Dic 1-9)
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys
import os
from contextlib import contextmanager

from tensorflow import keras
import warnings

# Ignorar warnings de pandas para una salida limpia
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning, message='.*incompatible dtype.*')

# Importar feature engineering
sys.path.append(str(Path(__file__).parent.parent))
from feature_engineering_batch import compute_features_for_family, load_historical_dataset


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

FECHA_MODELO = '2025-11-09'      # ← Carga model_1109.h5 (recién entrenado con 7 features)
FECHA_EVALUACION = '2025-11-30'  # ← Features para evaluar (por el momento siempre Nov 30)



@contextmanager
def suppress_stdout():
    """Redirige temporalmente la salida estándar a /dev/null para silenciar prints."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

        
def load_linear_predictions():
    """Carga predictions del Linear (Excel)"""
    print("\n📂 [LINEAR] Cargando predictions del Excel...")
    
    file_path = Path("/home/juanchx/Documents/Trabajo/recomendation_system/"
                    "Sistema_recomnendacion-20250924T195047Z-1-001/Sistema_recomnendacion/"
                    "perfil_cliente/features_with_subcat_names.xlsx")
    
    df = pd.read_excel(file_path)
    
    # Renombrar
    rename_map = {
        'nucleo': 'CODIGO_FAMILIA',
        'SCORE_SUBCATEGORIA': 'score_final'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    
    print(f"   ✓ {len(df)} registros")
    print(f"   ✓ {df['CODIGO_FAMILIA'].nunique()} familias")
    print(f"   ✓ Features del Excel (precalculadas hasta Nov 30)")
    
    return df


def load_test_data(fecha_limite='2025-12-21'):
    """Carga compras reales de diciembre (Dic 1-21)"""
    print("\n📂 Cargando data_test.csv (Dic 1-21)...")
    
    df = pd.read_csv(Path(__file__).parent.parent.parent / "Data" / "data_test.csv",
    sep=';',
    encoding='utf-8')
    df['DIM_PERIODO'] = pd.to_datetime(df['DIM_PERIODO'])

    #filtrar a solo 15 primeros dias de diciembre
    df = df[df['DIM_PERIODO'] <= fecha_limite]
    
    print(f"   ✓ {len(df)} compras reales")
    print(f"   ✓ {df['CODIGO_FAMILIA'].nunique()} familias")
    print(f"   ✓ Período: {df['DIM_PERIODO'].min()} a {df['DIM_PERIODO'].max()}")
    
    return df


def compute_fnn_features(fecha_corte_str):
    """Calcula features para FNN hasta fecha_corte"""
    print(f"\n🔮 [FNN] Calculando features hasta {fecha_corte_str}...")
    
    # Cargar histórico
    hist_file = Path(__file__).parent.parent.parent / "Data" / "Historico_08122025.csv"
    df_historico = load_historical_dataset(hist_file)
    
    ##*************############FILTRO DEBGUG PARA PERSONA Y CATEGORIA  113-116 29-34 398 ###########################
    #score alto muchas compras 1759533761 8792
    #1759533761	9627 score alto 0.3
    #socre bajo muchas compras  1711285286	2934 score 0.1
    #familia = 1712575826    

    #subcategoria = 9158
    #df_historico = df_historico[(df_historico['CODIGO_FAMILIA'] == familia) & (df_historico['COD_SUBCATEGORIA'] == subcategoria)]
    ####***************#################################################
    
    # Filtrar
    fecha_corte = pd.Timestamp(fecha_corte_str)
    df_hist_filtered = df_historico[df_historico['DIM_PERIODO'] <= fecha_corte].copy()
    
    print(f"   Histórico hasta: {fecha_corte}")
    print(f"   Calculando features por familia...")
    
    # Calcular features
    familias = df_hist_filtered['CODIGO_FAMILIA'].unique()
    print(f"   Total familias a procesar: {len(familias)}")
    results = []
    
    for idx, familia in enumerate(familias, 1):
        if idx % 100 == 0:
            print(f"   ✓ Procesadas {idx}/{len(familias)} familias ({idx/len(familias)*100:.1f}%)")
        
        df_fam = df_hist_filtered[df_hist_filtered['CODIGO_FAMILIA'] == familia].copy()
        
        try:
            # Pasar fecha_corte explícitamente para cálculo de ciclos largos (hasta 3 años)
            features = compute_features_for_family(df_fam, familia, fecha_corte=fecha_corte)
            if features.empty:
                continue
            if 'nucleo' in features.columns:
                features = features.rename(columns={'nucleo': 'CODIGO_FAMILIA'})
            results.append(features)
        except Exception as e:
            if idx <= 3:  # Solo mostrar primeros errores para debug
                print(f"\n   ⚠️ Error en familia {familia}: {str(e)}")
            continue
    
    print()
    df_features = pd.concat(results, ignore_index=True)
    
    print(f"\n📊 Dataset completo (antes de filtrar):")
    print(f"   Total: {len(df_features)} registros")
    print(f"   Familias: {df_features['CODIGO_FAMILIA'].nunique()}")
    
    # Mostrar distribución de ciclos
    tipo_dist = df_features['Ciclos_tipo_ciclo'].value_counts()
    print(f"\n   Distribución de ciclos:")
    for tipo in ['corto', 'corto_medio', 'mediano', 'largo', 'no_ciclico']:
        if tipo in tipo_dist.index:
            count = tipo_dist[tipo]
            pct = count/len(df_features)*100
            print(f"      {tipo:11s}: {count:5d} ({pct:4.1f}%)")
    
    # IMPORTANTE: NO filtrar no_ciclicos en evaluación
    # El modelo fue entrenado con cíclicos (patrones estables)
    # Pero puede generalizar a no_ciclico basándose en sow_24m (feature más poderosa)
    # Esto permite recomendar subcategorías no_ciclico con sow alto
    
    df_antes = len(df_features)
    # df_features = df_features[df_features['Ciclos_tipo_ciclo'] != 'no_ciclico'].copy()  # COMENTADO
    # df_features = df_features[df_features['Ciclos_tipo_ciclo'] == 'corto_medio'].copy()  # COMENTADO - AHORA SÍ
    # Mostrar distribución final
    tipo_dist_final = df_features['Ciclos_tipo_ciclo'].value_counts()
    print(f"\n   🎯 Dataset de evaluación (INCLUYE no_ciclico):")
    print(f"      Total: {len(df_features)} registros")
    print(f"      Familias: {df_features['CODIGO_FAMILIA'].nunique()}")
    print(f"\n   Distribución final:")
    for tipo in ['corto', 'corto_medio', 'mediano', 'largo', 'no_ciclico']:
        if tipo in tipo_dist_final.index:
            count = tipo_dist_final[tipo]
            pct = count/len(df_features)*100
            print(f"      {tipo:11s}: {count:5d} ({pct:4.1f}%)")
    
    print(f"\n   💡 Modelo entrenado con cíclicos → evalúa TODOS (red decide por sow)")
    
    return df_features

def load_fnn_model(fecha_str):
    """Carga modelo FNN según fecha"""
    print(f"\n📂 [FNN] Cargando modelo...")

    suffix = fecha_str.replace('-', '')[-4:]
    model_file = f'model_{suffix}.h5'
    scaler_file = f'scaler_{suffix}.pkl'
    
    model = keras.models.load_model(model_file)
    scaler = joblib.load(scaler_file)
    
    print(f"   ✓ {model_file} (entrenado hasta {fecha_str})")
    print(f"   ✓ {scaler_file}")
    
    return model, scaler


def evaluate_model(predictions_df, test_df, model_name, score_col='score_final', top_k=3):
    """
    Evalúa modelo vs compras reales
    
    Para cada familia:
    1. Ordenar por score (descendente)
    2. Tomar TOP-K
    3. Comparar con compras reales
    4. Calcular Precision, Recall, Hit Rate
    """
    print(f"\n📊 [{model_name}] Evaluando TOP-{top_k}...")
    
    precisions = []
    recalls = []
    hit_rates = []
    familias_sin_data = 0
    
    for familia in test_df['CODIGO_FAMILIA'].unique():
        # Predictions para esta familia
        df_pred = predictions_df[predictions_df['CODIGO_FAMILIA'] == familia].copy()
        
        if len(df_pred) == 0:
            familias_sin_data += 1
            continue
        
        # IMPORTANTE: Filtrar familias con menos de K subcategorías (mismo criterio que training)
        if len(df_pred) < top_k:
            familias_sin_data += 1
            continue
        
        # TOP-K
        top_k_items = df_pred.nlargest(top_k, score_col)['COD_SUBCATEGORIA'].values
        
        # Compras reales
        compradas = test_df[test_df['CODIGO_FAMILIA'] == familia]['COD_SUBCATEGORIA'].unique()
        
        if len(compradas) == 0:
            continue
        
        # Métricas
        n_correctas = len(set(top_k_items) & set(compradas))
        precision = n_correctas / top_k
        recall = n_correctas / len(compradas)
        hit_rate = 1.0 if n_correctas > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        hit_rates.append(hit_rate)
    
    results = {
        'precision@3': np.mean(precisions),
        'recall@3': np.mean(recalls),
        'hit_rate@3': np.mean(hit_rates),
        'n_families_evaluated': len(precisions),
        'n_families_skipped': familias_sin_data
    }
    
    print(f"   Familias evaluadas: {results['n_families_evaluated']}")
    if results['n_families_skipped'] > 0:
        print(f"   Familias sin data: {results['n_families_skipped']}")
    print(f"   Precision@{top_k}: {results[f'precision@{top_k}']:.4f} ({results[f'precision@{top_k}']*100:.1f}%)")
    print(f"   Recall@{top_k}: {results[f'recall@{top_k}']:.4f} ({results[f'recall@{top_k}']*100:.1f}%)")
    print(f"   Hit Rate@{top_k}: {results[f'hit_rate@{top_k}']:.4f} ({results[f'hit_rate@{top_k}']*100:.1f}%)")
    
    return results

def formatear_df_final(df):
    df_raw = load_linear_predictions()

    columnas_subcategrio = ["COD_SUBCATEGORIA","NOMBRE_SUBCATEGORIA"]

    df_raw_subcat = df_raw[columnas_subcategrio].copy().drop_duplicates()
    
    df = df.merge(df_raw_subcat, on="COD_SUBCATEGORIA", how="left")

    df.rename(columns={
    "CODIGO_FAMILIA":"CODIGO_FAMILIA",
    "COD_SUBCATEGORIA":"COD_SUBCATEGORIA",
    "NOMBRE_SUBCATEGORIA":"NOMBRE_SUBCATEGORIA",
    "recencia_hl":"RECENCIA",
    "freq_baja":"FREQ_BAJA",
    "freq_media":"FREQ_MEDIA",
    "freq_alta":"FREQ_ALTA",
    "cv_invertido":"CV_INVERTIDO",
    "sow_24m":"SOW_24M",
    "season_ratio":"ESTACIONALIDAD",
    "fnn_prob":"SCORE_SUBCATEGORIA_FNN",
    "Ciclos_ciclo_dias":"CICLO",
    "Recencia_dias_desde_ultima_compra":"DIAS_ULTIMA_COMPRA",
    "ciclo_dias_mu":"CICLO_DIAS_MU"}, inplace=True)

    #read item.xlsx
    item_file = Path(__file__).parent.parent.parent / "Data" / "item.xlsx"
    df_item = pd.read_excel(item_file)
    nombre_divicion = ["FARMA","CONSUMO"]
    df_item = df_item[df_item['NOMBRE_DIVISION_COMERCIAL'].isin(nombre_divicion)]
    
    #DROP DUPLICADOS 
    colums_unicas = ['COD_ITEM', 'COD_SUBCATEGORIA','COD_DIVISION_COMERCIAL']
    df_item = df_item.drop_duplicates(subset=colums_unicas).reset_index(drop=True)

    columnas_uso = ['COD_SUBCATEGORIA','COD_DIVISION_COMERCIAL','NOMBRE_DIVISION_COMERCIAL']
    item_merge = df_item[columnas_uso].drop_duplicates(subset=['COD_SUBCATEGORIA']).copy()
    
    df = df.merge(item_merge,on ="COD_SUBCATEGORIA",how="left")
    # --- FARMA ---
    feature_farma = df[df['NOMBRE_DIVISION_COMERCIAL'] == 'FARMA'].copy()
    # Ordenar por nucleo y score para ranking correcto por familia
    feature_farma = feature_farma.sort_values(by=["CODIGO_FAMILIA", "SCORE_SUBCATEGORIA_FNN"], ascending=[True, False])
    # Ranking por nucleo
    feature_farma['top_s'] = feature_farma.groupby("CODIGO_FAMILIA").cumcount() + 1

    # --- CONSUMO ---
    feature_consumo = df[df['NOMBRE_DIVISION_COMERCIAL'] == 'CONSUMO'].copy()
    # Ordenar por nucleo y score para ranking correcto por familia
    feature_consumo = feature_consumo.sort_values(by=["CODIGO_FAMILIA", "SCORE_SUBCATEGORIA_FNN"], ascending=[True, False])
    # Ranking por nucleo
    feature_consumo['top_s'] = feature_consumo.groupby("CODIGO_FAMILIA").cumcount() + 1

    feature_concat = pd.concat([feature_farma, feature_consumo], ignore_index=True)
    features_all_subcat = feature_concat.copy()

    features_all_subcat['PPMI'] = 0
    features_all_subcat['top'] = 0
    features_all_subcat.rename(columns={"top_s":"TOP_DC","SCORE_SUBCATEGORIA_FNN":"SCORE_SUBCATEGORIA"}, inplace=True)



    columnas_finales = [
    'top',
    'CODIGO_FAMILIA',
    'COD_SUBCATEGORIA',
    'NOMBRE_SUBCATEGORIA',
    'RECENCIA',
    'FRECUENCIA',
    'SOW',
    'ESTACIONALIDAD',
    'PPMI',
    'SCORE_SUBCATEGORIA',
    'CICLO',
    'DIAS_ULTIMA_COMPRA',
    'CICLO_DIAS_MU',
    'MEDIA_ESTACION',
    'STD_DEV_ESTACION',
    'COD_DIVISION_COMERCIAL',
    'NOMBRE_DIVISION_COMERCIAL',
    'TOP_DC']

    # DEBUG: Ver qué columnas existen vs. las que queremos
    print(f"\n🔍 DEBUG: Verificando columnas:")
    columnas_existentes = [c for c in columnas_finales if c in features_all_subcat.columns]
    columnas_faltantes = [c for c in columnas_finales if c not in features_all_subcat.columns]
    print(f"   ✅ Existen ({len(columnas_existentes)}): {columnas_existentes}")
    print(f"   ❌ Faltan ({len(columnas_faltantes)}): {columnas_faltantes}")
    
    # Usar solo las que existen
    if columnas_faltantes:
        print(f"\n⚠️  ADVERTENCIA: Faltan {len(columnas_faltantes)} columnas, usando solo las existentes")
        features_all = features_all_subcat[columnas_existentes].copy()
    else:
        features_all = features_all_subcat[columnas_finales].copy()
    
    features_all.to_excel("features_with_subcat_names.xlsx", index=False)
    print(f"   ✓ Guardado: features_with_subcat_names.xlsx ({len(features_all)} filas, {len(features_all.columns)} columnas)")

    return df

def main():
    """Pipeline principal"""
    
    print("=" * 80)
    print("COMPARACIÓN FINAL: LINEAR vs FNN")
    print("=" * 80)
    print(f"\n📝 Configuración:")
    print(f"   Modelo FNN: Entrenado hasta {FECHA_MODELO}")
    print(f"   Features evaluación: Calculadas hasta {FECHA_EVALUACION}")
    print(f"   Linear: Excel (features hasta Nov 30)")
    print(f"   Test: data_test.csv (Dic 1-9)")
    
    # 1. Cargar test data
    test_df = load_test_data()
    
    # 2. Evaluar LINEAR
    linear_df = load_linear_predictions()
    linear_results = evaluate_model(linear_df, test_df, 'Linear', 'score_final', 3)
    
    # 3. Calcular features FNN (SIEMPRE Nov 30 - toda la info disponible)
    with suppress_stdout():
        fnn_df = compute_fnn_features(FECHA_EVALUACION)
    
    ##*******##############################
    #return fnn_df
    #********##############################
    # 4. Cargar modelo FNN (SIEMPRE Nov 9 - sin leakage)
    model, scaler = load_fnn_model(FECHA_MODELO)
    
    # 5. Predecir con FNN
    print("\n🔮 [FNN] Generando predictions...")
    # 8 features base + 4 one-hot (tipo_ciclo) = 12 features
    base_feature_cols = [
        'recencia_hl', 
        'freq_baja', 
        'freq_media', 
        'freq_alta', 
        'cv_invertido', 
        'sow_24m', 
        'season_ratio',
        'ciclo_dias_mu',
        'Ciclos_ciclo_binario_c'
    ]
    
    # One-hot encoding de tipo_ciclo_b (mismo que train_fnn.py)
    tipo_ciclo_dummies = pd.get_dummies(fnn_df['Debug_ciclos_tipo_ciclo_b'], prefix='tipo', drop_first=True)
    
    # Concatenar features base + one-hot
    X_base = fnn_df[base_feature_cols]
    X_all = pd.concat([X_base.reset_index(drop=True), tipo_ciclo_dummies.reset_index(drop=True)], axis=1)
    
    print(f"   ✓ Features: {X_all.shape[1]} columnas")
    print(f"   ✓ Columnas: {list(X_all.columns)}")
    
    X = np.nan_to_num(X_all.values, nan=0.0)
    X_scaled = scaler.transform(X)
    
    probs = model.predict(X_scaled, verbose=0).flatten()
    fnn_df['fnn_prob'] = probs
    
    print(f"   ✓ {len(fnn_df)} predictions generadas")
    
    # Guardar predicciones FNN con features Nov 30
    fnn_df.to_csv('predictions_fnn_final.csv', index=False)
    print(f"   ✓ Guardado: predictions_fnn_final.csv")
    
    ############################################################################
    # FILTRO OPCIONAL: Descomentar para evaluar SOLO un tipo de ciclo específico
    ############################################################################
    # print(f"\n🔍 Filtrando test para solo evaluar corto_medio...")
    # print(f"   Test original: {len(test_df)} compras")
    # 
    # # Solo mantener compras de (FAMILIA, SUBCAT) que existen en fnn_df (corto_medio)
    # subcat_validas = set(fnn_df[['CODIGO_FAMILIA', 'COD_SUBCATEGORIA']].apply(tuple, axis=1))
    # test_df_filtrado = test_df[test_df[['CODIGO_FAMILIA', 'COD_SUBCATEGORIA']].apply(tuple, axis=1).isin(subcat_validas)].copy()
    # 
    # print(f"   Test filtrado: {len(test_df_filtrado)} compras (solo corto_medio)")
    # print(f"   ✓ Solo se evaluarán compras de subcategorías corto_medio")
    ############################################################################
    
    # 6. Evaluar FNN (con todas las compras del test)
    fnn_results = evaluate_model(fnn_df, test_df, 'FNN', 'fnn_prob', 3)
    
    # 7. Comparar
    print("\n" + "=" * 80)
    print("📊 COMPARACIÓN FINAL")
    print("=" * 80)
    
    comparison_df = pd.DataFrame([
        {'model': 'Linear', **linear_results},
        {'model': 'FNN', **fnn_results}
    ])
    
    print(f"\n{comparison_df.to_string(index=False)}")
    
    # Diferencias
    print(f"\n🎯 Diferencias (FNN - Linear):")
    for metric in ['precision@3', 'recall@3', 'hit_rate@3']:
        diff = fnn_results[metric] - linear_results[metric]
        pct = (diff / linear_results[metric] * 100) if linear_results[metric] > 0 else 0
        emoji = "✅" if diff > 0 else "❌"
        print(f"   {emoji} {metric}: {diff:+.4f} ({pct:+.1f}%)")
    
    # Veredicto
    if fnn_results['precision@3'] > linear_results['precision@3']:
        mejora = (fnn_results['precision@3'] - linear_results['precision@3']) / linear_results['precision@3'] * 100
        print(f"\n✅ FNN es MEJOR (+{mejora:.1f}%)")
    elif fnn_results['precision@3'] < linear_results['precision@3']:
        print(f"\n⚠️  Linear es MEJOR")
    else:
        print(f"\n🤷 EMPATE")
    
    # Guardar
    comparison_df.to_csv('comparison_final.csv', index=False)
    fnn_df = formatear_df_final(fnn_df) 
    print(f"\n💾 Guardado: comparison_final.csv")
    
    print("\n" + "=" * 80)
    print("✅ COMPARACIÓN COMPLETADA")
    print("=" * 80)


if __name__ == "__main__":
    main()
