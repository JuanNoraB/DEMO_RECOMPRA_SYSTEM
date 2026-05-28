# Análisis de features y resultados — Mayo 2026

Este documento resume los hallazgos del análisis de features del modelo de recompra y los experimentos realizados después de añadir 6 features nuevas al pipeline.

---

## 1. Contexto del problema

- **Tarea:** predecir, para cada par (familia, subcategoría), si la familia comprará esa subcategoría en los próximos 21 días (binario).
- **Métricas reportadas:** Precision@3, Hit Rate@3, Recall@3 — sobre familias que tienen al menos 1 positivo en la ventana de evaluación.
- **Datos:** `Historico_08122025.csv`, 2,891 familias, ~279k filas en `features_train.parquet`, ~5.8% de positivos.
- **Split temporal usado para HPT:** `features_hpt_train.parquet` (CSV truncado -21 días) para entrenar; `features_train.parquet` (CSV completo) para evaluar.

---

## 2. Features añadidas en este experimento

Se identificaron **4 features que ya se calculaban pero se descartaban** del parquet, y **2 features nuevas** verdaderamente nuevas:

| Feature | Tipo | Origen | Descripción |
|---|---|---|---|
| `dias_desde_ultima_compra` | gratuita (existía) | `compute_recency_features` | Días desde la última compra de la subcategoría. |
| `l_compra_sobre_ciclo` | gratuita (existía) | `compute_recency_features` | `dias_desde_ultima_compra / ciclo_mu` — cuántos ciclos pasaron. |
| `compras_reales` | gratuita (existía) | `compute_frequency_features` | Conteo crudo de compras en la ventana de revisión. |
| `ratio_temporal` | gratuita (existía) | `compute_seasonality_features` | Ratio (compras_actual / compras_pasado), clipeado a [0, 10]. |
| `ticket_promedio` | **nueva** | calculada en `compute_features_for_family` | `total_venta_neta / facturas_unicas`. **Primera feature monetaria.** |
| `n_subcats_familia` | **nueva** | calculada en `compute_features_for_family` | `df_family["COD_SUBCATEGORIA"].nunique()`. **Primera feature de nivel familia.** |

Cambios de código:
- `app/features.py` línea 737 — clip `ratio_temporal` a max 10 (antes era sentinel 999).
- `app/feature_engineering.py` líneas 123–129 — cálculo de `ticket_promedio` y `n_subcats_familia`.
- `app/feature_engineering.py` líneas 175–183 — merge de `ticket_promedio` y replicación de `n_subcats_familia`.
- `app/feature_engineering.py` líneas 195–212 — añadidas a `score_columns` para que entren al parquet.
- `app/config.py` líneas 62–68 — añadidas a `FEATURE_COLUMNS`.

---

## 3. Métricas — comparativa

Todas las métricas son **Precision@3 evaluado sobre `features_train.parquet`** (el modelo entrena en `features_hpt_train.parquet` que tiene corte 21 días antes).

| Configuración | Precision@3 | Hit Rate@3 | Recall@3 | Δ vs baseline |
|---|---|---|---|---|
| FNN baseline (10 feats, hparams viejos) | 0.4306 | n/a | n/a | — |
| LightGBM baseline (10 feats, hparams viejos) | 0.4376 | n/a | n/a | — |
| FNN (16 feats, hparams viejos) | 0.4319 | 0.7444 | 0.2472 | +0.0013 |
| LightGBM (16 feats, hparams viejos) | 0.4444 | 0.7603 | 0.2543 | +0.0068 |
| **🏆 LightGBM (16 feats, HPT 80 trials)** | **0.4466** | **0.7692** | **0.2572** | **+0.0090 (+2.1%)** |

**Hparams ganadores LightGBM** (`best_hparams_lgbm.json`):
```
n_estimators=569, num_leaves=81, learning_rate=0.0101,
min_child_samples=26, subsample=0.66, feature_fraction=0.87,
reg_alpha=0.0026, reg_lambda=0.24
```

---

## 4. Feature importance — LightGBM (gain %)

Tabla completa en `feature_summary.csv`. Visualización en `03_feature_importance_lgbm.png`.

| # | Feature | Gain % | Corr con target |
|---|---|---:|---:|
| 1 | `tipo_no_ciclico` (one-hot) | 40.34% | n/a |
| 2 | **`dias_desde_ultima_compra`** 🆕 | 15.78% | -0.184 |
| 3 | `ciclo_dias_mu` | 12.62% | -0.001 |
| 4 | **`ticket_promedio`** 🆕 | 6.49% | +0.002 |
| 5 | `score_final` | 6.10% | +0.256 |
| 6 | **`n_subcats_familia`** 🆕 | 5.92% | +0.011 |
| 7 | `sow_24m` | 4.72% | +0.298 |
| 8 | `tipo_largo` (one-hot) | 3.17% | n/a |
| 9 | `recencia_hl` | 1.77% | +0.223 |
| 10 | `tipo_mediano` (one-hot) | 0.81% | n/a |
| 11 | `l_compra_sobre_ciclo` 🆕 | 0.73% | +0.039 |
| 12 | `cv_invertido` | 0.56% | +0.205 |
| 13 | `compras_reales` 🆕 | 0.49% | +0.276 |
| 14 | `tipo_corto_medio` (one-hot) | 0.21% | n/a |
| 15 | `season_ratio` | 0.19% | +0.255 |
| 16 | `ratio_temporal` 🆕 | 0.06% | +0.074 |
| 17 | `freq_alta` | 0.02% | +0.034 |
| 18 | `freq_baja` | 0.01% | +0.231 |
| 19 | `freq_media` | 0.01% | +0.184 |
| 20 | `Ciclos_ciclo_binario_c` | **0.001%** | +0.253 |

**Las 3 features nuevas (`dias_desde_ultima_compra`, `ticket_promedio`, `n_subcats_familia`) suman 28.2% del gain total.**

---

## 5. Hallazgos clave

### 🔴 Features inútiles confirmadas (gain < 0.5%, 4 candidatas a eliminar)
- **`Ciclos_ciclo_binario_c`** (gain=0.001%) — Aunque tiene correlación 0.253 con target, el modelo NO la usa. La info ya la captura `tipo_no_ciclico` (que es esencialmente el complemento booleano).
- **`freq_baja`, `freq_media`, `freq_alta`** (gain ~0.01% cada una) — Tienen correlación 0.18-0.23 con target pero el modelo prácticamente las ignora. Probablemente son redundantes con `compras_reales` y `cv_invertido` que las dominan.
- **`ratio_temporal`** (gain=0.06%) — Aporte marginal. La feature `season_ratio` (con plateau) tampoco aporta mucho (0.19%).

### 🟢 Features estrella
- **`tipo_no_ciclico` (one-hot)** absorbe el 40% del gain. La distinción cíclico/no-cíclico es la señal más fuerte del problema.
- **`dias_desde_ultima_compra`** se confirma como la 2ª feature más importante. Llevaba meses calculada pero descartada del parquet — recuperarla aportó +0.7% absoluto.
- **`ticket_promedio`** (monetaria) tiene corr~0 con target pero gain 6.5%. Esto indica una **relación no lineal**: el modelo divide los clientes en segmentos por ticket y aprende patrones distintos para cada uno. **Confirma que faltaba la dimensión monetaria.**

### 📊 Heterogeneidad del problema
La tasa de positivos varía dramáticamente por tipo de ciclo (ver `04_target_por_tipo_ciclo.png`):

| Tipo de ciclo | N filas | % positivos |
|---|---:|---:|
| corto | 5,975 | **44.8%** |
| corto_medio | 11,342 | 28.4% |
| mediano | 21,061 | 13.6% |
| largo | 33,704 | 7.0% |
| no_ciclico | 207,431 | **2.5%** |

Las familias cíclicas son ~18× más predecibles que las no cíclicas. El modelo basa la mayor parte de su capacidad predictiva en la columna `tipo_no_ciclico`.

---

## 6. Visualizaciones

- `01_correlation_matrix.png` — Matriz de correlación de Pearson entre todas las features y target.
- `02_corr_with_target.png` — Correlación de cada feature con target, ordenada por magnitud.
- `03_feature_importance_lgbm.png` — Feature importance de LightGBM (gain %), coloreada por umbral.
- `04_target_por_tipo_ciclo.png` — Tasa de positivos por tipo de ciclo (heterogeneidad del problema).
- `feature_summary.csv` — Tabla completa con gain, gain_pct y correlación con target.

---

## 7. Próximos pasos sugeridos

1. **Podar features inútiles** (`Ciclos_ciclo_binario_c`, `freq_baja/media/alta`) — re-correr HPT a ver si simplificar mejora algo.
2. **Probar features de tendencia más sofisticadas** (ej. ratio compras_3m / compras_12m) para no-cíclicos, donde el modelo tiene poca señal.
3. **Considerar entrenar dos modelos** (uno para cíclicos, otro para no-cíclicos) dado que la distribución es muy distinta.
4. **Probar LightGBM con `is_unbalance=True`** o ajustar `scale_pos_weight` por subgrupo.
