# Resumen de hallazgos — Análisis de features del modelo de recompra

## El problema

Predecir las **3 subcategorías más probables** de recompra de cada familia en los próximos 21 días.

---

## ¿Qué hicimos?

1. Auditamos las features que el modelo usaba realmente.
2. **Recuperamos 4 features que ya se calculaban pero estaban siendo descartadas** del dataset final.
3. **Añadimos 2 features verdaderamente nuevas:**
   - `ticket_promedio` — promedio de gasto por factura (primera variable monetaria del modelo).
   - `n_subcats_familia` — cuántas subcategorías diferentes compra cada familia.
4. Reentrenamos y medimos el impacto.

---

## Resultados — métricas

| Modelo | Antes | Después | Mejora |
|---|---:|---:|:---:|
| Red neuronal (FNN) | 0.4306 | 0.4319 | +0.3% |
| **LightGBM (árbol) — actual ganador** | **0.4376** | **0.4466** | **+2.1%** |

> _Métrica: Precision@3 — de cada 3 sugerencias del modelo, qué porcentaje son recompras reales._

**El árbol pasó de acertar 43.8% a 44.7% de las recomendaciones.** No suena épico, pero en Top-3 cualquier punto es difícil de ganar y se traduce a miles de aciertos extra al mes a escala real.

Métricas complementarias del LightGBM final:
- **Hit Rate@3 = 76.9%** → en 77 de cada 100 familias acertamos al menos 1 subcategoría dentro del top 3.
- **Recall@3 = 25.7%** → cubrimos el 26% de todas las recompras que ocurren.

---

## Hallazgos importantes

### 1. La feature más valiosa que añadimos era… "gratis"
`dias_desde_ultima_compra` ya se calculaba en el código pero se descartaba antes de guardar. **Es la 2ª variable más importante del modelo**, solo superada por la categoría del ciclo. Recuperarla aportó la mayor parte de la mejora.

### 2. La dimensión monetaria estaba ausente
El modelo solo contaba transacciones. Al añadir `ticket_promedio` (cuánto gasta el cliente por factura) ganamos una nueva señal que el árbol usa intensamente — aunque su correlación lineal con el target es casi cero, el modelo aprovecha relaciones no lineales.

### 3. Hay features que el modelo nunca usa
Confirmamos que **4 variables son prácticamente inútiles** (aportan menos del 0.05% del modelo):
- `Ciclos_ciclo_binario_c`
- `freq_baja`, `freq_media`, `freq_alta`

Las llevamos meses guardando sin que sirvan para nada. Se pueden quitar para simplificar.

### 4. El problema es muy heterogéneo
La probabilidad de recompra varía 18× según el tipo de cliente:

| Tipo de familia | % que recompra (en 21 días) |
|---|---:|
| Cíclico corto | **44.8%** |
| Cíclico medio-corto | 28.4% |
| Cíclico mediano | 13.6% |
| Cíclico largo | 7.0% |
| **No cíclico** | **2.5%** |

Las familias cíclicas son fáciles de predecir. **Las no-cíclicas son ~75% de los datos pero solo el 2.5% recompra** — son el verdadero reto del modelo.

### 5. El árbol vence a la red neuronal
LightGBM (gradient boosting) supera consistentemente a la red neuronal con menos esfuerzo de tuning. **Recomendación: el modelo de producción debería ser LightGBM.**

---

## Visualizaciones (carpeta `docs/feature_analysis/`)

- `01_correlation_matrix.png` — matriz de correlaciones entre features.
- `02_corr_with_target.png` — qué tanto correlaciona cada feature con la recompra.
- `03_feature_importance_lgbm.png` — ranking de importancia de features.
- `04_target_por_tipo_ciclo.png` — heterogeneidad del problema.

---

## Próximos pasos sugeridos

1. **Quitar las 4 features inútiles** del pipeline (limpieza, no afecta métrica).
2. **Considerar 2 modelos separados**: uno para familias cíclicas (alta señal) y otro para no-cíclicas (baja señal). Hoy un solo modelo intenta atender ambos casos.
3. **Explorar más variables monetarias** (frecuencia de gasto, ticket por subcategoría) — la dimensión monetaria mostró ser valiosa.
4. **Re-tunear hiperparámetros** del LightGBM con un set de features más limpio para extraer el último 1-2%.
