# TRACKING — Mejoras al modelo de recompra

Bitácora de los experimentos realizados y pendientes. Cada paso registra **qué se hizo, qué se midió y qué archivo lo respalda**.

---

## ✅ Paso 1 — Aumentamos features (10 → 16 numéricas)

**Qué se hizo:** Auditoría del pipeline. Se identificaron 4 features que ya se calculaban pero estaban siendo descartadas, y se añadieron 2 features verdaderamente nuevas.

| # | Feature añadida | Tipo |
|---|---|---|
| 1 | `dias_desde_ultima_compra` | gratuita (existía pero descartada) |
| 2 | `l_compra_sobre_ciclo` | gratuita |
| 3 | `compras_reales` | gratuita |
| 4 | `ratio_temporal` | gratuita (con clip a [0,10]) |
| 5 | `ticket_promedio` | **nueva monetaria** |
| 6 | `n_subcats_familia` | **nueva nivel familia** |

**Total inputs al modelo:** de `10 num + 4 one-hot = 14` → a `16 num + 4 one-hot = 20`.

**Archivos modificados:**
- `app/features.py:737-739` — clip `ratio_temporal` (antes era sentinel 999)
- `app/feature_engineering.py:123-129, 175-183, 195-212` — cálculo y merge de las 2 nuevas
- `app/config.py:51-67` — añadidas las 6 a `FEATURE_COLUMNS`

---

## ✅ Paso 2 — Feature importance con LightGBM

**Qué se hizo:** Reentrenamos LightGBM con `best_hparams_lgbm.json` y extrajimos `booster.feature_importance(importance_type='gain')`.

**Resultado:** 9 features acumulan **96.7% del gain total**, 6 features tienen gain < 0.5% (prácticamente inútiles para el árbol).

**Outputs:**
- `docs/feature_analysis/03_feature_importance_lgbm.png` — gráfico de barras
- `docs/feature_analysis/feature_summary.csv` — tabla con gain + correlación con target
- `docs/feature_analysis/01_correlation_matrix.png` — matriz de correlación
- `docs/feature_analysis/04_target_por_tipo_ciclo.png` — heterogeneidad del problema

> ⚠️ El cálculo se ejecutó inline en consola, **no quedó como script reproducible**. Pendiente: guardarlo como `app/feature_importance.py` para reproducirlo a futuro.

---

## ✅ Paso 3 — Decidimos quedarnos con top-9 features (7 num + 4 one-hot = 11 inputs)

**Top features (gain ≥ 1%):**

| # | Feature | Gain % |
|---|---|---:|
| 1 | `tipo_no_ciclico` (one-hot) | 40.34% |
| 2 | `dias_desde_ultima_compra` 🆕 | 15.78% |
| 3 | `ciclo_dias_mu` | 12.62% |
| 4 | `ticket_promedio` 🆕 | 6.49% |
| 5 | `score_final` | 6.10% |
| 6 | `n_subcats_familia` 🆕 | 5.92% |
| 7 | `sow_24m` | 4.72% |
| 8 | `tipo_largo` (one-hot) | 3.17% |
| 9 | `recencia_hl` | 1.77% |

> Las one-hots `tipo_corto_medio` y `tipo_mediano` se mantienen porque `pd.get_dummies` las genera juntas → no se pueden quitar individualmente. Resultado: **7 numéricas + 4 one-hots = 11 inputs**.

**Archivo modificado:** `app/config.py:51-75` — 9 features comentadas con su gain registrado en el comentario.

---

## ✅ Paso 4 — Reentrenamiento con 11 features + hparams viejos (los tuneados para modelo anterior)

**Qué se hizo:** Entrenamos FNN y LightGBM con las 11 features pero usando los mejores hiperparámetros del experimento anterior

| Modelo | Antes (20 in) | Después (11 in) | Δ |
|---|---:|---:|:---:|
| FNN | 0.4319 | **0.4118** | −0.0201 (−4.7%) ❌ |
| LightGBM | 0.4466 | **0.4425** | −0.0041 (−0.9%) ⚠️ |

**Conclusión parcial:** LGBM aguantó la poda; FNN sufrió. Pero los hparams no estaban optimizados para 11 inputs, así que la comparación era injusta → motivo del Paso 5.

---

## ✅ Paso 5 — HPT del FNN con 11 features (40 trials × 25 epochs, ~60 min)

**Qué se hizo:** Búsqueda Optuna de hiperparámetros del FNN con las 11 features podadas.

| Configuración | Precision@3 |
|---|---:|
| FNN baseline (20 in, hparams viejos) | 0.4319 |
| **FNN HPT (11 in, mejor trial #14)** | **0.3363** ❌ |
| Trial #37 (empate) | 0.3363 |
| Trial #23 | 0.3343 |

**Mejores hparams del trial #14:** `lr=0.00254, dropout=0.32, act=leaky_relu, layers=[256,128,128]`

**Conclusión:** La poda destruye al FNN (−22% relativo). El FNN necesita las features de bajo gain porque sus capas densas combinan TODAS las entradas; quitarle features le quita información que descomponía internamente. **El comportamiento del FNN es opuesto al de LightGBM.**

> ⚠️ El HPT sobrescribió `data/models/best_hparams.json` con los hparams del FNN podado (peores). Si se quiere volver al modelo anterior, hay que correr HPT FNN con 20 features de nuevo.

---

## 📋 Estado actual del repo (al cerrar la sesión)

- `app/config.py` — **podado**, 7 numéricas + 4 one-hots activos
- `data/models/best_hparams.json` — **sobrescrito**, contiene hparams del FNN podado (Precision@3 = 0.3363)
- `data/models/best_hparams_lgbm.json` — **intacto**, hparams del LGBM con 20 features (Precision@3 = 0.4466) ← **MEJOR MODELO HASTA AHORA**

---

# Próximos pasos pendientes

## 🔲 Paso 6 — HPT FNN con 20 features (lo que NO hicimos)

Cuando aumentamos de 10 → 16 features, **NO re-tuneamos** los hparams del FNN. Solo medimos con los hparams viejos (que daban 0.4306 con 10 features) y vimos que pasó a 0.4319 con 20. **Puede que el FNN mejore más con HPT propio sobre 20 features.**

Comando:
```bash
cd app
# 1. Descomentar primero las 9 features en config.py:51-75
# 2. Correr HPT FNN
setsid nohup python -u hptuning.py --trials 40 --epochs 25 > /tmp/hpt_fnn_full.log 2>&1 < /dev/null &
echo "PID: $!"
disown
# Monitorear:
tail -f /tmp/hpt_fnn_full.log
```

**Tiempo estimado:** ~60 min.

## 🔲 Paso 7 — HPT LightGBM con 11 features (lo que NO hicimos)

Análogamente, al podar features no re-tuneamos el árbol — solo lo probamos con los hparams viejos (0.4425). El HPT podría mejorar.

Comando:
```bash
cd app
# Asumiendo que config.py sigue podado (11 features)
setsid nohup python -u hptuning_lgbm.py --trials 80 > /tmp/hpt_lgbm_pruned.log 2>&1 < /dev/null &
echo "PID: $!"
disown
tail -f /tmp/hpt_lgbm_pruned.log
```

**Tiempo estimado:** ~12 min.

## 🔲 Paso 8 — Si el LGBM con 11 features sale peor en HPT (Paso 7), volver al 100% de features

Si el HPT del LGBM con 11 features no supera el 0.4466 actual, hay que:
1. Descomentar las 9 features en `app/config.py:51-75`
2. Correr HPT LGBM con las 20 features para reconfirmar el ganador
3. Correr HPT FNN con las 20 features (Paso 6)

Comandos:
```bash
# 1. Editar config.py — descomentar líneas con freq_baja, freq_media, freq_alta,
#    cv_invertido, season_ratio, Ciclos_ciclo_binario_c, l_compra_sobre_ciclo,
#    compras_reales, ratio_temporal
# 2. HPT ambos modelos:
cd app
setsid nohup python -u hptuning_lgbm.py --trials 80 > /tmp/hpt_lgbm_full.log 2>&1 < /dev/null &
disown
# Esperar ~12 min, ver resultado, luego:
setsid nohup python -u hptuning.py --trials 40 --epochs 25 > /tmp/hpt_fnn_full.log 2>&1 < /dev/null &
disown
```

---

# Resumen ejecutivo (lo que sabemos hasta hoy)

| Experimento | Precision@3 | Hit Rate@3 | Recall@3 |
|---|---:|---:|---:|
| FNN baseline (10 features) | 0.4306 | — | — |
| LGBM baseline (10 features) | 0.4376 | — | — |
| FNN (20 features, hparams viejos) | 0.4319 | 0.7444 | 0.2472 |
| LGBM (20 features, hparams viejos) | 0.4444 | 0.7603 | 0.2543 |
| **🏆 LGBM (20 features, HPT 80 trials)** | **0.4466** | **0.7692** | **0.2572** |
| FNN (11 features, hparams viejos) | 0.4118 | 0.7383 | 0.2408 |
| LGBM (11 features, hparams viejos) | 0.4425 | 0.7623 | 0.2532 |
| FNN (11 features, HPT 40 trials) | 0.3363 | — | — |

**Conclusión hasta hoy:** El mejor modelo es **LightGBM con 20 features** (10 originales + 6 añadidas, HPT 80 trials) = **Precision@3 = 0.4466**.

**Para sellar la conclusión faltan los pasos 6, 7 y 8.**
