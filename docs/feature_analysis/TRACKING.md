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

| Modelo | Antes (14 in) | Después (11 in) | Δ |
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

---

## ✅ Paso 6 — Log de ejecución automático en feature engineering

**Qué se hizo:** Se agregó registro automático de tiempo y estadísticas al final de `run_pipeline`.

Cada ejecución del pipeline appends una línea JSON a:
`data/logs/feature_engineering_runs.jsonl`

**Campos registrados por ejecución:**
```json
{
  "timestamp": "2026-06-10T12:40:00",
  "duracion_min": 16.5,
  "duracion_seg": 990.0,
  "num_familias": 2891,
  "num_pares_fam_subcat": 279513,
  "num_features": 18,
  "features_calculadas": ["recencia_hl", "sow_24m", ...],
  "target_1_positivos": 16212,
  "target_0_negativos": 263301,
  "series_con_ciclo": 12000,
  "workers": 16,
  "prediction_window_days": 21,
  "historico_path": "...",
  "output_path": "...",
  "con_filtro": false
}
```

**Archivos modificados:** `app/feature_engineering.py` — imports `time/json/datetime` + log al final de `run_pipeline`.

---

## 📋 Estado actual del parquet (verificado Jun 2026)

| Métrica | Valor |
|---|---:|
| Familias únicas | 2,891 |
| Pares familia/subcategoría | 279,513 |
| Target = 1 (compraron) | 16,212 (5.8%) |
| Target = 0 (no compraron) | 263,301 (94.2%) |
| Columnas totales en parquet | 20 |

**Path:** `data/features_store/features_train.parquet`

**Features activas para el modelo** (de `app/config.py:51-75`):
- 7 numéricas: `recencia_hl`, `sow_24m`, `score_final`, `ciclo_dias_mu`, `dias_desde_ultima_compra`, `ticket_promedio`, `n_subcats_familia`
- 4 one-hots (auto desde `Debug_ciclos_tipo_ciclo_b`): `tipo_corto_medio`, `tipo_largo`, `tipo_mediano`, `tipo_no_ciclico`
- **Total: 11 inputs al modelo**

---

## 📋 Comandos de referencia

### Feature Engineering

```bash
# Sin Docker — todas las familias
python app/feature_engineering.py --historico Historico_08122025.csv --workers 16

# Sin Docker — con filtro (entrenamiento continuo)
python app/feature_engineering.py --historico Historico_08122025.csv \
  --filtro data/nuevas_series.csv --workers 16

# Sin Docker — pipeline completo (features + FNN)
python app/entrypoint_train.py --historico Historico_08122025.csv --workers 16 --epochs 50 --force

# Con Docker — todas las familias
docker run --rm -v $(pwd)/data:/data -v $(pwd)/Historico_08122025.csv:/data/raw/historico.csv \
  fnn-train python feature_engineering.py --historico /data/raw/historico.csv --workers 16

# Con Docker — con filtro
docker run --rm -v $(pwd)/data:/data -v $(pwd)/Historico_08122025.csv:/data/raw/historico.csv \
  fnn-train python feature_engineering.py --historico /data/raw/historico.csv \
  --filtro /data/nuevas_series.csv --workers 16
```

**Archivo de filtro (entrenamiento continuo):** `data/nuevas_series.csv`
Formato: CSV con columnas `CODIGO_FAMILIA;COD_SUBCATEGORIA`

### Ver log de ejecuciones
```bash
cat data/logs/feature_engineering_runs.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    print(f'{r[\"timestamp\"]} | {r[\"duracion_min\"]}min | {r[\"num_familias\"]} fam | {r[\"num_pares_fam_subcat\"]} pares | target+={r[\"target_1_positivos\"]}')
"
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

---

# Búsqueda de Hiperparámetros (HPT) — FNN y LightGBM

## Parquets necesarios (generados UNA sola vez, sirven para ambos modelos)

```
prepare_hpt_split(historico, trim_days=21) genera:

  historico_hpt_train.csv = transacciones desde inicio hasta (fecha_max - 21d)
  historico_hpt_eval.csv  = historico completo (sin truncar)

Luego run_pipeline sobre historico_hpt_train.csv con prediction_window=21:
  → features_hpt_train.parquet  ← TRAIN de HPT
  → features_train.parquet ya existe ← TEST de HPT
```

## Fechas concretas (ejemplo con fecha_max = 2025-12-01)

```
Tiempo ──────────────────────────────────────────────────────────►
         inicio histórico         2025-10-19   2025-11-10   2025-12-01
              │                       │             │             │
              ├─── features TRAIN ────┤             │             │
                                      ├─ target T ──┤             │
                                      │             │             │
                                      ├─── features TEST ─────────┤
                                                    ├─ target T ──┤

features_hpt_train.parquet:
  FEATURES  → datos hasta 2025-10-19   (fecha_max_truncado - 21d)
  TARGET    → ¿compró entre 2025-10-19 y 2025-11-09?

features_train.parquet (TEST):
  FEATURES  → datos hasta 2025-11-10   (fecha_max_completo - 21d)
  TARGET    → ¿compró entre 2025-11-10 y 2025-12-01?
```

Código de referencia: `simulation.py:144-146` + `hptuning.py:84-89`

---

## HPT — FNN (`hptuning.py`)

### Split interno durante cada trial

| Rol | Tamaño | Separación temporal |
|---|---|---|
| Train (gradiente) | 90% aleatorio de features_hpt_train | No (split random) |
| Val. interna (early stopping) | 10% aleatorio del mismo parquet | No (mismas fechas) |
| Test / Eval (métrica del trial) | features_train.parquet completo | Sí (ventana posterior) |

Normalización: Z-score via `FeatureScaler` antes de entrar al modelo.

### Ejecución sin Docker

```bash
cd "/home/juanchx/Documents/Maestria_IA/desplieje a produccion/labs/Final/app"

# Con preparación de parquets (primera vez)
python hptuning.py \
  --prepare \
  --historico ../data/raw/historico_1000.csv \
  --trials 100 \
  --epochs 30

# Solo búsqueda (parquets ya existen)
python hptuning.py --trials 100 --epochs 30
```

### Ejecución con Docker

```bash
cd "/home/juanchx/Documents/Maestria_IA/desplieje a produccion/labs/Final"
docker build -f Dockerfile.train -t recompra-train .

docker run --rm \
  -v "$(pwd)/data":/data \
  -e HISTORICO_PATH=/data/raw/historico_1000.csv \
  --entrypoint python \
  recompra-train \
  hptuning.py --prepare --trials 100 --epochs 30
```

Resultado: `data/models/best_hparams.json`

---

## HPT — LightGBM (`hptuning_lgbm.py`)

### Split interno durante cada trial

| Rol | Tamaño | Separación temporal |
|---|---|---|
| Train | 100% de features_hpt_train (sin split) | — |
| Test / Eval (métrica del trial) | features_train.parquet completo | Sí (ventana posterior) |

Sin normalización. Sin early stopping (entrena `n_estimators` fijo por trial).

### Ejecución sin Docker

```bash
cd "/home/juanchx/Documents/Maestria_IA/desplieje a produccion/labs/Final/app"

# Paso 1: generar parquets HPT (mismo comando que FNN — solo una vez)
python hptuning.py --prepare --historico ../data/raw/historico_1000.csv

# Paso 2: buscar hiperparámetros LightGBM
python hptuning_lgbm.py --trials 100
```

### Ejecución con Docker

```bash
cd "/home/juanchx/Documents/Maestria_IA/desplieje a produccion/labs/Final"

# Parquets deben existir (generar con paso de FNN primero)
docker run --rm \
  -v "$(pwd)/data":/data \
  -e HISTORICO_PATH=/data/raw/historico_1000.csv \
  --entrypoint python \
  recompra-train \
  hptuning_lgbm.py --trials 100
```

Resultado: `data/models/best_hparams_lgbm.json`

---

## Diferencias FNN vs LightGBM en HPT

| | FNN | LightGBM |
|---|---|---|
| Script | `hptuning.py` | `hptuning_lgbm.py` |
| Parquets | Mismos | Mismos |
| Fechas | Idénticas | Idénticas |
| Split interno | 90/10 aleatorio | 100% train (sin split) |
| Normalización | Sí (Z-score) | No |
| Early stopping | Sí (best val_loss) | No (n_estimators fijo) |
| Métrica objetivo | precision@3 | precision@3 |
| Resultado | `best_hparams.json` | `best_hparams_lgbm.json` |



# Path archivo de caracteristicas 
data/features_store/features_train.parquet

# comando par ejecutar calculo de caracterisicas completas sin docker
cd /home/juanchx/Documents/Maestria_IA/desplieje\ a\ produccion/labs/Final
python app/feature_engineering.py --historico Historico_08122025.csv --workers 16

# Sin Docker — pipeline completo (features + FNN):
python app/entrypoint_train.py --historico Historico_08122025.csv --workers 16 --epochs 50 --force

# Con Docker — features:
docker run --rm -v $(pwd)/data:/data -v $(pwd)/Historico_08122025.csv:/data/raw/historico.csv \
  fnn-train python feature_engineering.py --historico /data/raw/historico.csv --workers 16

# Con Docker — pipeline completo:
docker run --rm -v $(pwd)/data:/data -v $(pwd)/Historico_08122025.csv:/data/raw/historico.csv \
  fnn-train python entrypoint_train.py --historico /data/raw/historico.csv --workers 16 --force

# 3.1 Archivo de filtro para entrenamiento continuo: data/nuevas_series.csv — CSV con columnas CODIGO_FAMILIA;COD_SUBCATEGORIA, Sin Docker — con filtro:
python app/feature_engineering.py --historico Historico_08122025.csv \
  --filtro data/nuevas_series.csv --workers 16

# Con Docker — con filtro:
docker run --rm -v $(pwd)/data:/data -v $(pwd)/Historico_08122025.csv:/data/raw/historico.csv \
  fnn-train python feature_engineering.py --historico /data/raw/historico.csv \
  --filtro /data/nuevas_series.csv --workers 16
-------------------------------------
### ejecucion FNN TUNNING 
cd "/home/juanchx/Documents/Maestria_IA/desplieje a produccion/labs/Final/app"

python hptuning.py \
  --prepare \
  --historico ../data/raw/historico_1000.csv \
  --trials 100 \
  --epochs 30
  ### Si los parquets ya existen (solo buscar):
python hptuning.py --trials 100 --epochs 30

### 3. Ejecución con Docker — FNN
# Desde la raíz del proyecto
cd "/home/juanchx/Documents/Maestria_IA/desplieje a produccion/labs/Final"

docker build -f Dockerfile.train -t recompra-train .

docker run --rm \
  -v "$(pwd)/data":/data \
  -e HISTORICO_PATH=/data/raw/historico_1000.csv \
  --entrypoint python \
  recompra-train \
  hptuning.py --prepare --trials 100 --epochs 30

  ------
  Nada de dudas, voy directo al código.

Tengo todo. Respondo punto por punto con base en el código.

---

## 1. Script ya implementado para HPT FNN

`@/home/juanchx/Documents/Maestria_IA/desplieje a produccion/labs/Final/app/hptuning.py:1-22`

---

## 2. Ejecución sin Docker — FNN

**Paso 1: generar parquets HPT + buscar hiperparámetros** (todo junto):
```bash
cd "/home/juanchx/Documents/Maestria_IA/desplieje a produccion/labs/Final/app"

python hptuning.py \
  --prepare \
  --historico ../data/raw/historico_1000.csv \
  --trials 100 \
  --epochs 30
```

**Si los parquets ya existen** (solo buscar):
```bash
python hptuning.py --trials 100 --epochs 30
```

---

## 3. Ejecución con Docker — FNN

```bash
# Desde la raíz del proyecto
cd "/home/juanchx/Documents/Maestria_IA/desplieje a produccion/labs/Final"

docker build -f Dockerfile.train -t recompra-train .

docker run --rm \
  -v "$(pwd)/data":/data \
  -e HISTORICO_PATH=/data/raw/historico_1000.csv \
  --entrypoint python \
  recompra-train \
  hptuning.py --prepare --trials 100 --epochs 30
```

---

## 4. Fechas — con ejemplo concreto usando `fecha_max = 2025-12-01`

### 4.1 ¿Train + Validation + Test?

Hay **dos splits** distintos:

| Rol | Qué es | Cómo se genera |
|---|---|---|
| **Train (90%)** | Gradiente descend | 90% aleatorio de `features_hpt_train.parquet` |
| **Val. interna (10%)** | Early stopping | 10% aleatorio del MISMO parquet — **mismas fechas, split random** |
| **Test / Eval** | Métrica final del trial (precision@3) | `features_train.parquet` — **ventana temporal diferente** |

No hay separación temporal entre train y validación interna. El único split temporal real es train vs test.

---

### 4.2 Datos de TRAIN (y validación interna) — `features_hpt_train.parquet`

```
prepare_hpt_split corta el CSV:
  historico_hpt_train.csv = transacciones desde inicio hasta (2025-12-01 - 21d) = hasta 2025-11-09

run_pipeline sobre ese CSV con prediction_window=21:
  fecha_max del CSV truncado = 2025-11-09
  fecha_corte_features     = 2025-11-09 - 21d = 2025-10-19

┌─────────────────────────────────────────────────────────┐
│ FEATURES: datos desde inicio histórico hasta 2025-10-19 │
│ TARGET:   ¿compró entre 2025-10-19 y 2025-11-09?        │
└─────────────────────────────────────────────────────────┘
```

Código: `simulation.py:146-148` + `hptuning.py:87-89`

---

### 4.3 Datos de TEST — `features_train.parquet`

```
Este parquet ya existe (es el de entrenamiento final):
  historico completo → fecha_max = 2025-12-01
  run_pipeline con prediction_window=21:
  fecha_corte_features = 2025-12-01 - 21d = 2025-11-10

┌─────────────────────────────────────────────────────────┐
│ FEATURES: datos desde inicio histórico hasta 2025-11-10 │
│ TARGET:   ¿compró entre 2025-11-10 y 2025-12-01?        │
└─────────────────────────────────────────────────────────┘
```

Código: `hptuning.py:57` (`eval_features_path=FEATURES_TRAIN_FILE`)

---

**Resumen visual:**
```
Tiempo ──────────────────────────────────────────────────────────►
         inicio                  2025-10-19   2025-11-10   2025-12-01
         │                           │             │             │
         ├──── features TRAIN ───────┤             │             │
                                     ├── target T ─┤             │
                                     │             │             │
                                     ├── features TEST ──────────┤
                                                   ├── target T ─┤
```