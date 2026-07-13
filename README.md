# Sistema de Recomendación de Recompra (FNN)

Sistema de predicción de recompra por familia/subcategoría usando una red neuronal feedforward (FNN) en PyTorch, desplegado en Kubernetes con reentrenamiento automático.

## Arquitectura

- **Inference API** (FastAPI) — sirve predicciones en `/predict/{cedula}`
- **Training CronJob** — reentrena cada 3 min si hay ≥20 transacciones nuevas
- **Prometheus** — recolecta métricas de la API
- **Grafana** — dashboards de monitoreo
- **Kafka** — broker de eventos (pod activo, producer en API)

## Requisitos

- Docker
- Minikube
- kubectl

## Levantar el sistema

```bash
# 1. Iniciar minikube
minikube start --memory=4096 --cpus=4

# 2. Construir imágenes dentro de minikube
eval $(minikube docker-env)
docker build -f Dockerfile.inference -t fnn-inference:latest .
docker build -f Dockerfile.train -t fnn-train:latest .

# 3. Copiar datos a minikube
minikube cp data/raw/historico_1000_base.csv /data/raw/historico_1000_base.csv
minikube cp data/raw/historico_1000_sim.csv /data/raw/historico_1000_sim.csv

# 4. Desplegar todo
kubectl apply -f k8s/kafka.yaml
kubectl apply -f k8s/monitoring.yaml
kubectl apply -f k8s/inference-deployment.yaml
kubectl apply -f k8s/training-job.yaml

# 5. Port-forwards para acceso local
kubectl port-forward svc/fnn-inference-svc 8000:8000 &
kubectl port-forward svc/prometheus-svc 9090:9090 &
kubectl port-forward svc/grafana-svc 3000:3000 &
```

## URLs

| Servicio | URL | Credenciales |
|---|---|---|
| API | http://localhost:8000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

## Comandos útiles

```bash
# Ver pods corriendo
kubectl get pods

# Ver estado del CronJob y ejecuciones
kubectl get cronjobs
kubectl get jobs

# Ver logs del último training
kubectl logs job/<nombre-del-job>

# Simular nuevas transacciones (triggea reentrenamiento)
kubectl exec deploy/fnn-inference -- python simulate_ingest.py --days 7

# Forzar reentrenamiento manual con todo el histórico
kubectl exec deploy/fnn-inference -- python entrypoint_train.py \
  --historico /data/raw/historico_1000_base.csv --epochs 30 --workers 8 --force

# Health check de la API
curl http://localhost:8000/health

# Predicción
curl "http://localhost:8000/predict/401779061?top_n=3"
```

## Métricas Prometheus

| Métrica | Descripción |
|---|---|
| `requests_total` | Total de requests por status |
| `request_latency_seconds` | Latencia de inferencia (histograma) |

## Flujo de reentrenamiento

```
simulate_ingest.py --days N
        ↓
tx_counter.json acumula nuevas transacciones
        ↓
CronJob cada 3 min: ¿total_new >= 20?
   NO → sale sin entrenar
   SÍ → feature engineering → FNN training → resetea contador
```

## Observación metodológica sobre ventanas deslizantes y balanceo

Durante la generación de ventanas deslizantes con `app/sliced_windows.py`, la unidad de selección no debe entenderse como transacción individual, sino como serie completa a nivel `CODIGO_FAMILIA` + `COD_SUBCATEGORIA`. Esto es importante porque una serie puede tener múltiples transacciones históricas; eliminar registros aislados rompería la secuencia temporal usada para construir variables de recencia, frecuencia, ciclos, unidades, participación y estacionalidad.

La implementación actual de `sliced_windows.py` define las claves de serie como `KEY_COLS = ["CODIGO_FAMILIA", "COD_SUBCATEGORIA"]`. Para las ventanas `i > 0`, identifica primero los pares familia–subcategoría que compraron en la ventana objetivo, construye un pool de pares no compradores y selecciona una muestra de no compradores con relación `neg_ratio:1`. Después une esos pares seleccionados contra el histórico de la ventana mediante `merge(..., on=KEY_COLS, how="inner")`. Por tanto, cuando una serie se conserva, se conservan sus transacciones dentro del periodo de la ventana; cuando una serie se excluye, se excluye la serie completa familia–subcategoría.

Sin embargo, se observó una limitación relevante: el balanceo `neg_ratio=4` se aplica antes de `feature_engineering.py`, sobre el universo de pares detectados en el histórico crudo. Luego, `feature_engineering.py` recalcula features y target por familia/subcategoría a partir del histórico resultante. Por esta razón, la proporción final de `target=1` y `target=0` en los parquets de features puede no quedar exactamente en relación 1:4. En pruebas posteriores, algunas ventanas generadas con esta lógica terminaron con ratios cercanos a 1:6–1:7, aunque la intención inicial era 1:4.

Mejora futura recomendada: si se requiere un balance exacto para entrenamiento o análisis de importancia de características, el submuestreo debe aplicarse después de generar el parquet de features, usando la columna `target` final ya calculada. Alternativamente, `sliced_windows.py` debería exportar explícitamente un archivo de filtro de series familia–subcategoría y `feature_engineering.py` debería procesar únicamente esas series mediante `--filtro`, validando al final la proporción real de `target` en el parquet generado.
