# Arquitectura de Producción — Predicción de Compras

## Componentes

| Componente | Implementación | Estado |
|---|---|---|
| **Ingesta** | `simulate_ingest.py` CLI manual | Listo |
| **Training** | CronJob K8s cada 3min, incremental con filtro | Listo |
| **Inference** | FastAPI + hot-reload cada 2min | Listo |
| **Web UI** | HTML estático servido desde FastAPI | Pendiente |
| **Kafka** | Bus de eventos (tx.nuevas, training.completed, predictions.log) | Pendiente |
| **Prometheus** | Métricas custom (app) + infra (CPU/RAM/disco) | Pendiente |
| **Grafana** | 4 dashboards: Modelo, Recursos, Inference, Kafka | Pendiente |
| **GitHub Actions** | CI: tests + build Docker images | Pendiente |

## Kafka — Topics

| Topic | Productor | Propósito |
|---|---|---|
| `tx.nuevas` | simulate_ingest | Series afectadas por nueva ingesta |
| `training.completed` | entrypoint_train | Métricas del modelo (precision@3, hit_rate@3, duración, etc.) |
| `predictions.log` | inference_api | Cada predicción servida (cédula, subcats, latencia) |

## Prometheus — Métricas

### Inference API (scrape /metrics)
- `inference_request_total` (Counter)
- `inference_request_duration_seconds` (Histogram → p50/p95/p99)
- `inference_errors_total` (Counter)
- `inference_model_reload_total` (Counter)
- `inference_families_loaded` (Gauge)

### Training (push a Pushgateway)
- `model_precision_at_3`, `model_hit_rate_at_3`, `model_recall_at_3` (Gauge)
- `model_val_loss`, `model_val_accuracy` (Gauge)
- `training_duration_seconds`, `feature_eng_duration_seconds` (Gauge)

### Infraestructura (node-exporter + kube-state-metrics)
- CPU, RAM, disco por pod

## Grafana — Dashboards

1. **Modelo**: precision@3, hit_rate@3, recall@3, val_loss histórico
2. **Recursos**: CPU, RAM, disco por pod (training vs inference)
3. **Inference**: latencia p50/p95/p99, requests/s, errores
4. **Kafka**: throughput por topic, eventos recientes

## Quality Gate

Si precision@3 < 0.7 → modelo NO se despliega → alerta.

## GitHub Actions

```yaml
on: [push, pull_request]
jobs:
  test:
    - Feature engineering con sample data
    - Training 10 epochs → verifica modelo + meta
    - API: /health OK, /predict/{cedula} OK
  build:
    - docker build train + inference
```
