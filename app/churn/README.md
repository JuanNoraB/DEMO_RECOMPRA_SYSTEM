# Experimento de churn

Esta carpeta contiene scripts independientes del pipeline de recompra. Cada etapa se
ejecuta y revisa antes de construir la siguiente.

## 01. Diagnóstico inicial

El primer script inspecciona el Parquet sin construir todavía features, horizonte ni
target. Usa DuckDB para ejecutar agregaciones paralelas sin cargar las transacciones
completas en pandas.

```bash
python app/churn/01_diagnostico_inicial.py \
  --input /ruta/HISTORICO_300K_FULL.parquet \
  --output-dir data/churn/diagnostico_inicial \
  --threads 128 \
  --memory-limit 220GB
```

Salidas principales por ejecución:

- `diagnostico.log`: avance, tiempos y resultados esenciales.
- `resumen.json`: resumen reproducible de la ejecución.
- `schema.csv`, `null_profile.csv` y `numeric_profile.csv`.
- perfiles mensuales, de clientes y de intervalos en CSV.
- seis gráficas PNG listas para revisar.

La GPU no se utiliza en esta etapa. El diagnóstico es una carga de lectura y
agregación que aprovecha CPU, RAM y el paralelismo interno de DuckDB.

