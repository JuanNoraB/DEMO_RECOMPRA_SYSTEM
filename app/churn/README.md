# Experimento de churn

Los scripts de esta carpeta trabajan a nivel `CODIGO_FAMILIA` y se ejecutan paso a
paso. El historico esperado se encuentra en:

`data/raw/HISTORICO_300K_FULL.parquet`

## 01. Diagnostico inicial

Analiza solamente los dos anios finales del historico. No limpia nuevamente el
dataset y todavia no construye el target.

```bash
python app/churn/01_diagnostico_inicial.py --threads 128 --memory-limit 220GB
```

Genera seis artefactos en `data/churn/diagnostico_inicial`:

- `diagnostico.log` y `resumen.json`.
- `actividad_mensual.csv` e `intervalos_compra.csv`.
- `actividad_mensual.png` e `intervalos_compra.png`.

La GPU no se utiliza en este diagnostico; se reservara para el entrenamiento.

