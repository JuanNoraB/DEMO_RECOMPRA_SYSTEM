"""Calibracion empirica de un horizonte dinamico para churn a nivel cliente.

Se usan dos anios de historico y el periodo posterior disponible como seguimiento.
Para cada cliente con al menos un intervalo se calcula una base B:
media o mediana de sus intervalos historicos.

Para cada alfa:
    H_i = alfa * B_i

Clasificacion:
- no_churn: se observa una compra dentro de (T, T + H_i]
- churn_provisional: H_i es completamente observable y no hay compra en H_i
- reactivado: churn provisional que compra antes de T + H_i