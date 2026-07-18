"""Calibracion empirica de un horizonte dinamico para churn a nivel cliente.

Usa dos anios de historico anteriores a una fecha de corte T y el periodo
posterior disponible como seguimiento.

Para cada cliente con al menos un intervalo se calcula una base B (media o
mediana). Para cada alfa se define H_i = alfa * B_i. La comprobacion posterior
usa xB * B_i, donde xB es configurable.

La salida