"""Busqueda empirica del horizonte dinamico para churn a nivel cliente.

Objetivo
--------
Calibrar un horizonte individual de abandono usando dos bases temporales:
media y mediana de los intervalos historicos de cada cliente.

Para cada cliente i y cada alfa:
    H_i = alfa * B_i

donde B_i es la media o mediana de sus intervalos historicos.

El experimento usa:
- Dos anios de historico anteriores a una fecha de corte T.
- Una ventana