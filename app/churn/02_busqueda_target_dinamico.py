"""Busqueda simple de horizontes globales para churn.

Se usan dos anios de historico anteriores a una fecha de corte T.
Primero se calculan los intervalos de compra de cada cliente y luego se resume
cada cliente por separado. A partir de esos resumenes se construyen dos bases
globales, dando el mismo peso a cada cliente:

- media_clientes: promedio de la media de intervalos de cada cliente.
- mediana_clientes: mediana de la mediana de intervalos de cada cliente.

Para cada base B y cada alfa:
    H = alfa * B

Clasificacion:
- no_churn: compra dentro de (T, T + H]
- churn_provisional: no compra dentro de H
- churn_reactivado