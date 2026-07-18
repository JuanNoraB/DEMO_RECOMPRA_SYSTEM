"""Busqueda simple de horizontes globales para churn.

Se usan dos anios de historico anteriores a una fecha de corte T.
Se prueban dos bases globales dando el mismo peso a cada cliente:

1. mediana_clientes = mediana de las medianas individuales de intervalos.
2. media_clientes = media de las medias individuales de intervalos.

Para cada base B y cada alfa:
    H = alfa