"""Busqueda simple de horizontes globales para churn.

Se usan dos anios de historico anteriores a una fecha de corte T.

Para cada cliente se calculan sus intervalos entre compras y luego:
- media_cliente: media de sus intervalos.
- mediana_cliente: mediana de sus intervalos.

Se prueban dos bases globales dando el mismo peso a cada cliente:
- media_clientes: promedio de las medias individuales.