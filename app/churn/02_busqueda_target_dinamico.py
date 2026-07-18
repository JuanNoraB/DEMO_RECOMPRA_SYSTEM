"""Busqueda de horizonte hibrido para definir churn en un entorno no contractual.

Se usan dos anios de historico anteriores a una fecha de corte T.

Para cada cliente se calcula su mediana individual de intervalos entre compras.
La referencia global B es el promedio de las medianas individuales, de modo que
cada cliente aporta el mismo peso al comportamiento central del