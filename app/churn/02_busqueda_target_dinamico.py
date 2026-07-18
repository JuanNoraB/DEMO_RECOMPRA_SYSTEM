"""Busqueda de horizonte hibrido para definir churn en un entorno no contractual.

Se usan dos anios de historico anteriores a una fecha de corte T.
La unidad de analisis es IDENTIFICACION.

Solo se incluyen clientes con al menos cuatro fechas distintas de compra en el
periodo historico, de modo que la mediana individual se estime con al menos tres
intervalos entre compras.

Para cada cliente elegible se calcula su mediana individual de intervalos. La
referencia global B es el promedio de las medianas individuales, por lo que cada
cliente aporta
