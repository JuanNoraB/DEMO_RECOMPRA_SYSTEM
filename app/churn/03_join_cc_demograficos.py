"""Une datos demograficos con el historico usado para las features de churn.

El periodo historico se calcula con exactamente la misma regla temporal del script 02:
    T = fecha_max - 1 anio
    inicio = T - 2 anios + 1 dia

Se conserva solo [inicio, T] para construir posteriormente las caracteristicas.
El join es LEFT: no se elimina ninguna transaccion del historico por falta de datos
demogra