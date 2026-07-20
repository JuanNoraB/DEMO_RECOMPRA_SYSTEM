"""Join de datos demograficos (CC.csv) con el historico usado para churn.

Pasos:
1. Lee CC.csv (IDENTIFICACION, FECHA_NACIMIENTO, SEXO), tolerando lineas corruptas.
2. Limpia IDENTIFICACION y la conserva como string.
3. Convierte FECHA_NACIMIENTO y calcula EDAD.
4. Deduplica IDENTIFICACION priorizando registros con fecha de nacimiento valida.
5. Obtiene del historico exactamente la ventana de dos anios usada en