"""EDA simple de intervalos de compra para churn a nivel cliente.

Lee solo CODIGO_FAMILIA y DIM_PERIODO, conserva los dos anios mas
recientes del historico y genera:
1. Una tabla estadistica general.
2. Una tabla por terciles excluyentes.
3. Tres graficos de distribucion.
4. Correlacion de Spearman entre cuatro variables de ciclo.
5. PCA de dos componentes sobre esas cuatro variables.

No construye el target ni entrena modelos.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as