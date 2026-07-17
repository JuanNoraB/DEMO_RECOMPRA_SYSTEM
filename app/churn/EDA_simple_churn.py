"""EDA simple de intervalos de compra para churn a nivel cliente.

Lee solo CODIGO_FAMILIA y DIM_PERIODO, conserva los dos anios mas
recientes del historico y genera una tabla estadistica sencilla.
No construye el target ni entrena modelos.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


REPO_ROOT