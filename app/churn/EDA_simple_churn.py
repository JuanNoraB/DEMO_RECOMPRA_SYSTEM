"""EDA simple para churn a nivel de cliente.

- Usa los dos anios mas recientes del historico.
- Agrupa por cliente y fecha distinta de compra.
- Calcula intervalos entre compras consecutivas.
- Resume los intervalos sin construir target ni entrenar modelos.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "HISTORICO_300K_FULL.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "churn" / "logs"
REQUIRED_COLUMNS