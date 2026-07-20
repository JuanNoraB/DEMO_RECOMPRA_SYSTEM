from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CC = REPO_ROOT / "data" / "raw" / "CC.csv"
DEFAULT_HIST = REPO_ROOT / "data" / "raw" / "HISTORICO_300K_FULL.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw" / "historico_con_demograficos.parquet"
MIN_COMPRAS = 4


def