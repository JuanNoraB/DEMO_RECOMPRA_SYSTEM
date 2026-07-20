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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Join demografico para historico de churn")
    p.add_argument("--cc", type=Path, default=DEFAULT_CC)
    p.add_argument("--historico", type=Path, default=DEFAULT_HIST)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def cargar_cc(path: Path) -> pd.Data