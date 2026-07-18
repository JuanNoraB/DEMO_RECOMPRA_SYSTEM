"""Calibracion empirica de un horizonte dinamico para churn a nivel cliente.

Se usan dos anios de historico y el periodo posterior disponible como seguimiento.
Para cada cliente con al menos un intervalo se calcula una base B: media o mediana.
Para cada alfa: H_i = alfa * B_i.
La validacion posterior usa xB * B_i, donde xB es configurable.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT