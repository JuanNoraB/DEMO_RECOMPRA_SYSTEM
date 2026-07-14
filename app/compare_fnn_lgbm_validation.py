"""Comparación final FNN vs LightGBM GAIN95 sobre W4.

Usa las predicciones ya guardadas por ambos modelos finales. No reentrena,
no vuelve a inferir y no modifica los artefactos originales.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg