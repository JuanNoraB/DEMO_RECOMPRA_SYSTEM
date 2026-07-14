"""Diagnóstico pareado FNN vs LightGBM GAIN95 sobre la validación HPT.
No reentrena modelos y no utiliza W4/test.
"""
from __future__ import annotations

import argparse, json, math, os, shutil, time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from