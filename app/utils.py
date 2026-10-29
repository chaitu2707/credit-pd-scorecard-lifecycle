"""Read saved outputs for the dashboard. Nothing here trains or refits a model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TABLES = ROOT / "reports" / "tables"
MODELS = ROOT / "models"


def table(name: str):
    path = TABLES / name
    return pd.read_csv(path) if path.exists() else None


def load_json(name: str):
    path = MODELS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def oot_predictions():
    path = TABLES / "oot_predictions.parquet"
    return pd.read_parquet(path) if path.exists() else None
