"""Shared fixtures.

Most tests build small synthetic applicants from the schema and typical values saved with the served model
(``src/serving/model``), so they run without the Kaggle files. Tests marked ``data`` use the real files and are skipped
when ``data/raw`` or ``data/processed`` is absent (for example in CI).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW, PROCESSED, MODEL_DIR = ROOT / 'data' / 'raw', ROOT / 'data' / 'processed', ROOT / 'src' / 'serving' / 'model'
# SKIP_DATA_TESTS=1 runs the suite as CI does, without the Kaggle files
HAS_RAW = (RAW / 'application_train.csv').exists() and not os.environ.get('SKIP_DATA_TESTS')
HAS_PROCESSED = (PROCESSED / 'application_train_features.parquet').exists() and (PROCESSED / 'application_split.parquet').exists()


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason='needs the Kaggle files in data/raw and data/processed')
    for item in items:
        if 'data' in item.keywords and not (HAS_RAW and HAS_PROCESSED):
            item.add_marker(skip)


SCHEMA = json.loads((MODEL_DIR / 'input_schema.json').read_text(encoding='utf-8'))['tables']
TYPICAL = json.loads((MODEL_DIR / 'typical_applicant.json').read_text(encoding='utf-8'))['values']


def make_applications(n=20, target=False, seed=0, **overrides):
    """``n`` synthetic application rows with every raw column: typical values, varied ids and amounts."""
    rng = np.random.default_rng(seed)
    columns = list(SCHEMA['application']) + (['TARGET'] if target else [])
    rows = pd.DataFrame({c: [TYPICAL.get(c, np.nan)] * n for c in columns})
    rows['SK_ID_CURR'] = np.arange(100_000, 100_000 + n)
    rows['AMT_CREDIT'] = rng.integers(50_000, 2_000_000, n).astype(float)
    rows['AMT_ANNUITY'] = rows['AMT_CREDIT'] / rng.integers(10, 40, n)
    rows['AMT_GOODS_PRICE'] = rows['AMT_CREDIT'] * 0.9
    rows['DAYS_BIRTH'] = -rng.integers(20 * 365, 65 * 365, n)
    rows['EXT_SOURCE_1'], rows['EXT_SOURCE_2'], rows['EXT_SOURCE_3'] = rng.uniform(0.05, 0.95, (3, n))
    if target:
        rows['TARGET'] = (np.arange(n) % 5 == 0).astype(int)
    for c, v in overrides.items():
        rows[c] = v
    return rows


@pytest.fixture
def applications():
    return make_applications()


@pytest.fixture(scope='session')
def raw_sample():
    """The first 3,000 rows of application_train.csv."""
    return pd.read_csv(RAW / 'application_train.csv', nrows=3000)
