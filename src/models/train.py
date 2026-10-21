"""Train: fit the recipe of the config, with calibration, and save it in the format ``serving`` reads.

The recipe (notebooks 02 to 13), read from ``configs/final_model.yaml``:

1. text columns become category codes, with the levels learned from the training rows only;
2. the 111 features are kept;
3. LightGBM with the settings of notebook 02, 3,500 trees at a learning rate of 0.02;
4. Platt calibration, fitted on predictions for training rows the model has not seen (4 inner folds), after which the
   model is refitted on all the training rows.

Steps 1 to 4 are one scikit-learn pipeline from ``src/models/pipeline.py``, the module every notebook used, so inside
cross-validation everything that learns from data is fitted on the training folds only.

Usage::

    from src.models.train import train_model, save_model
    model = train_model(X_dev, y_dev, cfg)            # the development rows: the model the holdout is scored with
    save_model(model, cfg, 'artifacts/models/final_dev', rows=len(X_dev), defaults=int(y_dev.sum()))
"""
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.pipeline import SEED, CalibratedModel, lgbm_model, make_pipeline


def text_columns(X):
    """The text columns of a feature table; the pipeline encodes them as categories."""
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def build_model(cfg, text_features):
    """The unfitted pipeline the config describes."""
    m = cfg['model']
    if m['seed'] != SEED:
        raise ValueError(f"src/models/pipeline.py fixes the seed at {SEED}; the config says {m['seed']}")
    if m['calibration'] != 'platt':
        raise ValueError('only Platt calibration is supported by the saved model format')
    model = lgbm_model(dict(m['lightgbm']), int(m['trees']), float(m['learning_rate']))
    return make_pipeline(cfg.features, text_features, CalibratedModel(model, method='platt'))


def train_model(X, y, cfg):
    """Fit the recipe on the rows given. ``X`` may hold extra columns; only the configured features are used."""
    model = build_model(cfg, text_columns(X))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return model.fit(X, np.asarray(y))


def save_model(model, cfg, out_dir, rows, defaults, extra=None):
    """Write ``model.txt`` (the trees) and ``metadata.json`` (features, text levels, calibration, threshold, checksum).

    Only lightgbm, numpy and pandas are needed to load it again (``src/serving/inference.py``).
    """
    import lightgbm
    import sklearn

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = model.named_steps['model']
    step.model_.booster_.save_model(str(out_dir / 'model.txt'))
    levels = model.named_steps['preprocess'].named_steps['encode'].categories_
    metadata = {
        'name': cfg['name'],
        'predicts': 'probability that the applicant has payment difficulties (TARGET = 1)',
        'files': {'model': 'model.txt', 'model_sha256': hashlib.sha256((out_dir / 'model.txt').read_bytes()).hexdigest()},
        'features': cfg.features,
        'feature_sources': {table: list(columns) for table, columns in cfg['features'].items()},
        'text_feature_levels': {c: list(levels[c]) for c in cfg.features if c in levels},
        'calibration': {'method': 'Platt scaling', 'slope': float(step.calibrator_.coef_[0, 0]), 'intercept': float(step.calibrator_.intercept_[0]),
                        'formula': 'probability = sigmoid(slope * log(p / (1 - p)) + intercept), p = LightGBM output clipped to [1e-7, 1 - 1e-7]'},
        'decision': {'rule': 'decline when the probability is at or above the threshold',
                     'reference_threshold': float(cfg['decision']['reference_threshold']),
                     'thresholds_by_cost_ratio': {str(k): float(v) for k, v in cfg['decision']['thresholds_by_cost_ratio'].items()}},
        'training': {'rows': int(rows), 'defaults': int(defaults), 'trees': int(cfg['model']['trees']), 'learning_rate': float(cfg['model']['learning_rate']),
                     'seed': SEED, 'lightgbm_settings': dict(cfg['model']['lightgbm'])},
        'versions': {'python': sys.version.split()[0], 'lightgbm': lightgbm.__version__, 'pandas': pd.__version__, 'numpy': np.__version__,
                     'scikit-learn': sklearn.__version__},
        **(extra or {}),
    }
    (out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


if __name__ == '__main__':
    import argparse

    from src.data.load_data import load_joined
    from src.utils.utils import DEFAULT_CONFIG, PROJECT_ROOT, load_config
    from src.utils.validate_data import validate_features

    parser = argparse.ArgumentParser(description='Fit the recipe of a config and save the model.')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--rows', choices=['dev', 'all'], required=True,
                        help='dev: the development rows (the holdout can then be scored); all: all of train (for the submission)')
    parser.add_argument('--out-dir', default=None, help='default: artifacts/models/<config>_<rows>')
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / 'artifacts' / 'models' / f'{Path(args.config).stem}_{args.rows}'

    X = load_joined(cfg.path('processed_dir'), 'train', cfg.model_tables)
    y = X.pop('TARGET').to_numpy()
    passed, failed = validate_features(X, cfg.features, text_columns(X))
    if not passed:
        raise SystemExit(f'feature table check failed: {failed}')
    if args.rows == 'dev':
        split = pd.read_parquet(cfg.path('split'))
        assert split.index.equals(X.index), 'the split and the feature table list different applicants'
        dev = (split['split'] == 'dev').to_numpy()
        X, y = X.loc[dev], y[dev]
    started = time.time()
    model = train_model(X, y, cfg)
    note = ('Fitted on the development rows only; the holdout can be scored with it.' if args.rows == 'dev'
            else 'Fitted on all of train, holdout included; it has not itself been scored on the holdout.')
    meta = save_model(model, cfg, out_dir, rows=len(X), defaults=int(y.sum()), extra={'trained_on': args.rows, 'note': note})
    print(f"{cfg['name']}: fitted on {len(X):,} applicants in {(time.time() - started) / 60:.1f} minutes; "
          f"calibration {meta['calibration']['slope']:.4f}, {meta['calibration']['intercept']:.4f}; written to {out_dir}")
