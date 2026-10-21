"""Inference: score applicants with the final model, with the features built exactly as in training.

The model is stored as plain files in ``src/serving/model``:

- ``model.txt``           the LightGBM trees (3,500, fitted on all 307,511 training applicants in notebook 12)
- ``metadata.json``       the 111 features in order, the levels of the text features, the calibration, the threshold,
                          and the checksum of ``model.txt``, checked on loading
- ``input_schema.json``   the raw columns a request may carry, per table, with their types
- ``holdout_evaluation.json``  how the same recipe scored on the 61,503 holdout applicants (notebook 13)

**Training / serving consistency.** A request carries the applicant's raw records: the application row and, if there
are any, their credit bureau records, earlier applications, instalment payments, POS / cash balances and credit card
balances. They go through the same ``src/data/preprocess.py`` and ``src/features/build_features.py`` code that built
the training tables, so a live score is the score the model would give in batch (checked on test applicants).

Usage::

    from src.serving.inference import predict
    result = predict(application={'SK_ID_CURR': 100001, 'AMT_CREDIT': 568800.0, ...},
                     history={'bureau': [...], 'previous': [...], 'installments': [...]})
    result['probability'], result['decision'], result['reasons']

    from src.serving.inference import load_model
    probabilities = load_model().predict_proba(feature_table)      # batch, from saved feature tables
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.preprocess import preprocess
from src.features.build_features import build_features

MODEL_DIR = Path(__file__).resolve().parent / 'model'
HISTORY_TABLES = ['bureau', 'previous', 'installments', 'pos_cash', 'credit_card']


class ScoringModel:
    """The saved booster with its calibration and decision rule. Needs only lightgbm, numpy and pandas."""

    def __init__(self, booster, metadata):
        self.booster = booster
        self.metadata = metadata
        self.features = list(metadata['features'])
        self.text_levels = {c: list(v) for c, v in metadata['text_feature_levels'].items()}
        self.slope = float(metadata['calibration']['slope'])
        self.intercept = float(metadata['calibration']['intercept'])
        self.threshold = float(metadata['decision']['reference_threshold'])

    def prepare(self, features):
        """The model's columns in training order, text columns encoded with the training levels (an unseen level is missing)."""
        missing = [c for c in self.features if c not in features.columns]
        if missing:
            raise ValueError(f'columns the model needs are not in the input: {missing}')
        columns = {}
        for c in self.features:
            if c in self.text_levels:
                values = features[c].astype(object)
                columns[c] = pd.Categorical(values.where(values.isin(self.text_levels[c])), categories=self.text_levels[c])
            else:
                columns[c] = pd.to_numeric(features[c], errors='coerce').astype(float)
        return pd.DataFrame(columns, index=features.index)

    def predict_proba(self, features):
        """Calibrated probability of payment difficulties, one value per row."""
        raw = np.clip(self.booster.predict(self.prepare(features)), 1e-7, 1 - 1e-7)
        return 1 / (1 + np.exp(-(self.slope * np.log(raw / (1 - raw)) + self.intercept)))

    def decide(self, probability, threshold=None):
        """True where the application is declined: probability at or above the threshold."""
        return np.asarray(probability) >= (self.threshold if threshold is None else threshold)

    def explain(self, features, top=5):
        """Per row, the features that moved the risk most, as odds multipliers (above 1 raises the risk).

        Exact contributions of each feature to the model's log-odds (TreeSHAP, computed by LightGBM), scaled by the
        calibration slope, so the multipliers combine to the calibrated odds.
        """
        prepared = self.prepare(features)
        contributions = self.booster.predict(prepared, pred_contrib=True)[:, :-1] * self.slope
        out = []
        for i in range(len(prepared)):
            order = np.argsort(-np.abs(contributions[i]))[:top]
            out.append([{'feature': self.features[j], 'value': _plain(features.iloc[i][self.features[j]]),
                         'odds_multiplier': round(float(np.exp(contributions[i, j])), 4),
                         'direction': 'raises risk' if contributions[i, j] > 0 else 'lowers risk'} for j in order])
        return out


def _plain(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NA:
        return None
    return value.item() if isinstance(value, np.generic) else value


@lru_cache(maxsize=4)
def load_model(model_dir=MODEL_DIR):
    """Load the saved model; refuses a ``model.txt`` that does not match the checksum in ``metadata.json``."""
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / 'metadata.json').read_text(encoding='utf-8'))
    model_path = model_dir / metadata['files']['model']
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata['files']['model_sha256']:
        raise ValueError(f'{model_path} does not match the checksum recorded in metadata.json')
    return ScoringModel(lgb.Booster(model_file=str(model_path)), metadata)


@lru_cache(maxsize=1)
def input_schema(model_dir=MODEL_DIR):
    return json.loads((Path(model_dir) / 'input_schema.json').read_text(encoding='utf-8'))['tables']


def _frame(table, records, applicant_id, model_dir=MODEL_DIR):
    """Records of one table as a DataFrame with every raw column of that table, typed as in the raw files."""
    schema = input_schema(model_dir)[table]
    frame = pd.DataFrame(list(records), columns=list(schema))
    if 'SK_ID_CURR' in frame:
        frame['SK_ID_CURR'] = frame['SK_ID_CURR'].fillna(applicant_id) if len(frame) else frame['SK_ID_CURR']
    for c, kind in schema.items():
        frame[c] = pd.to_numeric(frame[c], errors='coerce') if kind == 'number' else frame[c].astype(object).where(frame[c].notna(), np.nan)
    if len(frame) and 'SK_ID_CURR' in frame:
        frame['SK_ID_CURR'] = frame['SK_ID_CURR'].astype('int64')
    return frame


def applicant_features(application, history=None, model_dir=MODEL_DIR):
    """One applicant's raw records to their row of model features, through the training preprocessing and features."""
    history = history or {}
    unknown = set(history) - set(HISTORY_TABLES)
    if unknown:
        raise ValueError(f'unknown history tables: {sorted(unknown)}; known: {HISTORY_TABLES}')
    if application.get('SK_ID_CURR') is None:
        raise ValueError('the application needs SK_ID_CURR')
    applicant_id = int(application['SK_ID_CURR'])
    for table, records in history.items():
        other = {r.get('SK_ID_CURR') for r in records} - {None, applicant_id}
        if other:
            raise ValueError(f'{table}: records of other applicants {sorted(other)}')
    features = build_features('application', preprocess('application', _frame('application', [application], applicant_id, model_dir)))
    features['REGION_RATING_CLIENT_W_CITY'] = features['REGION_RATING_CLIENT_W_CITY'].astype(float)
    ids = features.index
    for table in HISTORY_TABLES:
        rows = _frame(table, history.get(table, []), applicant_id, model_dir)
        features = features.join(build_features(table, preprocess(table, rows), ids), how='left')
    return features


def predict(application, history=None, threshold=None, top_reasons=5, model_dir=MODEL_DIR):
    """Score one applicant from their raw records. Returns probability, decision and the main reasons."""
    model = load_model(model_dir)
    features = applicant_features(application, history, model_dir)
    probability = float(model.predict_proba(features)[0])
    threshold = model.threshold if threshold is None else float(threshold)
    return {'SK_ID_CURR': int(features.index[0]), 'probability': round(probability, 6),
            'decision': 'decline' if probability >= threshold else 'approve', 'threshold': threshold,
            'reasons': model.explain(features, top=top_reasons)[0]}


def predict_table(features, model_dir=MODEL_DIR):
    """Batch: probabilities for a table of already-built features (e.g. the saved test feature tables, joined)."""
    return load_model(model_dir).predict_proba(features)


def write_submission(ids, probability, sample_submission, path):
    """A Kaggle submission in the order of the sample file; read back and checked after writing."""
    scores = pd.Series(np.asarray(probability), index=pd.Index(ids, name='SK_ID_CURR'))
    expected = sample_submission['SK_ID_CURR']
    if set(scores.index) != set(expected):
        raise ValueError('the applicants scored are not the applicants of the sample submission')
    frame = pd.DataFrame({'SK_ID_CURR': expected.values, 'TARGET': scores.loc[expected.values].values})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    back = pd.read_csv(path)
    if not (list(back.columns) == ['SK_ID_CURR', 'TARGET'] and len(back) == len(expected) and (back['SK_ID_CURR'].values == expected.values).all()
            and back['TARGET'].between(0, 1, inclusive='neither').all() and np.allclose(back['TARGET'].values, frame['TARGET'].values, rtol=0, atol=1e-12)):
        raise RuntimeError(f'{path} did not read back as written')
    return frame


if __name__ == '__main__':
    import argparse

    from src.data.load_data import load_joined, read_raw

    parser = argparse.ArgumentParser(description='Write a Kaggle submission from the saved test feature tables.')
    parser.add_argument('--processed-dir', default='data/processed')
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--output', default='artifacts/submissions/submission_from_serving.csv')
    args = parser.parse_args()
    tables = list(dict.fromkeys(['application', *load_model().metadata['feature_sources']]))
    names = {'application': 'application', 'credit bureau': 'bureau', 'previous applications': 'previous', 'instalment payments': 'installments',
             'credit card': 'credit_card'}
    X_test = load_joined(args.processed_dir, 'test', [names[t] for t in tables])
    frame = write_submission(X_test.index, predict_table(X_test), read_raw('sample_submission', args.raw_dir), args.output)
    print(f"{len(frame):,} applicants scored, average {frame['TARGET'].mean():.4f}; written: {args.output}")
