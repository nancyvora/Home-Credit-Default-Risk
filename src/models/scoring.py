"""Score loan applications with the saved final model.

The model is stored as plain files in ``artifacts/final_model``:

- ``model.txt``      the LightGBM trees
- ``metadata.json``  feature order, levels of the text features, calibration, decision threshold

Nothing here depends on the training notebook. Only numpy, pandas and lightgbm are needed.

The input is a table of engineered features, one row per application, as produced by
``notebooks/01_application_train_eda.ipynb`` (``data/processed/application_*_features.parquet``).
Extra columns are ignored and column order does not matter.

Usage::

    from scoring import load_model
    model = load_model('artifacts/final_model')
    probability = model.predict_proba(features)      # probability of payment difficulties
    declined = model.decide(features)                # True where the application is declined
"""
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


class ScoringModel:
    def __init__(self, booster, metadata):
        self.booster = booster
        self.metadata = metadata
        self.features = list(metadata['features'])
        self.text_levels = {c: list(v) for c, v in metadata['text_feature_levels'].items()}
        self.slope = float(metadata['calibration']['slope'])
        self.intercept = float(metadata['calibration']['intercept'])
        self.threshold = float(metadata['decision']['reference_threshold'])

    def prepare(self, features):
        """Put the columns in the order the model was trained on and encode them the same way."""
        missing = [c for c in self.features if c not in features.columns]
        if missing:
            raise ValueError(f"columns the model needs are not in the input: {missing}")
        out = pd.DataFrame(index=features.index)
        for c in self.features:
            if c in self.text_levels:
                values = features[c].astype(object)
                # a level that was not in the training data is treated as a missing value
                values = values.where(values.isin(self.text_levels[c]))
                out[c] = pd.Categorical(values, categories=self.text_levels[c])
            else:
                out[c] = features[c].astype(float)
        return out

    def predict_proba(self, features):
        """Calibrated probability of payment difficulties, one value per row."""
        raw = np.clip(self.booster.predict(self.prepare(features)), 1e-7, 1 - 1e-7)
        log_odds = self.slope * np.log(raw / (1 - raw)) + self.intercept
        return 1 / (1 + np.exp(-log_odds))

    def decide(self, features, threshold=None):
        """True where the application is declined: predicted probability at or above the threshold."""
        threshold = self.threshold if threshold is None else threshold
        return self.predict_proba(features) >= threshold


def load_model(model_dir):
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / 'metadata.json').read_text(encoding='utf-8'))
    model_path = model_dir / metadata['files']['model']
    found = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if found != metadata['files']['model_sha256']:
        raise ValueError(f"{model_path.name} does not match the checksum recorded in metadata.json")
    return ScoringModel(lgb.Booster(model_file=str(model_path)), metadata)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Score a table of engineered application features.')
    parser.add_argument('model_dir')
    parser.add_argument('features', help='parquet file indexed by SK_ID_CURR')
    parser.add_argument('output', help='csv file to write, with columns SK_ID_CURR and TARGET')
    args = parser.parse_args()

    table = pd.read_parquet(args.features)
    scores = load_model(args.model_dir).predict_proba(table)
    pd.DataFrame({'SK_ID_CURR': table.index.values, 'TARGET': scores}).to_csv(args.output, index=False)
    print(f"{len(scores)} rows scored")
