"""The modelling pipeline shared by the notebooks.

The classes and the two small functions marked "from notebook 02" are exact copies of the code developed and tested in
``notebooks/02_model_selection_training.ipynb``. ``notebooks/04_model_with_bureau.ipynb`` checks, every time it runs,
that their source here is still identical to the source there.

Everything that learns from data is a scikit-learn step, so that inside cross-validation it is fitted on the training
folds only:

- ``CategoryEncoder``   learns the levels of each text column
- ``KeepColumns``       keeps a fixed list of columns (learns nothing)
- ``CalibratedModel``   fits the model, and a calibrator on predictions for rows the model has not seen
"""
import hashlib
import inspect
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

SEED = 42
N_SPLITS = 5

# from notebook 02
def score_predictions(y_true, y_prob):
    y_true, y_prob = np.asarray(y_true), np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    return {'roc_auc': roc_auc_score(y_true, y_prob), 'pr_auc': average_precision_score(y_true, y_prob),
            'log_loss': log_loss(y_true, y_prob), 'brier': brier_score_loss(y_true, y_prob)}


# from notebook 02
class CategoryEncoder(BaseEstimator, TransformerMixin):
    # Encodes text columns using the levels seen in fit. Numeric columns pass through as floats.
    # output: 'category' (pandas categorical), 'code' (float codes, missing stays missing), 'string' (text, missing becomes '(missing)')
    def __init__(self, columns, output='category'):
        self.columns = columns
        self.output = output

    def fit(self, X, y=None):
        self.categories_ = {c: sorted(X[c].dropna().astype(str).unique()) for c in self.columns}
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        out = X.copy()
        for c in out.columns:
            if c not in self.columns:
                out[c] = out[c].astype(float)
                continue
            values = out[c].astype(object)
            values = values.where(values.isin(self.categories_[c]))   # unseen levels become missing, like missing values
            encoded = pd.Categorical(values, categories=self.categories_[c])
            if self.output == 'category':
                out[c] = encoded
            elif self.output == 'code':
                codes = encoded.codes.astype(float)
                codes[codes < 0] = np.nan
                out[c] = codes
            else:
                text = np.asarray(encoded.astype(object))
                out[c] = pd.Series(text, index=out.index).where(pd.notna(text), '(missing)').astype(str)
        return out

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


# from notebook 02
class KeepColumns(BaseEstimator, TransformerMixin):
    # keeps a fixed list of columns
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"columns not in the data: {missing}")
        self.columns_ = list(self.columns)      # recorded so that scikit-learn knows the step has been fitted
        return self
    def transform(self, X):
        return X[self.columns_]


# from notebook 02
def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


# from notebook 02
class CalibratedModel(ClassifierMixin, BaseEstimator):
    # A model followed by a calibration step.
    #   'prior'    divides the predicted odds by a fixed number. Nothing is fitted.
    #   'platt'    fits a logistic curve on the model's log-odds.
    #   'isotonic' fits a monotone step function.
    # For the fitted methods, the calibrator is trained on predictions for rows the model has not seen, obtained by
    # cross-validating the model inside the rows passed to fit. The model is then refitted on all of those rows.
    def __init__(self, model, method='platt', odds_multiplier=1.0, inner_folds=4, seed=SEED):
        self.model = model
        self.method = method
        self.odds_multiplier = odds_multiplier
        self.inner_folds = inner_folds
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        if self.method in ('platt', 'isotonic'):
            unseen = np.empty(len(y))
            for fit_idx, cal_idx in StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed).split(X, y):
                unseen[cal_idx] = clone(self.model).fit(X.iloc[fit_idx], y[fit_idx]).predict_proba(X.iloc[cal_idx])[:, 1]
            if self.method == 'platt':
                self.calibrator_ = LogisticRegression(C=1e6, max_iter=1000).fit(logit(unseen).reshape(-1, 1), y)
            else:
                self.calibrator_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip').fit(unseen, y)
        elif self.method != 'prior':
            raise ValueError(self.method)
        self.model_ = clone(self.model).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.model_.predict_proba(X)[:, 1]
        if self.method == 'prior':
            q = p / (p + self.odds_multiplier * (1 - p))
        elif self.method == 'platt':
            q = self.calibrator_.predict_proba(logit(p).reshape(-1, 1))[:, 1]
        else:
            q = self.calibrator_.predict(p)
        return np.column_stack([1 - q, q])


# from notebook 02
def describe(obj):
    # a stable, text description of a pipeline and all its settings, used to recognise a run that was done before
    if hasattr(obj, 'get_params'):
        return {'class': type(obj).__name__, 'params': {k: describe(v) for k, v in sorted(obj.get_params(deep=False).items())}}
    if isinstance(obj, (list, tuple)):
        return [describe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): describe(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, np.ndarray):
        return hashlib.md5(obj.tobytes()).hexdigest()
    if callable(obj):
        return getattr(obj, '__name__', type(obj).__name__)
    return repr(obj)


def lgbm_model(params, n_estimators, learning_rate):
    # every default is kept at each round (pos_bagging_fraction=1) and repaid loans are sampled at neg_bagging_fraction
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=SEED, verbose=-1, learning_rate=learning_rate, n_estimators=n_estimators,
                          pos_bagging_fraction=1.0, bagging_freq=1, **params)


def make_pipeline(features, text_features, model):
    # encode the text columns, keep the listed features, then the model. Same structure as with_columns() in notebook 02.
    return Pipeline([('preprocess', Pipeline([('encode', CategoryEncoder(columns=list(text_features))), ('keep', KeepColumns(list(features)))])),
                     ('model', model)])


def _fingerprint(X, y, fold):
    h = hashlib.md5()
    h.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    h.update(''.join(X.columns).encode())
    h.update(np.asarray(y).tobytes())
    h.update(np.asarray(fold).tobytes())
    return h.hexdigest()


def cache_key(pipeline, X, y, fold, fit_params=None):
    import lightgbm
    import sklearn
    spec = {'pipeline': describe(pipeline), 'fit_params': describe(fit_params or {}), 'data': _fingerprint(X, y, fold),
            'versions': [sklearn.__version__, lightgbm.__version__, pd.__version__],
            'code': hashlib.md5(Path(__file__).read_bytes()).hexdigest()}
    return hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def cross_validate(pipeline, X, y, fold, cache_dir=None, fit_params=None):
    """Score a pipeline on fixed folds. Each fold's model is a fresh copy fitted on the other folds only.

    A finished run is saved in cache_dir and reloaded when the same pipeline is scored again on the same data and folds
    with the same library versions and the same version of this file.
    """
    y, fold = np.asarray(y), np.asarray(fold)
    assert len(X) == len(y) == len(fold)
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{cache_key(pipeline, X, y, fold, fit_params)}.joblib"
        if path.exists():
            saved = joblib.load(path)
            return {**saved, 'mean': saved['folds'].mean(), 'std': saved['folds'].std(ddof=1), 'cached': True}

    oof, rows = np.full(len(X), np.nan), []
    for k in np.unique(fold):
        fit_rows, val_rows = fold != k, fold == k
        model = clone(pipeline)                                             # a fresh, unfitted copy for every fold
        started = time.time()
        model.fit(X[fit_rows], y[fit_rows], **(fit_params or {}))           # training folds only
        seconds = time.time() - started
        oof[val_rows] = model.predict_proba(X[val_rows])[:, 1]
        rows.append({'fold': int(k), **score_predictions(y[val_rows], oof[val_rows]), 'fit_seconds': seconds})
    assert np.isfinite(oof).all()
    folds = pd.DataFrame(rows).set_index('fold')
    result = {'folds': folds, 'oof': oof, 'overall': score_predictions(y, oof)}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, path, compress=3)
    return {**result, 'mean': folds.mean(), 'std': folds.std(ddof=1), 'cached': False}


def compare(result, reference):
    """Fold-by-fold difference in ROC AUC between two results scored on the same folds."""
    diff = result['folds']['roc_auc'] - reference['folds']['roc_auc']
    standard_error = diff.std(ddof=1) / np.sqrt(len(diff))
    return {'mean_difference': float(diff.mean()), 'standard_error': float(standard_error), 'folds_better': int((diff > 0).sum()),
            'smallest': float(diff.min()), 'largest': float(diff.max()),
            'ratio_to_standard_error': float(diff.mean() / standard_error) if standard_error > 0 else float('inf')}
