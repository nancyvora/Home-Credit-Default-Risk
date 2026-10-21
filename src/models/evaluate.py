"""Evaluate: measure the recipe on the development folds, and report any set of predictions.

- ``cross_validate_model``   five-fold scores of the whole recipe; each fold's model, calibration included, is fitted on
                             the other four folds only. The holdout is never used.
- ``evaluate_model``         ranking (ROC AUC, PR AUC), probabilities (log loss, Brier, calibration slope and
                             intercept, worst tenth) and decisions at the threshold (confusion matrix, share declined,
                             defaults caught, default rate among approved)
- ``holdout_report``         ``evaluate_model`` plus a 95% bootstrap interval for ROC AUC and, given a reference
                             model's predictions on the same rows, a paired interval for the gain over it

The measures are those of notebooks 02 and 13.

Usage::

    from src.models.evaluate import cross_validate_model, evaluate_model
    cv = cross_validate_model(X_dev, y_dev, fold, cfg)
    report = evaluate_model(y_dev, cv['oof'], threshold=cfg['decision']['reference_threshold'])
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score

from src.models.pipeline import cross_validate, logit
from src.models.train import build_model, text_columns


def cross_validate_model(X_dev, y_dev, fold, cfg, cache_dir=None):
    """Five-fold scores of the calibrated recipe: the dict of ``pipeline.cross_validate`` (folds, oof, mean, cached)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return cross_validate(build_model(cfg, text_columns(X_dev)), X_dev, np.asarray(y_dev), np.asarray(fold), cache_dir=cache_dir)


def calibration_line(y, p):
    """Slope and intercept of a logistic fit of the outcome on the predicted log-odds: 1 and 0 when calibrated."""
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit(np.asarray(p)).reshape(-1, 1), y)
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def reliability(y, p, bins=10):
    """Predicted and actual default rate (%) in equal-sized groups of predicted risk."""
    p, y = pd.Series(np.asarray(p)), pd.Series(np.asarray(y))
    groups = pd.qcut(p.rank(method='first'), bins, labels=False)
    return pd.DataFrame({'predicted_pct': p.groupby(groups).mean() * 100, 'actual_pct': y.groupby(groups).mean() * 100, 'rows': y.groupby(groups).size()})


def decisions(y, p, threshold):
    """Declining every applicant at or above the threshold: the confusion matrix and what it means."""
    y, declined = np.asarray(y), np.asarray(p) >= threshold
    (tn, fp), (fn, tp) = confusion_matrix(y, declined, labels=[0, 1])
    return {'declined %': declined.mean() * 100, 'defaults caught %': tp / (tp + fn) * 100,
            'default rate among approved %': fn / (fn + tn) * 100, 'default rate among declined %': tp / (tp + fp) * 100,
            'confusion matrix': {'approved, repaid': int(tn), 'declined, would have repaid': int(fp),
                                 'approved, defaulted': int(fn), 'declined, would have defaulted': int(tp)}}


def evaluate_model(y, p, threshold):
    """Ranking, probability and decision measures for one set of predictions."""
    y, p = np.asarray(y), np.asarray(p)
    slope, intercept = calibration_line(y, p)
    tenths = reliability(y, p)
    return {'ROC AUC': roc_auc_score(y, p), 'PR AUC': average_precision_score(y, p), 'log loss': log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
            'Brier score': brier_score_loss(y, p), 'average prediction %': p.mean() * 100, 'actual default rate %': y.mean() * 100,
            'calibration slope': slope, 'calibration intercept': intercept,
            'largest gap in a tenth (pts)': float((tenths['predicted_pct'] - tenths['actual_pct']).abs().max()), **decisions(y, p, threshold)}


def bootstrap_auc(y, p, p_reference=None, n_boot=1000, seed=42):
    """ROC AUC on ``n_boot`` resamples of the rows; with a reference, its ROC AUC on the same resamples."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    out = np.full((n_boot, 2), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        out[b, 0] = roc_auc_score(y[idx], p[idx])
        if p_reference is not None:
            out[b, 1] = roc_auc_score(y[idx], np.asarray(p_reference)[idx])
    return out


def holdout_report(y, p, threshold, p_reference=None, n_boot=1000, seed=42):
    """``evaluate_model`` plus a 95% bootstrap interval for ROC AUC and, with a reference, for the gain over it."""
    result = evaluate_model(y, p, threshold)
    boot = bootstrap_auc(y, p, p_reference, n_boot, seed)
    result['ROC AUC 95% interval'] = [float(v) for v in np.percentile(boot[:, 0], [2.5, 97.5])]
    if p_reference is not None:
        result['reference ROC AUC'] = float(roc_auc_score(np.asarray(y), np.asarray(p_reference)))
        result['gain over reference'] = result['ROC AUC'] - result['reference ROC AUC']
        result['gain 95% interval'] = [float(v) for v in np.percentile(boot[:, 0] - boot[:, 1], [2.5, 97.5])]
    return result


if __name__ == '__main__':
    import argparse
    import time
    from pathlib import Path

    from src.data.load_data import load_joined
    from src.utils.utils import DEFAULT_CONFIG, PROJECT_ROOT, load_config, write_json

    parser = argparse.ArgumentParser(description='Cross-validate the recipe of a config on the five development folds.')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--cache-dir', default=str(PROJECT_ROOT / 'artifacts' / 'cv_cache'), help='finished runs are kept here and reloaded')
    args = parser.parse_args()
    cfg = load_config(args.config)

    X = load_joined(cfg.path('processed_dir'), 'train', cfg.model_tables)
    y = X.pop('TARGET').to_numpy()
    split = pd.read_parquet(cfg.path('split'))
    assert split.index.equals(X.index), 'the split and the feature table list different applicants'
    dev = (split['split'] == 'dev').to_numpy()
    started = time.time()
    cv = cross_validate_model(X.loc[dev], y[dev], split.loc[dev, 'fold'].to_numpy(), cfg, cache_dir=args.cache_dir)
    print(f"{cfg['name']}: {'loaded from cache' if cv['cached'] else f'{(time.time() - started) / 60:.1f} minutes'}")
    print(cv['folds'][['roc_auc', 'pr_auc', 'log_loss', 'brier']].to_string(float_format=lambda v: f'{v:.5f}'))
    print(f"mean ROC AUC {cv['mean']['roc_auc']:.5f}")
    report = evaluate_model(y[dev], cv['oof'], cfg['decision']['reference_threshold'])
    for k, v in report.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    out = write_json({'folds': cv['folds'].to_dict(orient='index'), 'mean': cv['mean'].to_dict(), 'out_of_fold': report},
                     PROJECT_ROOT / 'artifacts' / 'reports' / f'{Path(args.config).stem}_cross_validation.json')
    print(f'written: {out}')
