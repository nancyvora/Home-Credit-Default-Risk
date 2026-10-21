"""Tune: search the LightGBM settings on the development folds, and change them only if the winner wins twice.

The protocol of notebooks 02, 04 and 13:

1. **Search.** Optuna's TPE sampler, 30 trials; trial 0 is the current settings. Each trial grows on 85% of each set of
   training folds at a learning rate of 0.05 and stops when ROC AUC on the other 15% has not improved for 100 trees,
   so the validation fold decides nothing inside a trial. It is scored on the validation folds.
2. **First test.** The best trial must beat trial 0 by all four conditions (``four_conditions``).
3. **Second test.** Both settings are refitted at the final learning rate on all the training-fold rows and scored at
   several numbers of trees. Each takes the number where its own curve is highest, and the winner must again beat the
   current settings by all four conditions.

A search reports the best of many tries, so its winner is partly the best by luck; the second test, under the
conditions the final model is fitted in, guards against that. In notebook 13 the winner passed the first test and
failed the second, so the settings of notebook 02 were kept.

Usage::

    from src.models.tune import tune_model
    result = tune_model(X_dev, y_dev, fold, cfg, cache_dir='artifacts/tune_cache')
    result['params'], result['trees'], result['changed']
"""
import hashlib
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.models.pipeline import SEED, CategoryEncoder, lgbm_model
from src.models.train import text_columns

NOISE_FLOOR = 0.0005                    # notebook 04: how much the score moves with the seed alone
SEARCH_RATE, MAX_TREES, PATIENCE, WATCH_SHARE = 0.05, 4000, 100, 0.15
CHECKPOINTS = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000]


def suggest(trial):
    """The search space of notebooks 02 and 04, with the share of features per tree allowed down to 15% (notebook 13)."""
    return {'num_leaves': trial.suggest_int('num_leaves', 8, 128, log=True), 'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 1000, log=True), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.15, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5), 'cat_smooth': trial.suggest_float('cat_smooth', 1.0, 100.0, log=True),
            'neg_bagging_fraction': trial.suggest_float('neg_bagging_fraction', 0.1, 1.0), 'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0)}


def four_conditions(fold_differences, noise_floor=NOISE_FLOOR):
    """The rule for every change in the project, applied to fold-by-fold differences in ROC AUC."""
    d = np.asarray(fold_differences, dtype=float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    conditions = {'1. mean ROC AUC is higher': bool(d.mean() > 0), '2. higher in at least four of five folds': bool((d > 0).sum() >= 4),
                  '3. gain is more than twice its standard error': bool(d.mean() > 0 and d.mean() > 2 * se),
                  f'4. gain exceeds the noise floor of {noise_floor}': bool(d.mean() > noise_floor)}
    return {'mean_difference': float(d.mean()), 'standard_error': float(se), 'folds_better': int((d > 0).sum()),
            'conditions': conditions, 'passed': all(conditions.values())}


def prepare_folds(X_dev, y_dev, fold, features, text_features, seed=SEED):
    """Per fold: encoded training and validation rows, and the 85/15 split of the training rows used for early stopping."""
    y_dev, fold = np.asarray(y_dev), np.asarray(fold)
    folds = []
    for k in np.unique(fold):
        fit, val = fold != k, fold == k
        encoder = CategoryEncoder(columns=list(text_features)).fit(X_dev[fit])            # levels from the training folds only
        grow, watch = train_test_split(np.arange(int(fit.sum())), test_size=WATCH_SHARE, stratify=y_dev[fit], random_state=seed)
        folds.append({'X_fit': encoder.transform(X_dev[fit])[features], 'y_fit': y_dev[fit], 'X_val': encoder.transform(X_dev[val])[features],
                      'y_val': y_dev[val], 'grow': grow, 'watch': watch})
    return folds


def _key(spec):
    return hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def search_trial(params, folds, features, cache_dir=None):
    """One trial: early-stopped fits on each fold. Cached by settings, features, protocol and library versions."""
    import lightgbm
    import sklearn

    path = None
    if cache_dir is not None:
        key = _key({'params': {k: repr(v) for k, v in sorted(params.items())}, 'features': list(features), 'protocol': [SEARCH_RATE, MAX_TREES, PATIENCE, SEED],
                    'versions': [lightgbm.__version__, sklearn.__version__]})
        path = Path(cache_dir) / f'{key}.joblib'
        if path.exists():
            return joblib.load(path)
    aucs, rounds, seconds = [], [], []
    for d in folds:
        model = lgbm_model(params, MAX_TREES, SEARCH_RATE).set_params(metric='auc', first_metric_only=True)
        started = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(d['X_fit'].iloc[d['grow']], d['y_fit'][d['grow']], eval_set=[(d['X_fit'].iloc[d['watch']], d['y_fit'][d['watch']])], eval_metric='auc',
                      callbacks=[lightgbm.early_stopping(PATIENCE, verbose=False)])
            prediction = model.predict_proba(d['X_val'])[:, 1]
        seconds.append(time.time() - started)
        aucs.append(roc_auc_score(d['y_val'], prediction))
        rounds.append(int(model.best_iteration_))
    result = {'folds': aucs, 'rounds': rounds, 'seconds': float(np.sum(seconds))}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, path, compress=3)
    return result


def run_search(folds, features, current_params, n_trials=30, seed=SEED, cache_dir=None, log=print):
    """Trials as a table (one row per trial; trial 0 is the current settings) and the raw results by trial number."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results = {}

    def objective(trial):
        params = suggest(trial)
        r = search_trial(params, folds, features, cache_dir)
        results[trial.number] = {**r, 'params': params}
        log(f"  trial {trial.number:>2}: ROC AUC {np.mean(r['folds']):.5f}, {np.mean(r['rounds']):.0f} trees")
        return float(np.mean(r['folds']))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(dict(current_params))
    study.optimize(objective, n_trials=n_trials)
    trials = pd.DataFrame({n: {'roc_auc': np.mean(r['folds']), 'fold_sd': np.std(r['folds'], ddof=1), 'mean_trees': np.mean(r['rounds']), **r['params']}
                           for n, r in results.items()}).T
    trials.index.name = 'trial'
    return trials, results


def tree_curve(params, folds, features, learning_rate, checkpoints=CHECKPOINTS, cache_dir=None):
    """ROC AUC on each validation fold at each number of trees, fitted on all the training-fold rows."""
    path = None
    if cache_dir is not None:
        key = _key({'p': {k: repr(v) for k, v in sorted(params.items())}, 'f': list(features), 'c': list(checkpoints), 'rate': learning_rate, 'seed': SEED})
        path = Path(cache_dir) / f'trees_{key}.joblib'
    if path is not None and path.exists():
        curve = joblib.load(path)
    else:
        curve = {}
        for k, d in enumerate(folds):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model = lgbm_model(params, max(checkpoints), learning_rate).fit(d['X_fit'], d['y_fit'])
                curve[k] = {n: roc_auc_score(d['y_val'], model.predict_proba(d['X_val'], num_iteration=n)[:, 1]) for n in checkpoints}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(curve, path)
    curve = pd.DataFrame(curve)
    curve.columns = [f'fold {k}' for k in curve.columns]
    curve['mean'] = curve.mean(axis=1)
    curve.index.name = 'trees'
    return curve


def tune_model(X_dev, y_dev, fold, cfg, n_trials=30, cache_dir=None, log=print):
    """The whole protocol. Returns the settings and number of trees to use, and the evidence for them."""
    features = cfg.features
    current, trees_now, rate = dict(cfg['model']['lightgbm']), int(cfg['model']['trees']), float(cfg['model']['learning_rate'])
    search_dir = None if cache_dir is None else Path(cache_dir) / 'search'
    folds = prepare_folds(X_dev, y_dev, fold, features, text_columns(X_dev))
    log(f'search: {n_trials} trials')
    trials, results = run_search(folds, features, current, n_trials, cache_dir=search_dir, log=log)
    best_n = int(trials['roc_auc'].astype(float).idxmax())
    first = four_conditions(np.array(results[best_n]['folds']) - np.array(results[0]['folds']))
    first['passed'] = first['passed'] and best_n != 0
    log(f"first test, trial {best_n} against the current settings: {first['mean_difference']:+.5f} -> {'PASSED' if first['passed'] else 'NOT PASSED'}")

    candidate = dict(results[best_n]['params'])
    log(f'tree curves at learning rate {rate}')
    cur = tree_curve(current, folds, features, rate, cache_dir=cache_dir)
    cand = tree_curve(candidate, folds, features, rate, cache_dir=cache_dir) if best_n != 0 else cur
    fold_cols = [c for c in cur.columns if c.startswith('fold')]
    current_best, candidate_best = int(cur['mean'].idxmax()), int(cand['mean'].idxmax())
    second = four_conditions(cand.loc[candidate_best, fold_cols].astype(float).values - cur.loc[current_best, fold_cols].astype(float).values)
    second['passed'] = second['passed'] and first['passed']
    log(f"second test, {candidate_best:,} against {current_best:,} trees: {second['mean_difference']:+.5f} -> {'PASSED' if second['passed'] else 'NOT PASSED'}")

    if second['passed']:
        params, trees = candidate, candidate_best
    else:
        # notebook 04's rule: the current settings keep their number of trees unless their own curve beats it
        # by more than the noise floor in at least four folds
        more = cur.loc[current_best, fold_cols].astype(float).values - cur.loc[trees_now, fold_cols].astype(float).values
        params, trees = current, (current_best if (more.mean() > NOISE_FLOOR and (more > 0).sum() >= 4) else trees_now)
    return {'params': params, 'trees': int(trees), 'changed': params != current or trees != trees_now, 'best_trial': best_n,
            'first_test': first, 'second_test': second, 'trials': trials, 'curves': {'current settings': cur, f'trial {best_n}': cand}}


if __name__ == '__main__':
    import argparse

    import yaml

    from src.data.load_data import load_joined
    from src.utils.utils import DEFAULT_CONFIG, PROJECT_ROOT, load_config, write_json

    parser = argparse.ArgumentParser(description='Search the settings on the development folds (two-test rule).')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--trials', type=int, default=30)
    parser.add_argument('--cache-dir', default=str(PROJECT_ROOT / 'artifacts' / 'tune_cache'),
                        help='finished fits are kept here, so an interrupted run resumes; artifacts/cv_cache_13 holds notebook 13\'s')
    args = parser.parse_args()
    cfg = load_config(args.config)

    X = load_joined(cfg.path('processed_dir'), 'train', cfg.model_tables)
    y = X.pop('TARGET').to_numpy()
    split = pd.read_parquet(cfg.path('split'))
    assert split.index.equals(X.index), 'the split and the feature table list different applicants'
    dev = (split['split'] == 'dev').to_numpy()
    started = time.time()
    result = tune_model(X.loc[dev], y[dev], split.loc[dev, 'fold'].to_numpy(), cfg, n_trials=args.trials, cache_dir=args.cache_dir)
    print(f'tuning: {(time.time() - started) / 60:.1f} minutes')

    out_dir = PROJECT_ROOT / 'artifacts' / 'reports' / 'tuning'
    out_dir.mkdir(parents=True, exist_ok=True)
    result['trials'].to_csv(out_dir / 'trials.csv')
    for label, curve in result['curves'].items():
        curve.to_csv(out_dir / f"tree_curve_{label.replace(' ', '_')}.csv")
    write_json({k: v for k, v in result.items() if k not in ('trials', 'curves')}, out_dir / 'result.json')
    if result['changed']:
        # the config is never overwritten: a tuned copy is written beside it, to be evaluated before use
        tuned = dict(cfg)
        tuned['model'] = {**cfg['model'], 'trees': result['trees'], 'lightgbm': result['params']}
        path = Path(args.config).with_name(f'{Path(args.config).stem}_tuned.yaml')
        path.write_text(yaml.safe_dump(tuned, sort_keys=False), encoding='utf-8')
        print(f'the settings should change: written {path}, for review')
    else:
        print(f"the settings stay as they are: {result['trees']:,} trees")
    print(f'written: {out_dir}')
