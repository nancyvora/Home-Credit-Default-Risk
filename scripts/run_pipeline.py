#!/usr/bin/env python3
"""Run the whole pipeline in order: load -> validate -> preprocess -> features -> evaluate -> train -> test -> submit.

    python scripts/run_pipeline.py                         # everything from the raw files, about 45 minutes
    python scripts/run_pipeline.py --skip-features         # use the feature tables the notebooks saved in data/processed
    python scripts/run_pipeline.py --cv-cache artifacts/cv_cache_12 --skip-features   # notebook 12's cross-validation: about 20 minutes
    python scripts/run_pipeline.py --tune                  # also run the settings search (adds 1 to 4 hours)
    python scripts/run_pipeline.py --mlflow                # also log the run to MLflow (artifacts/mlflow/mlflow.db)

Stages:

1. **load and validate** the raw files (``src/utils/validate_data.py``); the run stops if any check fails
2. **preprocess and build features** for every table (``src/data/preprocess.py``, ``src/features/build_features.py``)
   into ``artifacts/pipeline/features``, and create or check the frozen split (it is never overwritten)
3. **validate** the feature table the model will see
4. **tune** (optional): the settings search with the two-test rule; it never changes the config
5. **evaluate**: five-fold cross-validation of the recipe on the development rows
6. **train** on the development rows, and **test** once on the holdout against the application-only reference model.
   Nothing is chosen from the holdout: the recipe is fixed by the config before it is opened.
7. **train** on all of train and write the Kaggle submission

Everything the run writes goes to ``artifacts/pipeline/``. The served model in ``src/serving/model`` is replaced only
with ``--deploy``, and only after the new model reproduces the cross-validated and holdout results.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.load_data import load_joined, read_raw
from src.data.preprocess import load_or_create_split
from src.features.build_features import build_all
from src.models.evaluate import cross_validate_model, holdout_report
from src.models.train import save_model, text_columns, train_model
from src.models.tune import tune_model
from src.serving.inference import MODEL_DIR, load_model, write_submission
from src.utils.utils import DEFAULT_CONFIG, load_config, setup_logger, write_json
from src.utils.validate_data import validate_features, validate_raw

REFERENCE_CONFIG = ROOT / 'configs' / 'application_only.yaml'


def flat(report, prefix):
    """Numeric entries of a report as MLflow metric names."""
    return {f"{prefix}_{k.replace(' ', '_').replace('%', 'pct').replace('(', '').replace(')', '')}": float(v)
            for k, v in report.items() if isinstance(v, (int, float, np.floating))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--out-dir', default=str(ROOT / 'artifacts' / 'pipeline'))
    parser.add_argument('--skip-features', action='store_true', help='use the feature tables already in processed_dir')
    parser.add_argument('--cv-cache', default=str(ROOT / 'artifacts' / 'cv_cache'), help='finished cross-validation runs are kept here')
    parser.add_argument('--tune', action='store_true', help='run the settings search first (long)')
    parser.add_argument('--tune-cache', default=str(ROOT / 'artifacts' / 'tune_cache'))
    parser.add_argument('--mlflow', action='store_true', help='log parameters, metrics and outputs to MLflow')
    parser.add_argument('--deploy', action='store_true', help='copy the all-train model into src/serving/model if it passes the checks')
    args = parser.parse_args(argv)

    cfg, out = load_config(args.config), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = setup_logger('pipeline', out / 'pipeline.log')
    threshold = cfg['decision']['reference_threshold']
    results, started_all = {'config': str(Path(args.config).name)}, time.time()

    run = None
    if args.mlflow:
        import mlflow
        mlflow.set_tracking_uri('sqlite:///' + (ROOT / 'artifacts' / 'mlflow' / 'mlflow.db').as_posix())
        name = 'home-credit-default-risk'
        if mlflow.get_experiment_by_name(name) is None:
            mlflow.create_experiment(name, artifact_location=(ROOT / 'artifacts' / 'mlflow' / 'runs').as_uri())
        mlflow.set_experiment(name)
        run = mlflow.start_run()
        mlflow.log_params({'config': Path(args.config).name, 'features': len(cfg.features), 'trees': cfg['model']['trees'],
                           'learning_rate': cfg['model']['learning_rate'], 'threshold': threshold, **cfg['model']['lightgbm']})

    try:
        # 1. load and validate
        t = time.time()
        passed, failed = validate_raw(cfg.path('raw_dir'), log=lambda m: log.info(f'validate raw: {m}'))
        results['raw_data_valid'] = passed
        if not passed:
            raise SystemExit(f'raw data checks failed: {failed}')
        log.info(f'1. raw data valid ({time.time() - t:.0f} s)')

        # 2. preprocess and features
        t = time.time()
        # rebuilt tables go to the run's own folder; data/processed keeps the tables the notebooks saved and checksummed
        features_dir = cfg.path('processed_dir') if args.skip_features else out / 'features'
        if not args.skip_features:
            build_all(cfg.path('raw_dir'), features_dir, cfg['tables'], log=lambda m: log.info(f'features: {m}'))
        X = load_joined(features_dir, 'train', cfg.model_tables)
        y = X.pop('TARGET').to_numpy()
        s = cfg['split']
        split = load_or_create_split(cfg.path('split'), X.index.values, y, s['holdout_fraction'], s['n_folds'], s['seed'])
        dev, holdout = (split['split'] == 'dev').to_numpy(), (split['split'] == 'holdout').to_numpy()
        fold = split.loc[dev, 'fold'].to_numpy()
        log.info(f'2. features ready: {X.shape[1]} columns; {dev.sum():,} development and {holdout.sum():,} holdout applicants ({time.time() - t:.0f} s)')

        # 3. the feature table the model will see
        passed, failed = validate_features(X, cfg.features, text_columns(X))
        if not passed:
            raise SystemExit(f'feature table checks failed: {failed}')
        log.info('3. feature table valid')

        # 4. tune (optional)
        if args.tune:
            t = time.time()
            tuned = tune_model(X.loc[dev], y[dev], fold, cfg, cache_dir=args.tune_cache, log=lambda m: log.info(f'tune: {m}'))
            results['tuning'] = {k: v for k, v in tuned.items() if k not in ('trials', 'curves')}
            log.info(f"4. tuning: settings {'should change (see configs/*_tuned.yaml)' if tuned['changed'] else 'stay as they are'} ({(time.time() - t) / 60:.1f} min)")

        # 5. evaluate on the development folds
        t = time.time()
        cv = cross_validate_model(X.loc[dev], y[dev], fold, cfg, cache_dir=args.cv_cache)
        results['cross_validation'] = {'folds': cv['folds']['roc_auc'].round(5).tolist(), **{k: float(cv['mean'][k]) for k in ('roc_auc', 'pr_auc', 'log_loss', 'brier')}}
        log.info(f"5. cross-validated ROC AUC {cv['mean']['roc_auc']:.5f} (folds {cv['folds']['roc_auc'].min():.5f} to {cv['folds']['roc_auc'].max():.5f}) "
                 f"({'cached' if cv['cached'] else f'{(time.time() - t) / 60:.1f} min'})")

        # 6. train on the development rows; test once on the holdout, against the application-only reference
        t = time.time()
        model_dev = train_model(X.loc[dev], y[dev], cfg)
        save_model(model_dev, cfg, out / 'model_dev', rows=int(dev.sum()), defaults=int(y[dev].sum()), extra={'trained_on': 'dev'})
        reference_cfg = load_config(REFERENCE_CONFIG)
        reference = train_model(X.loc[dev], y[dev], reference_cfg)
        p_hold, p_ref = model_dev.predict_proba(X.loc[holdout])[:, 1], reference.predict_proba(X.loc[holdout])[:, 1]
        report = holdout_report(y[holdout], p_hold, threshold, p_reference=p_ref)
        results['holdout'] = report
        log.info(f"6. holdout ROC AUC {report['ROC AUC']:.4f} (95% interval {report['ROC AUC 95% interval'][0]:.4f} to {report['ROC AUC 95% interval'][1]:.4f}); "
                 f"gain over the application-only model {report['gain over reference']:+.4f}; "
                 f"average prediction {report['average prediction %']:.2f}% against {report['actual default rate %']:.2f}% ({(time.time() - t) / 60:.1f} min)")

        # 7. train on all of train; submission
        t = time.time()
        model_all = train_model(X, y, cfg)
        save_model(model_all, cfg, out / 'model_all', rows=len(X), defaults=int(y.sum()),
                   extra={'trained_on': 'all', 'evidence': {'cross_validation': results['cross_validation'],
                                                            'holdout': {k: report[k] for k in ('ROC AUC', 'ROC AUC 95% interval', 'PR AUC', 'log loss', 'gain over reference')}}})
        X_test = load_joined(features_dir, 'test', cfg.model_tables)
        p_test = load_model(out / 'model_all').predict_proba(X_test)
        write_submission(X_test.index, p_test, read_raw('sample_submission', cfg.path('raw_dir')), out / 'submission.csv')
        same_as_pipeline = float(np.abs(p_test - model_all.predict_proba(X_test)[:, 1]).max())
        results['submission'] = {'rows': len(p_test), 'mean': float(p_test.mean()), 'saved_model_matches_pipeline': same_as_pipeline < 1e-9}
        log.info(f'7. trained on all {len(X):,} applicants; submission written, mean {p_test.mean():.4f} ({(time.time() - t) / 60:.1f} min)')

        # compare with the model being served
        served = load_model(MODEL_DIR)
        if served.metadata['features'] == cfg.features:
            gap = float(np.abs(served.predict_proba(X_test) - p_test).max())
            results['matches_served_model'] = gap < 1e-9
            log.info(f'served model in src/serving/model: test predictions differ by at most {gap:.2e}')
        if args.deploy:
            if not results['submission']['saved_model_matches_pipeline']:
                raise SystemExit('not deployed: the saved model does not reproduce the pipeline')
            for f in ('model.txt', 'metadata.json'):
                shutil.copy2(out / 'model_all' / f, MODEL_DIR / f)
            load_model.cache_clear()
            log.info(f'deployed to {MODEL_DIR}')
    finally:
        results['minutes'] = round((time.time() - started_all) / 60, 1)
        write_json(results, out / 'results.json')
        if run is not None:
            import mlflow
            if 'cross_validation' in results:
                mlflow.log_metrics({f"cv_{k}": v for k, v in results['cross_validation'].items() if k != 'folds'})
            if 'holdout' in results:
                mlflow.log_metrics(flat(results['holdout'], 'holdout'))
            mlflow.log_metric('raw_data_valid', int(results.get('raw_data_valid', False)))
            for path in [out / 'results.json', out / 'pipeline.log', out / 'model_all' / 'metadata.json', out / 'model_all' / 'model.txt']:
                if path.exists():
                    mlflow.log_artifact(str(path))
            mlflow.end_run()
    log.info(f"done in {results['minutes']} minutes; results in {out / 'results.json'}")


if __name__ == '__main__':
    main()
