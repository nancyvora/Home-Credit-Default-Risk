"""Validate data: checks that must pass before anything is built or trained, using Great Expectations.

Three levels:

- ``validate_raw_files``     every raw file present, the competition's row counts, unique ids, no applicant in both
                             train and test (reads only headers and key columns)
- ``validate_application``   the application table: schema, ids, target, ranges, allowed text values
- ``validate_table``         a related table (bureau, previous, installments, pos_cash, credit_card): keys, amounts,
                             and the **time rule**: every record is dated before the current application. The
                             feature modules rely on it; a record dated on or after the application would leak the
                             future into the features.
- ``validate_features``      a feature table before training or scoring: every model feature present, ids unique,
                             no infinite values

Each ``validate_*`` returns ``(passed, failed)``, ``failed`` listing the expectations that did not hold, with a few of
the offending values. The facts encoded here come from the EDA notebooks (01, 03, 05, 07, 09, 11).

Usage::

    from src.utils.validate_data import validate_application
    passed, failed = validate_application(application_train, which='train')
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import RAW_FILES, read_raw

logging.getLogger('great_expectations').setLevel(logging.ERROR)

DAYS_EMPLOYED_PLACEHOLDER = 365243          # "not employed" (notebook 01); made missing by preprocessing
OLDEST_DAYS, YOUNGEST_DAYS = -25_550, -6_570  # 70 and 18 years before the application

# ---------------------------------------------------------------- raw files

KEYS = {
    'application_train': ['SK_ID_CURR', 'TARGET'],
    'application_test': ['SK_ID_CURR'],
    'bureau': ['SK_ID_CURR', 'SK_ID_BUREAU'],
    'previous': ['SK_ID_CURR', 'SK_ID_PREV'],
    'installments': ['SK_ID_CURR', 'SK_ID_PREV'],
    'pos_cash': ['SK_ID_CURR', 'SK_ID_PREV'],
    'credit_card': ['SK_ID_CURR', 'SK_ID_PREV'],
    'sample_submission': ['SK_ID_CURR', 'TARGET'],
}
# the competition's files as downloaded; a different count means a different or damaged file
EXPECTED_ROWS = {'application_train': 307_511, 'application_test': 48_744, 'sample_submission': 48_744}
# columns in the raw application files: all of them feed the count of missing values (N_MISSING)
APPLICATION_COLUMNS = {'train': 122, 'test': 121}


def validate_raw_files(raw_dir):
    """File-level checks. Returns ``(passed, failed, summary)``; reads only headers and key columns."""
    raw_dir = Path(raw_dir)
    failed, summary, ids = [], {}, {}
    for name, filename in RAW_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            failed.append(f'{filename}: missing')
            continue
        columns = list(pd.read_csv(path, nrows=0).columns)
        absent = [c for c in KEYS[name] if c not in columns]
        if absent:
            failed.append(f'{filename}: key columns missing {absent}')
            continue
        keys = read_raw(name, raw_dir, KEYS[name])
        summary[name] = {'rows': len(keys), 'columns': len(columns), 'applicants': int(keys['SK_ID_CURR'].nunique()),
                         'size_mb': round(path.stat().st_size / 1e6, 1)}
        if name in EXPECTED_ROWS and len(keys) != EXPECTED_ROWS[name]:
            failed.append(f'{filename}: {len(keys):,} rows, expected {EXPECTED_ROWS[name]:,}')
        if name.startswith('application'):
            ids[name] = set(keys['SK_ID_CURR'])
        if name == 'application_train':
            summary[name]['defaults'] = int((keys['TARGET'] == 1).sum())
    if len(ids) == 2 and ids['application_train'] & ids['application_test']:
        failed.append('an applicant is in both application_train and application_test')
    return not failed, failed, summary


# ---------------------------------------------------------------- Great Expectations

def _run(df, name, expectations):
    """Validate ``df`` against a list of expectations. Returns ``(passed, failed)``."""
    import great_expectations as gx

    context = gx.get_context(mode='ephemeral')
    try:
        context.variables.progress_bars = gx.data_context.types.base.ProgressBarsConfig(globally=False)
    except Exception:
        pass
    batch = (context.data_sources.add_pandas('pandas').add_dataframe_asset(name)
             .add_batch_definition_whole_dataframe('all').get_batch(batch_parameters={'dataframe': df}))
    suite = gx.ExpectationSuite(name=name)
    for expectation in expectations:
        suite.add_expectation(expectation)
    result = batch.validate(suite)
    failed = []
    for r in result.results:
        if not r.success:
            config = r.expectation_config
            examples = (r.result or {}).get('partial_unexpected_list', [])[:5]
            failed.append(f"{config.type}({config.kwargs.get('column', '')}){f': e.g. {examples}' if examples else ''}")
    return bool(result.success), failed


def _between(column, low=None, high=None, strict_high=False, **kw):
    import great_expectations as gx
    return gx.expectations.ExpectColumnValuesToBeBetween(column=column, min_value=low, max_value=high, strict_max=strict_high, **kw)


def application_expectations(which):
    import great_expectations as gx
    E = gx.expectations
    expectations = [
        E.ExpectTableColumnCountToEqual(value=APPLICATION_COLUMNS[which]),
        E.ExpectColumnValuesToNotBeNull(column='SK_ID_CURR'),
        E.ExpectColumnValuesToBeUnique(column='SK_ID_CURR'),
        # ages between 18 and 70; dates on or before the application
        _between('DAYS_BIRTH', OLDEST_DAYS, YOUNGEST_DAYS),
        _between('DAYS_EMPLOYED', OLDEST_DAYS, 0, row_condition=f'DAYS_EMPLOYED != {DAYS_EMPLOYED_PLACEHOLDER}', condition_parser='pandas'),
        _between('DAYS_REGISTRATION', OLDEST_DAYS, 0),
        _between('DAYS_ID_PUBLISH', OLDEST_DAYS, 0),
        _between('DAYS_LAST_PHONE_CHANGE', OLDEST_DAYS, 0),
        # amounts
        _between('AMT_CREDIT', 0, None, strict_min=True),
        _between('AMT_ANNUITY', 0, None, strict_min=True),
        _between('AMT_GOODS_PRICE', 0, None, strict_min=True),
        _between('AMT_INCOME_TOTAL', 0, None, strict_min=True),
        _between('CNT_CHILDREN', 0, None),
        _between('CNT_FAM_MEMBERS', 1, None),
        # external scores are normalised to [0, 1]
        *[_between(c, 0, 1) for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']],
        # text values the features and the model know; placeholders XNA and Unknown are made missing by preprocessing
        E.ExpectColumnValuesToBeInSet(column='NAME_CONTRACT_TYPE', value_set=['Cash loans', 'Revolving loans']),
        E.ExpectColumnValuesToBeInSet(column='CODE_GENDER', value_set=['F', 'M', 'XNA']),
        E.ExpectColumnValuesToBeInSet(column='FLAG_OWN_CAR', value_set=['N', 'Y']),
        E.ExpectColumnValuesToBeInSet(column='NAME_EDUCATION_TYPE', value_set=['Lower secondary', 'Secondary / secondary special', 'Incomplete higher',
                                                                              'Higher education', 'Academic degree']),
        E.ExpectColumnValuesToBeInSet(column='NAME_FAMILY_STATUS', value_set=['Civil marriage', 'Married', 'Separated', 'Single / not married', 'Unknown', 'Widow']),
        # a rating outside 1 to 3 is an entry error (a handful of test rows); preprocessing makes it missing
        E.ExpectColumnValuesToBeInSet(column='REGION_RATING_CLIENT_W_CITY', value_set=[1, 2, 3], mostly=0.999),
    ]
    if which == 'train':
        expectations += [E.ExpectColumnValuesToNotBeNull(column='TARGET'), E.ExpectColumnValuesToBeInSet(column='TARGET', value_set=[0, 1])]
    return expectations


def validate_application(df, which):
    """The raw application_train (``which='train'``) or application_test table."""
    return _run(df, f'application_{which}', application_expectations(which))


# per related table: its own id, its date columns that must be before the application, and amounts that cannot be negative
RELATED = {
    'bureau': {'id': 'SK_ID_BUREAU', 'before_application': ['DAYS_CREDIT'], 'not_negative': ['AMT_CREDIT_SUM']},
    'previous': {'id': 'SK_ID_PREV', 'before_application': ['DAYS_DECISION'], 'not_negative': ['AMT_APPLICATION', 'AMT_CREDIT']},
    'installments': {'id': None, 'before_application': ['DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT'], 'not_negative': ['AMT_INSTALMENT', 'AMT_PAYMENT']},
    'pos_cash': {'id': None, 'before_application': ['MONTHS_BALANCE'], 'not_negative': ['SK_DPD', 'SK_DPD_DEF']},
    'credit_card': {'id': None, 'before_application': ['MONTHS_BALANCE'], 'not_negative': ['AMT_CREDIT_LIMIT_ACTUAL', 'SK_DPD', 'SK_DPD_DEF']},
}


def table_columns(name):
    rule = RELATED[name]
    return list(dict.fromkeys(KEYS[name] + ([rule['id']] if rule['id'] else []) + rule['before_application'] + rule['not_negative']))


def validate_table(name, df):
    """A raw related table; ``df`` needs at least ``table_columns(name)``."""
    import great_expectations as gx
    E = gx.expectations
    rule = RELATED[name]
    expectations = [E.ExpectColumnValuesToNotBeNull(column=c) for c in KEYS[name]]
    if rule['id']:
        expectations.append(E.ExpectColumnValuesToBeUnique(column=rule['id']))
    # the time rule: on bureau DAYS_CREDIT the day of the application itself (0) is allowed, everywhere else strictly before
    expectations += [_between(c, None, 0, strict_high=(c != 'DAYS_CREDIT')) for c in rule['before_application']]
    expectations += [_between(c, 0, None) for c in rule['not_negative']]
    return _run(df, name, expectations)


# ---------------------------------------------------------------- feature tables

def validate_features(X, features, text_features=()):
    """A joined feature table about to be used by a model. Plain pandas checks; returns ``(passed, failed)``."""
    failed = []
    missing = [c for c in features if c not in X.columns]
    if missing:
        failed.append(f'features missing: {missing}')
    if not X.index.is_unique:
        failed.append('SK_ID_CURR is not unique')
    present = [c for c in features if c in X.columns]
    numeric = [c for c in present if c not in text_features]
    not_numeric = [c for c in numeric if not pd.api.types.is_numeric_dtype(X[c])]
    if not_numeric:
        failed.append(f'expected numeric, found text: {not_numeric}')
    numeric = [c for c in numeric if c not in not_numeric]
    infinite = [c for c in numeric if np.isinf(X[c].to_numpy(dtype=float)).any()]
    if infinite:
        failed.append(f'infinite values in {infinite}')
    empty = [c for c in present if X[c].isna().all()]
    if empty:
        failed.append(f'entirely missing: {empty}')
    return not failed, failed


def validate_raw(raw_dir, log=print):
    """Every level of raw-data checks, one table at a time. Returns ``(passed, failed by check)``."""
    from src.data.load_data import load_application

    results = {}
    ok, failed, summary = validate_raw_files(raw_dir)
    results['files'] = failed
    log(f"files: {'passed' if ok else 'FAILED'} ({len(summary)} of {len(RAW_FILES)} read)")
    if not ok:
        return False, results
    train, test = load_application(raw_dir)
    for which, df in [('train', train), ('test', test)]:
        ok, results[f'application_{which}'] = validate_application(df, which)
        log(f"application_{which}: {'passed' if ok else 'FAILED'}")
    del train, test
    for name in RELATED:
        ok, results[name] = validate_table(name, read_raw(name, raw_dir, table_columns(name)))
        log(f"{name}: {'passed' if ok else 'FAILED'}")
    return all(not f for f in results.values()), results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validate the raw Kaggle files.')
    parser.add_argument('--raw-dir', default='data/raw')
    args = parser.parse_args()
    passed, results = validate_raw(args.raw_dir)
    for check, failed in results.items():
        for f in failed:
            print(f'  {check}: {f}')
    raise SystemExit(0 if passed else 1)
