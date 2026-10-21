"""Data validation: good data passes, and each kind of bad data is caught."""
import numpy as np
import pandas as pd
import pytest

from conftest import RAW, SCHEMA, make_applications
from src.utils.validate_data import table_columns, validate_application, validate_features, validate_raw_files, validate_table


def test_valid_application_passes():
    passed, failed = validate_application(make_applications(target=True).drop(columns=[]), 'train')
    assert passed, failed


@pytest.mark.parametrize('column, value, expected', [
    ('TARGET', 2, 'TARGET'),
    ('DAYS_BIRTH', 100, 'DAYS_BIRTH'),                  # born after the application
    ('DAYS_REGISTRATION', 30, 'DAYS_REGISTRATION'),     # registered after the application
    ('EXT_SOURCE_2', 1.7, 'EXT_SOURCE_2'),              # scores are in [0, 1]
    ('AMT_CREDIT', 0.0, 'AMT_CREDIT'),
    ('CODE_GENDER', 'Q', 'CODE_GENDER'),
])
def test_bad_application_values_are_caught(column, value, expected):
    rows = make_applications(target=True)
    rows.loc[0, column] = value
    passed, failed = validate_application(rows, 'train')
    assert not passed and any(expected in f for f in failed), failed


def test_duplicate_applicant_is_caught():
    rows = make_applications(target=True)
    rows.loc[1, 'SK_ID_CURR'] = rows.loc[0, 'SK_ID_CURR']
    passed, failed = validate_application(rows, 'train')
    assert not passed and any('unique' in f for f in failed)


def test_not_employed_placeholder_is_allowed():
    passed, failed = validate_application(make_applications(target=True, DAYS_EMPLOYED=365243), 'train')
    assert passed, failed


def test_missing_column_is_caught():
    passed, failed = validate_application(make_applications(target=True).drop(columns='FLAG_DOCUMENT_3'), 'train')
    assert not passed


def history(table, n=10, **overrides):
    rows = pd.DataFrame({c: [1.0] * n for c in table_columns(table)})
    rows['SK_ID_CURR'] = 100_000
    for c in ('SK_ID_BUREAU', 'SK_ID_PREV'):
        if c in rows:
            rows[c] = np.arange(n)
    for c in ('DAYS_CREDIT', 'DAYS_DECISION', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'MONTHS_BALANCE'):
        if c in rows:
            rows[c] = -np.arange(1, n + 1)
    for c, v in overrides.items():
        rows.loc[0, c] = v
    return rows


@pytest.mark.parametrize('table', ['bureau', 'previous', 'installments', 'pos_cash', 'credit_card'])
def test_valid_history_passes(table):
    passed, failed = validate_table(table, history(table))
    assert passed, failed


@pytest.mark.parametrize('table, column', [('bureau', 'DAYS_CREDIT'), ('previous', 'DAYS_DECISION'), ('installments', 'DAYS_ENTRY_PAYMENT'),
                                           ('pos_cash', 'MONTHS_BALANCE'), ('credit_card', 'MONTHS_BALANCE')])
def test_record_after_the_application_is_caught(table, column):
    # the time rule the feature code relies on: a future record would leak into the features
    passed, failed = validate_table(table, history(table, **{column: 5}))
    assert not passed and any(column in f for f in failed), failed


def test_negative_payment_is_caught():
    passed, failed = validate_table('installments', history('installments', AMT_PAYMENT=-5.0))
    assert not passed and any('AMT_PAYMENT' in f for f in failed)


def test_feature_table_checks():
    X = pd.DataFrame({'a': [1.0, np.inf], 'b': ['x', 'y'], 'c': [np.nan, np.nan]}, index=[1, 1])
    passed, failed = validate_features(X, ['a', 'b', 'c', 'd'], text_features=['b'])
    joined = ' '.join(failed)
    assert not passed and "'d'" in joined and 'unique' in joined and 'infinite' in joined and 'entirely missing' in joined


def test_clean_feature_table_passes():
    X = pd.DataFrame({'a': [1.0, 2.0], 'b': ['x', 'y']}, index=[1, 2])
    assert validate_features(X, ['a', 'b'], text_features=['b']) == (True, [])


@pytest.mark.data
def test_raw_files_pass():
    passed, failed, summary = validate_raw_files(RAW)
    assert passed, failed
    assert summary['application_train']['rows'] == 307_511 and summary['application_train']['defaults'] == 24_825
