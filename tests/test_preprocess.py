"""Preprocessing: the cleaning rules, the time guards, and the frozen split."""
import numpy as np
import pandas as pd
import pytest

from conftest import PROCESSED, make_applications
from src.data.preprocess import clean_application, compute_split, load_or_create_split, preprocess


def test_placeholders_become_missing():
    rows = make_applications(n=3)
    rows.loc[0, 'DAYS_EMPLOYED'] = 365243
    rows.loc[1, 'CODE_GENDER'] = 'XNA'
    rows.loc[2, 'NAME_FAMILY_STATUS'] = 'Unknown'
    clean = clean_application(rows)
    assert pd.isna(clean.loc[0, 'DAYS_EMPLOYED']) and pd.isna(clean.loc[1, 'CODE_GENDER']) and pd.isna(clean.loc[2, 'NAME_FAMILY_STATUS'])


def test_isolated_extreme_values_and_impossible_rating_become_missing():
    rows = make_applications(n=3)
    rows.loc[0, 'AMT_INCOME_TOTAL'] = 117_000_000
    rows.loc[1, 'DEF_30_CNT_SOCIAL_CIRCLE'] = 34
    rows.loc[2, 'REGION_RATING_CLIENT_W_CITY'] = -1
    clean = clean_application(rows)
    assert pd.isna(clean.loc[0, 'AMT_INCOME_TOTAL']) and pd.isna(clean.loc[1, 'DEF_30_CNT_SOCIAL_CIRCLE']) and pd.isna(clean.loc[2, 'REGION_RATING_CLIENT_W_CITY'])


def test_cleaning_keeps_every_row_and_ordinary_values():
    rows = make_applications(n=50)
    clean = clean_application(rows)
    assert len(clean) == 50 and clean['SK_ID_CURR'].equals(rows['SK_ID_CURR']) and clean['AMT_CREDIT'].equals(rows['AMT_CREDIT'])
    assert clean_application(rows).equals(clean)          # deterministic, and the input is not modified
    assert rows['DAYS_EMPLOYED'].equals(make_applications(n=50)['DAYS_EMPLOYED'])


@pytest.mark.parametrize('table, rows', [
    ('pos_cash', pd.DataFrame({'SK_ID_PREV': [1], 'SK_ID_CURR': [1], 'MONTHS_BALANCE': [0]})),
    ('credit_card', pd.DataFrame({'SK_ID_PREV': [1], 'SK_ID_CURR': [1], 'MONTHS_BALANCE': [2]})),
])
def test_records_on_or_after_the_application_are_refused(table, rows):
    with pytest.raises(AssertionError):
        preprocess(table, rows)


def test_unknown_table_is_refused():
    with pytest.raises(ValueError):
        preprocess('bureau_balance', pd.DataFrame())


def test_split_is_stratified_and_folds_are_inside_development():
    rng = np.random.default_rng(1)
    ids, y = np.arange(10_000), (rng.random(10_000) < 0.08).astype(int)
    split = compute_split(ids, y)
    dev, hold = split['split'] == 'dev', split['split'] == 'holdout'
    assert hold.mean() == pytest.approx(0.20, abs=1e-3)
    assert y[hold.values].mean() == pytest.approx(y.mean(), abs=0.003)
    assert set(split.loc[dev, 'fold']) == {0, 1, 2, 3, 4} and set(split.loc[hold, 'fold']) == {-1}
    rates = pd.Series(y[dev.values]).groupby(split.loc[dev, 'fold'].values).mean()
    assert rates.max() - rates.min() < 0.005
    assert compute_split(ids, y).equals(split)            # the same every time


def test_frozen_split_is_never_overwritten(tmp_path):
    rng = np.random.default_rng(2)
    ids, y = np.arange(2_000), (rng.random(2_000) < 0.1).astype(int)
    path = tmp_path / 'split.parquet'
    created = load_or_create_split(path, ids, y)
    assert load_or_create_split(path, ids, y).equals(created)
    with pytest.raises(RuntimeError):
        load_or_create_split(path, ids, y, seed=7)        # a different split is refused, the file is kept
    assert pd.read_parquet(path).equals(created)


@pytest.mark.data
def test_saved_split_matches_a_fresh_one():
    application = pd.read_parquet(PROCESSED / 'application_train_features.parquet', columns=['TARGET'])
    saved = pd.read_parquet(PROCESSED / 'application_split.parquet')
    fresh = compute_split(application.index.values, application['TARGET'].values)
    assert saved.index.equals(fresh.index) and (saved['split'].values == fresh['split'].values).all() and (saved['fold'].values == fresh['fold'].values).all()
    assert (saved['split'] == 'holdout').sum() == 61_503
