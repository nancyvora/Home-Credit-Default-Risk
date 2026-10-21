"""Features: the engineered values, their independence from the target and from other applicants, and the saved tables."""
import numpy as np
import pandas as pd
import pytest

from conftest import PROCESSED, RAW, SCHEMA, make_applications
from src.data.preprocess import preprocess
from src.features.build_features import APPLICATION_FEATURES, build_features, engineer_application_features


def application_table(rows):
    return build_features('application', preprocess('application', rows))


def test_columns_and_index_are_those_of_the_saved_table():
    table = application_table(make_applications())
    assert list(table.columns) == APPLICATION_FEATURES and len(APPLICATION_FEATURES) == 55
    assert table.index.name == 'SK_ID_CURR' and table.index.is_unique


def test_engineered_values():
    rows = make_applications(n=3)
    rows.loc[0, ['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE']] = [20_000.0, 400_000.0, 320_000.0]
    rows.loc[1, ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']] = [np.nan, 0.2, 0.6]
    rows.loc[2, 'NAME_EDUCATION_TYPE'] = 'Academic degree'
    f = engineer_application_features(preprocess('application', rows))
    assert f.loc[0, 'ANNUITY_TO_CREDIT'] == pytest.approx(0.05) and f.loc[0, 'CREDIT_TO_GOODS'] == pytest.approx(1.25)
    assert f.loc[1, 'EXT_MEAN'] == pytest.approx(0.4) and f.loc[1, 'EXT_MIN'] == pytest.approx(0.2) and f.loc[1, 'EXT_N_MISSING'] == 1
    assert f.loc[2, 'EDUCATION_LEVEL'] == 4


def test_division_by_zero_gives_missing_not_infinite():
    rows = make_applications(n=2)
    rows.loc[0, 'AMT_GOODS_PRICE'] = 0.0
    f = engineer_application_features(preprocess('application', rows))
    assert pd.isna(f.loc[0, 'CREDIT_TO_GOODS']) and not np.isinf(f.select_dtypes('number').to_numpy(dtype=float)).any()


def test_missing_values_are_counted():
    rows = make_applications(n=2)
    rows.loc[1, ['OWN_CAR_AGE', 'OCCUPATION_TYPE', 'AMT_REQ_CREDIT_BUREAU_YEAR']] = np.nan
    f = engineer_application_features(preprocess('application', rows))
    assert f.loc[1, 'N_MISSING'] - f.loc[0, 'N_MISSING'] == (3 - int(pd.isna(rows.loc[0, ['OWN_CAR_AGE', 'OCCUPATION_TYPE', 'AMT_REQ_CREDIT_BUREAU_YEAR']]).sum()))
    assert f.loc[1, 'ENQUIRIES_MISSING'] == 1


def test_features_do_not_use_the_target():
    rows = make_applications(target=True)
    flipped = rows.assign(TARGET=1 - rows['TARGET'])
    assert application_table(rows).equals(application_table(flipped))


def test_an_applicants_features_use_only_their_own_row():
    rows = make_applications(n=30)
    changed = rows.copy()
    changed.loc[1:, ['AMT_CREDIT', 'EXT_SOURCE_1', 'DAYS_BIRTH']] = [9e6, 0.01, -9000]
    assert application_table(rows).iloc[0].equals(application_table(changed).iloc[0])


@pytest.mark.parametrize('table, count', [('bureau', 'BUREAU_N_CREDITS'), ('previous', 'PREV_N_APPLICATIONS')])
def test_applicant_without_records_gets_a_count_of_zero(table, count):
    empty = pd.DataFrame(columns=list(SCHEMA[table]))
    features = build_features(table, preprocess(table, empty), [123])
    assert features.loc[123, count] == 0 and features.drop(columns=count).isna().all(axis=None)


@pytest.mark.parametrize('table', ['installments', 'pos_cash', 'credit_card'])
def test_applicant_without_records_gets_missing_values(table):
    empty = pd.DataFrame(columns=list(SCHEMA[table]))
    features = build_features(table, preprocess(table, empty), [123])
    assert len(features) == 1 and features.index[0] == 123


def test_history_table_needs_the_applicants():
    with pytest.raises(ValueError):
        build_features('bureau', pd.DataFrame())


@pytest.mark.data
def test_application_features_match_the_saved_table(raw_sample):
    saved = pd.read_parquet(PROCESSED / 'application_train_features.parquet').drop(columns='TARGET')
    built = application_table(raw_sample)
    assert built.equals(saved.loc[built.index])


@pytest.mark.data
@pytest.mark.slow
def test_bureau_features_of_a_few_applicants_match_the_full_build():
    # built from only these applicants' rows, they equal the saved table built from everyone's: no applicant's features depend on another's
    saved = pd.read_parquet(PROCESSED / 'bureau_train_features.parquet')
    ids = saved.index[::1500][:200]
    raw = pd.read_csv(RAW / 'bureau.csv')
    built = build_features('bureau', preprocess('bureau', raw[raw['SK_ID_CURR'].isin(ids)]), ids)
    assert built.equals(saved.loc[ids])
