"""Build features: turn each preprocessed table into one row of features per applicant, and save the tables.

=============  ===============================================  ==============  ========
table          features                                          module          notebook
=============  ===============================================  ==============  ========
application    41 cleaned raw columns and 14 engineered ones     this file       01
bureau         40 summaries of credit bureau records             bureau.py       03
previous       41 summaries of earlier Home Credit applications  previous_application.py  05
installments   29 summaries of the payment record                installments.py 07
pos_cash       12 summaries of POS and cash loan balances        pos_cash.py     09
credit_card    15 summaries of monthly card balances             credit_card.py  11
=============  ===============================================  ==============  ========

Every feature of an applicant is computed from that applicant's own rows, without the target and without any number
estimated from the data. So the tables are built once for all applicants and used inside cross-validation as they
are, with no leakage between folds. The final model uses 111 of these features (``configs`` / model metadata).

Usage::

    from src.features.build_features import build_features, build_all
    features = build_features('bureau', preprocess('bureau', raw_bureau), applicant_ids)
    build_all('data/raw', 'data/processed')          # every table, train and test, one raw file at a time
"""
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import feature_path, load_application, read_raw
from src.data.preprocess import preprocess
from src.features import bureau, credit_card, installments, pos_cash, previous_application

TABLES = ['application', 'bureau', 'previous', 'installments', 'pos_cash', 'credit_card']

# ---------------------------------------------------------------- application table (notebook 01, stages 11 to 13)

EXT_COLS = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
CITY_FLAGS = ['REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']
EDUCATION_LEVELS = {'Lower secondary': 0, 'Secondary / secondary special': 1, 'Incomplete higher': 2,
                    'Higher education': 3, 'Academic degree': 4}
ID_AND_TARGET = ['SK_ID_CURR', 'TARGET']

# the saved table's columns, in order: 41 cleaned raw columns, then the 14 engineered ones
APPLICATION_FEATURES = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
    'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',
    'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT_W_CITY',
    'HOUR_APPR_PROCESS_START', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
    'ORGANIZATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'OBS_30_CNT_SOCIAL_CIRCLE',
    'DEF_30_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_6',
    'AMT_REQ_CREDIT_BUREAU_YEAR', 'FLOORSMAX_AVG', 'LIVINGAREA_AVG', 'TOTALAREA_MODE',
    'N_BUILDING_MISSING', 'EXT_MEAN', 'EXT_MIN', 'EXT_MAX', 'EXT_STD', 'EXT_N_MISSING', 'ANNUITY_TO_CREDIT',
    'CREDIT_TO_GOODS', 'CREDIT_TO_INCOME', 'ANNUITY_TO_INCOME', 'N_MISSING', 'ENQUIRIES_MISSING', 'N_CITY_MISMATCH',
    'EDUCATION_LEVEL']


def engineer_application_features(clean):
    """The 14 engineered columns kept in notebook 01, from a cleaned application table. Each uses the applicant's own row."""
    raw_columns = [c for c in clean.columns if c not in ID_AND_TARGET]
    building = [c for c in raw_columns if c.endswith(('_AVG', '_MODE', '_MEDI'))]
    ext = clean[EXT_COLS]
    f = pd.DataFrame(index=clean.index)
    # missingness
    f['N_BUILDING_MISSING'] = clean[building].isna().sum(axis=1)
    f['N_MISSING'] = clean[raw_columns].isna().sum(axis=1)
    f['ENQUIRIES_MISSING'] = clean['AMT_REQ_CREDIT_BUREAU_YEAR'].isna().astype(int)
    # external scores
    f['EXT_MEAN'] = ext.mean(axis=1)
    f['EXT_MIN'] = ext.min(axis=1)
    f['EXT_MAX'] = ext.max(axis=1)
    f['EXT_STD'] = ext.std(axis=1)
    f['EXT_N_MISSING'] = ext.isna().sum(axis=1)
    # loan structure
    f['ANNUITY_TO_CREDIT'] = clean['AMT_ANNUITY'] / clean['AMT_CREDIT']
    f['CREDIT_TO_GOODS'] = clean['AMT_CREDIT'] / clean['AMT_GOODS_PRICE']
    f['CREDIT_TO_INCOME'] = clean['AMT_CREDIT'] / clean['AMT_INCOME_TOTAL']
    f['ANNUITY_TO_INCOME'] = clean['AMT_ANNUITY'] / clean['AMT_INCOME_TOTAL']
    # counts and encoding
    f['N_CITY_MISMATCH'] = clean[CITY_FLAGS].sum(axis=1)
    f['EDUCATION_LEVEL'] = clean['NAME_EDUCATION_TYPE'].map(EDUCATION_LEVELS)
    return f.replace([np.inf, -np.inf], np.nan)


def application_features(clean):
    """Cleaned application rows to the saved application feature table, indexed by SK_ID_CURR. TARGET is not included."""
    table = pd.concat([clean, engineer_application_features(clean)], axis=1)[APPLICATION_FEATURES]
    table.index = pd.Index(clean['SK_ID_CURR'].values, name='SK_ID_CURR')
    return table


# ---------------------------------------------------------------- every table

def _screened(build, module, prefixed=True):
    # build the full candidate set from preprocessed rows, name it as saved, and drop the features removed in screening
    def run(clean, ids):
        table = build(clean, ids)
        table = table.add_prefix(module.PREFIX) if prefixed else table
        return table.drop(columns=module.DROPPED_FEATURES)
    return run


BUILDERS = {
    'bureau': _screened(bureau.build_bureau_features, bureau, prefixed=False),                      # adds its own BUREAU_ prefix
    'previous': _screened(previous_application.build_previous_features, previous_application, prefixed=False),   # adds PREV_
    'installments': _screened(installments.build_instalment_features, installments),
    'pos_cash': _screened(pos_cash.build_pos_features, pos_cash),
    'credit_card': _screened(credit_card.build_card_features, credit_card),
}


def build_features(table, clean, ids=None):
    """Feature table of one source table, from its preprocessed rows.

    ``ids`` are the applicants to build for, in that order (every applicant gets a row; one without records gets
    missing values and a count of 0). Not used for the application table, whose rows are the applicants.
    """
    if table == 'application':
        return application_features(clean)
    if table not in BUILDERS:
        raise ValueError(f'no features for {table!r}; known: {TABLES}')
    if ids is None:
        raise ValueError(f'{table}: the applicant ids to build for are needed')
    return BUILDERS[table](clean, pd.Index(ids, name='SK_ID_CURR'))


def build_all(raw_dir, out_dir, tables=TABLES, log=print):
    """Build and save every table for train and test. Each raw file is read once and freed before the next."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    raw = dict(zip(('train', 'test'), load_application(raw_dir)))
    ids = {}
    for which, rows in raw.items():
        table = build_features('application', preprocess('application', rows))
        if which == 'train':
            table['TARGET'] = rows['TARGET'].values
        table.to_parquet(feature_path(out_dir, 'application', which))
        ids[which] = table.index
    del raw
    log(f"application: {len(ids['train']):,} train and {len(ids['test']):,} test applicants ({time.time() - started:.0f} s)")
    for name in tables:
        if name == 'application':
            continue
        started = time.time()
        clean = preprocess(name, read_raw(name, raw_dir))
        for which in ('train', 'test'):
            table = build_features(name, clean, ids[which])
            table.to_parquet(feature_path(out_dir, name, which))
        del clean
        gc.collect()
        log(f"{name}: {table.shape[1]} features ({time.time() - started:.0f} s)")


def compare_with(out_dir, reference_dir, tables=TABLES):
    """For each table and set, whether the built file equals the reference exactly (values, types, order)."""
    return {f'{t}_{w}': bool(pd.read_parquet(feature_path(out_dir, t, w)).equals(pd.read_parquet(feature_path(reference_dir, t, w))))
            for t in tables for w in ('train', 'test')}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build every feature table from the raw files.')
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--out-dir', default='data/processed')
    parser.add_argument('--compare-to', default=None, help='a folder of feature tables to compare the new ones with')
    args = parser.parse_args()
    build_all(args.raw_dir, args.out_dir)
    if args.compare_to:
        same = compare_with(args.out_dir, args.compare_to)
        for name, ok in same.items():
            print(f"{'SAME     ' if ok else 'DIFFERENT'}  {name}")
        raise SystemExit(0 if all(same.values()) else 1)
