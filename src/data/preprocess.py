"""Preprocess: clean each raw table, and split train into a development set and a frozen holdout.

Cleaning, one function per table, each developed and verified in its notebook:

=============  ==========================================================  ========
table          what is done                                                notebook
=============  ==========================================================  ========
application    placeholders to missing, isolated extreme values, rating    01
bureau         rules 1-9 (repeats, groupings, currency, placeholders) and  03
               rule 10: nothing recorded after the application is used
previous       rules 1-9 (superseded records, placeholders, ...) and a    05
               guard that nothing is dated after the current application
installments   payments of zero, redundant rows, cards, penalty charges,   07
               then one row per instalment
pos_cash       time guard and sort                                         09
credit_card    time guard and sort                                         11
=============  ==========================================================  ========

The cleaning of the five related tables lives in their feature modules in ``src/features``, where the notebooks
check it against the saved tables; it is called from here, not copied. No cleaning step uses the target or a number
estimated from the data, so a table can be cleaned once and used inside cross-validation as it is.

The split (notebook 02): 20% of train, stratified on the target, is set aside as a holdout; the other 80% is the
development set, cut into five stratified folds. It is computed once and saved; every later run recomputes it and
refuses to continue if the result differs, so that scores from different runs stay comparable.

Usage::

    from src.data.preprocess import preprocess, load_or_create_split
    clean = preprocess('bureau', read_raw('bureau', 'data/raw'))
    split = load_or_create_split('data/processed/application_split.parquet', ids, y)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.features.bureau import prepare_bureau
from src.features.credit_card import prepare_cards
from src.features.installments import prepare_instalments
from src.features.pos_cash import prepare_pos
from src.features.previous_application import prepare_previous

# ---------------------------------------------------------------- application table (notebook 01, stages 4 and 10)

# values above these limits are isolated single rows (notebook 01, section 4.1) and are treated as entry errors
OUTLIER_LIMITS = {
    'AMT_INCOME_TOTAL': 20_000_000,
    'OBS_30_CNT_SOCIAL_CIRCLE': 100,
    'OBS_60_CNT_SOCIAL_CIRCLE': 100,
    'DEF_30_CNT_SOCIAL_CIRCLE': 20,
    'DEF_60_CNT_SOCIAL_CIRCLE': 20,
    'AMT_REQ_CREDIT_BUREAU_QRT': 100,
}
PLACEHOLDERS = {'DAYS_EMPLOYED': 365243, 'CODE_GENDER': 'XNA', 'NAME_FAMILY_STATUS': 'Unknown'}
VALID_RATINGS = [1, 2, 3]


def clean_application(raw):
    """No row is removed; values that stand for nothing, or are entry errors, become missing."""
    df = raw.copy()
    # placeholders that stand for a missing value
    for col, placeholder in PLACEHOLDERS.items():
        df[col] = df[col].replace(placeholder, np.nan)
    # isolated extreme values
    for col, limit in OUTLIER_LIMITS.items():
        df[col] = df[col].where(~(df[col] > limit))
    # a rating outside 1 to 3 is impossible (stage 10)
    rating = df['REGION_RATING_CLIENT_W_CITY']
    df['REGION_RATING_CLIENT_W_CITY'] = rating.where(rating.isin(VALID_RATINGS))
    return df


# ---------------------------------------------------------------- every table

PREPROCESSORS = {
    'application': clean_application,
    'bureau': prepare_bureau,
    'previous': prepare_previous,
    'installments': prepare_instalments,
    'pos_cash': prepare_pos,
    'credit_card': prepare_cards,
}


def preprocess(table, raw):
    """The cleaned rows of one raw table, ready for its feature module."""
    if table not in PREPROCESSORS:
        raise ValueError(f'no preprocessing for {table!r}; known: {sorted(PREPROCESSORS)}')
    return PREPROCESSORS[table](raw)


# ---------------------------------------------------------------- the frozen split (notebook 02)

def make_folds(y, n_splits=5, seed=42):
    """Each row's fold number, stratified on the target."""
    fold = np.empty(len(y), dtype=int)
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        fold[val_idx] = k
    return fold


def compute_split(ids, y, holdout_fraction=0.20, n_splits=5, seed=42):
    """A table indexed by SK_ID_CURR with columns ``split`` ('dev' or 'holdout') and ``fold`` (-1 for the holdout)."""
    ids, y = np.asarray(ids), np.asarray(y)
    _, holdout_ids = train_test_split(ids, test_size=holdout_fraction, stratify=y, random_state=seed)
    split = pd.DataFrame({'split': 'dev', 'fold': -1}, index=pd.Index(ids, name='SK_ID_CURR'))
    split.loc[holdout_ids, 'split'] = 'holdout'
    # folds are assigned inside the development set only
    dev = (split['split'] == 'dev').values
    split.loc[dev, 'fold'] = make_folds(y[dev], n_splits, seed)
    return split


def same_split(a, b):
    return a.index.equals(b.index) and (a['split'].values == b['split'].values).all() and (a['fold'].values == b['fold'].values).all()


def load_or_create_split(path, ids, y, holdout_fraction=0.20, n_splits=5, seed=42):
    """Load the frozen split, after checking that recomputing it gives the same result. Create it if there is none."""
    path = Path(path)
    fresh = compute_split(ids, y, holdout_fraction, n_splits, seed)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh.to_parquet(path)
        return fresh
    saved = pd.read_parquet(path)
    if not same_split(saved, fresh):
        raise RuntimeError(f'The split computed now differs from the frozen split in {path}. '
                           'It is not overwritten: results from earlier runs would no longer be comparable.')
    return saved


if __name__ == '__main__':
    import argparse

    from src.data.load_data import read_raw

    parser = argparse.ArgumentParser(description='Create, or check, the frozen development / holdout split.')
    parser.add_argument('--raw-dir', default='data/raw')
    parser.add_argument('--split', default='data/processed/application_split.parquet')
    args = parser.parse_args()
    train = read_raw('application_train', args.raw_dir, ['SK_ID_CURR', 'TARGET'])
    existed = Path(args.split).exists()
    split = load_or_create_split(args.split, train['SK_ID_CURR'].values, train['TARGET'].values)
    counts = split['split'].value_counts()
    print(f"{'checked' if existed else 'created'} {args.split}: {counts['dev']:,} development applicants in 5 folds, {counts['holdout']:,} holdout")
