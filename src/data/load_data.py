"""Load data: read the raw Kaggle files and the processed feature tables.

The raw files are the ones downloaded from the Home Credit Default Risk competition, unchanged, in ``data/raw``;
``src/utils/validate_data.py`` checks them before use.
``bureau_balance.csv`` is deliberately not used anywhere in the project.

- ``read_raw``           one raw file, read exactly as the notebooks read it
- ``load_application``   application_train and application_test
- ``load_joined``        the processed feature tables of one set (train or test) joined on SK_ID_CURR

Usage::

    from src.data.load_data import read_raw, load_joined
    bureau = read_raw('bureau', 'data/raw')
    X_train = load_joined('data/processed', 'train', ['application', 'bureau', 'previous', 'installments', 'credit_card'])
"""
from pathlib import Path

import pandas as pd

RAW_FILES = {
    'application_train': 'application_train.csv',
    'application_test': 'application_test.csv',
    'bureau': 'bureau.csv',
    'previous': 'previous_application.csv',
    'installments': 'installments_payments.csv',
    'pos_cash': 'POS_CASH_balance.csv',
    'credit_card': 'credit_card_balance.csv',
    'sample_submission': 'sample_submission.csv',
}


def read_raw(name, raw_dir, columns=None):
    """One raw file as read by the notebooks: ``pd.read_csv`` with no options (``columns`` reads only those)."""
    return pd.read_csv(Path(raw_dir) / RAW_FILES[name], usecols=columns)


def load_application(raw_dir):
    """application_train (with TARGET) and application_test, one row per applicant."""
    return read_raw('application_train', raw_dir), read_raw('application_test', raw_dir)


def feature_path(processed_dir, table, which):
    """``data/processed/{table}_{train|test}_features.parquet``."""
    if which not in ('train', 'test'):
        raise ValueError(which)
    return Path(processed_dir) / f'{table}_{which}_features.parquet'


def load_feature_table(processed_dir, table, which):
    return pd.read_parquet(feature_path(processed_dir, table, which))


def load_joined(processed_dir, which, tables):
    """The application feature table joined with the other feature tables on SK_ID_CURR, one row per applicant.

    For ``which='train'`` the result keeps the ``TARGET`` column of the application table.
    """
    joined = load_feature_table(processed_dir, 'application', which)
    # an integer rating in the saved table; the models were fitted with it as a float
    joined['REGION_RATING_CLIENT_W_CITY'] = joined['REGION_RATING_CLIENT_W_CITY'].astype(float)
    for table in tables:
        if table == 'application':
            continue
        part = load_feature_table(processed_dir, table, which)
        if not part.index.equals(joined.index):
            raise ValueError(f'{table} {which} features are not in the order of the application table')
        joined = joined.join(part, how='left')
    return joined


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Read the application tables and report their size.')
    parser.add_argument('--raw-dir', default='data/raw')
    args = parser.parse_args()
    train, test = load_application(args.raw_dir)
    print(f"application_train: {train.shape[0]:,} rows, {train.shape[1]} columns, {int(train['TARGET'].sum()):,} defaults")
    print(f"application_test:  {test.shape[0]:,} rows, {test.shape[1]} columns")
