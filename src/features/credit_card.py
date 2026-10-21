"""Features from the monthly balances of an applicant's earlier Home Credit credit cards, one row per applicant.

This is the code developed and verified in ``notebooks/11_credit_card_eda_features.ipynb``, in a form that can be
used without the notebook. The notebook checks that this module reproduces its saved tables exactly.

- ``prepare_cards``         checks that no month is on or after the application, and sorts the rows
- ``build_card_features``   22 summaries per applicant; ``DROPPED_FEATURES`` lists the 7 removed in screening

Properties the notebook verified: every feature of an applicant is computed from that applicant's own rows; no
function uses the target or a number estimated from the data; no row is dated on or after the application.

Usage::

    from credit_card import card_feature_table
    features = card_feature_table(pd.read_csv('data/raw/credit_card_balance.csv'), applicant_ids)
"""
import numpy as np
import pandas as pd

RECENT_MONTHS, LATEST_WINDOW, PAY_TO_MIN_CAP = 12, 3, 5
MIN_DUE_ABOVE = 1
PREFIX = 'CC_'
FEATURES = ['N_CARDS', 'MONTHS_BACK', 'MONTHS_SINCE_LATEST_ROW', 'UTIL_MEAN', 'UTIL_MAX', 'UTIL_LATEST', 'UTIL_MEAN_12M', 'SHARE_OVER_LIMIT', 'SHARE_WITH_BALANCE', 'BALANCE_LATEST', 'LIMIT_LATEST', 'LIMIT_GROWTH', 'SHARE_ATM', 'ATM_TO_LIMIT_MEAN', 'N_ATM_12M', 'SHARE_UNDER_MIN', 'SHARE_PAID_NOTHING', 'PAY_TO_BALANCE_MEAN', 'PAY_TO_MIN_MEAN', 'SHARE_ARREARS', 'MAX_DPD', 'N_DEMAND']

# removed in stage 8 of the notebook: five near-copies of another candidate, one with no signal, and one that records calendar time
DROPPED_FEATURES = ['CC_UTIL_MEAN', 'CC_UTIL_LATEST', 'CC_UTIL_MEAN_12M', 'CC_ATM_TO_LIMIT_MEAN', 'CC_SHARE_ARREARS', 'CC_N_CARDS', 'CC_MONTHS_BACK']


def prepare_cards(raw):
    """Check the time rule and sort. No row is removed."""
    assert (raw['MONTHS_BALANCE'] < 0).all(), 'a month on or after the application'
    return raw.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'], kind='mergesort').reset_index(drop=True)


def build_card_features(prepared, ids):
    """One row per applicant in `ids`, from prepared monthly card rows. Own rows only, no outcome, no number estimated from the data."""
    p = prepared
    who, month = p['SK_ID_CURR'], p['MONTHS_BALANCE']
    limit = p['AMT_CREDIT_LIMIT_ACTUAL'].where(p['AMT_CREDIT_LIMIT_ACTUAL'] > 0)
    balance = p['AMT_BALANCE']
    util = (balance / limit).clip(lower=0)
    atm = p['AMT_DRAWINGS_ATM_CURRENT'].fillna(0)
    atm_count = p['CNT_DRAWINGS_ATM_CURRENT'].fillna(0)
    paid = p['AMT_PAYMENT_TOTAL_CURRENT']
    min_due = p['AMT_INST_MIN_REGULARITY'].where(p['AMT_INST_MIN_REGULARITY'] > MIN_DUE_ABOVE)
    recent = month >= -RECENT_MONTHS
    mean = lambda values: values.groupby(who).mean()

    last = p.groupby('SK_ID_PREV', sort=False).tail(1)
    last = last[last['MONTHS_BALANCE'] >= -LATEST_WINDOW]
    last_limit = last['AMT_CREDIT_LIMIT_ACTUAL'].where(last['AMT_CREDIT_LIMIT_ACTUAL'] > 0)
    first_limit = p.loc[p['AMT_CREDIT_LIMIT_ACTUAL'] > 0].groupby('SK_ID_PREV', sort=False).agg(applicant=('SK_ID_CURR', 'first'), first=('AMT_CREDIT_LIMIT_ACTUAL', 'first'), last=('AMT_CREDIT_LIMIT_ACTUAL', 'last'))

    f = pd.DataFrame(index=pd.Index(sorted(who.unique()), name='SK_ID_CURR'))
    f['N_CARDS'] = p.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()
    f['MONTHS_BACK'] = -month.groupby(who).min()
    f['MONTHS_SINCE_LATEST_ROW'] = -month.groupby(who).max()
    f['UTIL_MEAN'] = mean(util)
    f['UTIL_MAX'] = util.groupby(who).max()
    f['UTIL_LATEST'] = (last['AMT_BALANCE'] / last_limit).clip(lower=0).groupby(last['SK_ID_CURR']).max()
    f['UTIL_MEAN_12M'] = mean(util.where(recent))
    f['SHARE_OVER_LIMIT'] = mean((util > 1).astype(float).where(util.notna()))
    f['SHARE_WITH_BALANCE'] = mean((balance > 0).astype(float))
    f['BALANCE_LATEST'] = last.groupby('SK_ID_CURR')['AMT_BALANCE'].sum()
    f['LIMIT_LATEST'] = last.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].sum()
    f['LIMIT_GROWTH'] = (first_limit['last'] / first_limit['first']).groupby(first_limit['applicant']).max()
    f['SHARE_ATM'] = mean((atm > 0).astype(float))
    f['ATM_TO_LIMIT_MEAN'] = mean(atm / limit)
    f['N_ATM_12M'] = atm_count.where(recent, 0).groupby(who).sum()
    f['SHARE_UNDER_MIN'] = mean((paid < min_due - 1).astype(float).where(min_due.notna()))
    f['SHARE_PAID_NOTHING'] = mean((paid == 0).astype(float).where(min_due.notna()))
    f['PAY_TO_BALANCE_MEAN'] = mean((paid / balance.where(balance > 0)).clip(upper=1))
    f['PAY_TO_MIN_MEAN'] = mean((paid / min_due).clip(upper=PAY_TO_MIN_CAP))
    f['SHARE_ARREARS'] = mean((p['SK_DPD_DEF'] > 0).astype(float))
    f['MAX_DPD'] = p['SK_DPD_DEF'].groupby(who).max()
    f['N_DEMAND'] = (p['NAME_CONTRACT_STATUS'] == 'Demand').groupby(p['SK_ID_PREV'], sort=False).any().groupby(p.groupby('SK_ID_PREV', sort=False)['SK_ID_CURR'].first()).sum()
    out = f[FEATURES].astype(float).reindex(pd.Index(ids, name='SK_ID_CURR'))
    out['N_CARDS'] = out['N_CARDS'].fillna(0)
    return out


def card_feature_table(raw, ids, screened=True):
    """The saved table: features named CC_, one row per applicant in ``ids``, in that order."""
    table = build_card_features(prepare_cards(raw), ids).add_prefix(PREFIX)
    return table.drop(columns=DROPPED_FEATURES) if screened else table


if __name__ == '__main__':
    import sys
    raw_path, application_path, out_path = sys.argv[1:4]
    ids = pd.read_parquet(application_path, columns=[]).index
    table = card_feature_table(pd.read_csv(raw_path), ids)
    table.to_parquet(out_path)
    print(f'{len(table)} applicants, {table.shape[1]} features')
