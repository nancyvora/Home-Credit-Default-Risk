"""Features from the monthly balances of an applicant's earlier point-of-sale and cash loans, one row per applicant.

This is the code developed and verified in ``notebooks/09_pos_cash_eda_features.ipynb``, in a form that can be used
without the notebook. The notebook checks that this module reproduces its saved tables exactly.

- ``prepare_pos``          checks that no month is on or after the application, and sorts the rows
- ``build_pos_features``   21 summaries per applicant; ``DROPPED_FEATURES`` lists the 9 removed in screening

Properties the notebook verified: every feature of an applicant is computed from that applicant's own rows; no
function uses the target or a number estimated from the data; no row is dated on or after the application.

Usage::

    from pos_cash import pos_feature_table
    features = pos_feature_table(pd.read_csv('data/raw/POS_CASH_balance.csv'), applicant_ids)
"""
import numpy as np
import pandas as pd

RECENT_MONTHS, LATEST_WINDOW, EARLY_SHARE = 12, 3, 0.75
CALLED_IN = ['Demand', 'Amortized debt']
PREFIX = 'POS_'
FEATURES = ['N_LOANS', 'N_MONTHS', 'MONTHS_BACK', 'MONTHS_SINCE_LATEST_ROW', 'SHARE_MONTHS_ARREARS', 'MAX_DPD', 'MAX_DPD_RAW', 'N_LOANS_ARREARS', 'MONTHS_ARREARS_12M', 'MONTHS_SINCE_ARREARS', 'N_ARREARS_AT_APPLICATION', 'N_CALLED_IN', 'N_RETURNED', 'SHARE_FINISHED_EARLY', 'SHARE_TERM_CHANGED', 'N_RUNNING', 'INSTALMENTS_LEFT', 'MAX_INSTALMENTS_LEFT', 'MAX_SHARE_LEFT', 'N_PAST_TERM', 'MONTHS_SINCE_LAST_START']

# removed in stage 8 of the notebook: near-copies of another candidate or of a saved feature, and two with no signal
DROPPED_FEATURES = ['POS_SHARE_MONTHS_ARREARS', 'POS_MAX_DPD', 'POS_MAX_INSTALMENTS_LEFT', 'POS_N_LOANS', 'POS_N_MONTHS', 'POS_MONTHS_BACK', 'POS_INSTALMENTS_LEFT', 'POS_N_CALLED_IN', 'POS_MONTHS_SINCE_LAST_START']


def prepare_pos(raw):
    """Check the time rule and sort. No row is removed."""
    assert (raw['MONTHS_BALANCE'] < 0).all(), 'a month on or after the application'
    return raw.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'], kind='mergesort').reset_index(drop=True)


def build_pos_features(prepared, ids):
    """One row per applicant in `ids`, from prepared monthly rows. Uses the applicant's own rows only, no outcome, and no number estimated from the data."""
    p = prepared
    who, month = p['SK_ID_CURR'], p['MONTHS_BALANCE']
    arrears = p['SK_DPD_DEF'] > 0
    status = p['NAME_CONTRACT_STATUS']
    per_loan = p.groupby('SK_ID_PREV', sort=False).agg(
        applicant=('SK_ID_CURR', 'first'), first_month=('MONTHS_BALANCE', 'min'), last_month=('MONTHS_BALANCE', 'max'), months=('MONTHS_BALANCE', 'size'),
        first_term=('CNT_INSTALMENT', 'first'), terms=('CNT_INSTALMENT', 'nunique'), last_status=('NAME_CONTRACT_STATUS', 'last'), last_left=('CNT_INSTALMENT_FUTURE', 'last'),
        last_term=('CNT_INSTALMENT', 'last'), last_dpd=('SK_DPD_DEF', 'last'))
    per_loan['ever_arrears'] = arrears.groupby(p['SK_ID_PREV'], sort=False).any()
    per_loan['called_in'] = status.isin(CALLED_IN).groupby(p['SK_ID_PREV'], sort=False).any()
    per_loan['returned'] = (status == 'Returned to the store').groupby(p['SK_ID_PREV'], sort=False).any()
    per_loan['completed'] = (status == 'Completed').groupby(p['SK_ID_PREV'], sort=False).any()
    per_loan['running'] = (per_loan['last_status'] == 'Active') & (per_loan['last_month'] >= -LATEST_WINDOW)
    per_loan['early'] = per_loan['completed'] & (per_loan['months'] < EARLY_SHARE * per_loan['first_term'])
    per_loan['share_left'] = per_loan['last_left'] / per_loan['last_term'].where(per_loan['last_term'] > 0)
    by_loan = per_loan.groupby('applicant')
    running = per_loan[per_loan['running']].groupby('applicant')
    completed = per_loan[per_loan['completed']].groupby('applicant')

    f = pd.DataFrame(index=pd.Index(sorted(who.unique()), name='SK_ID_CURR'))
    f['N_LOANS'] = by_loan.size()
    f['N_MONTHS'] = who.groupby(who).size()
    f['MONTHS_BACK'] = -month.groupby(who).min()
    f['MONTHS_SINCE_LATEST_ROW'] = -month.groupby(who).max()
    f['SHARE_MONTHS_ARREARS'] = arrears.groupby(who).mean()
    f['MAX_DPD'] = p['SK_DPD_DEF'].groupby(who).max()
    f['MAX_DPD_RAW'] = p['SK_DPD'].groupby(who).max()
    f['N_LOANS_ARREARS'] = by_loan['ever_arrears'].sum()
    f['MONTHS_ARREARS_12M'] = (arrears & (month >= -RECENT_MONTHS)).groupby(who).sum()
    f['MONTHS_SINCE_ARREARS'] = -month.where(arrears).groupby(who).max()
    f['N_ARREARS_AT_APPLICATION'] = (per_loan['running'] & (per_loan['last_dpd'] > 0)).groupby(per_loan['applicant']).sum()
    f['N_CALLED_IN'] = by_loan['called_in'].sum()
    f['N_RETURNED'] = by_loan['returned'].sum()
    f['SHARE_FINISHED_EARLY'] = completed['early'].mean()
    f['SHARE_TERM_CHANGED'] = (per_loan['terms'] > 1).groupby(per_loan['applicant']).mean()
    f['N_RUNNING'] = by_loan['running'].sum()
    f['INSTALMENTS_LEFT'] = running['last_left'].sum(min_count=1)
    f['MAX_INSTALMENTS_LEFT'] = running['last_left'].max()
    f['MAX_SHARE_LEFT'] = running['share_left'].max()
    f['N_PAST_TERM'] = (per_loan['running'] & (per_loan['last_left'] == 0)).groupby(per_loan['applicant']).sum()
    f['MONTHS_SINCE_LAST_START'] = -by_loan['first_month'].max()
    out = f[FEATURES].astype(float).reindex(pd.Index(ids, name='SK_ID_CURR'))
    out['N_LOANS'] = out['N_LOANS'].fillna(0)
    return out


def pos_feature_table(raw, ids, screened=True):
    """The saved table: features named POS_, one row per applicant in ``ids``, in that order."""
    table = build_pos_features(prepare_pos(raw), ids).add_prefix(PREFIX)
    return table.drop(columns=DROPPED_FEATURES) if screened else table


if __name__ == '__main__':
    import sys
    raw_path, application_path, out_path = sys.argv[1:4]
    ids = pd.read_parquet(application_path, columns=[]).index
    table = pos_feature_table(pd.read_csv(raw_path), ids)
    table.to_parquet(out_path)
    print(f'{len(table)} applicants, {table.shape[1]} features')
