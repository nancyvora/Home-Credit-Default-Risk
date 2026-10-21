"""Features from the payments an applicant made on earlier Home Credit loans, one row per applicant.

This is the code developed and verified in ``notebooks/07_installments_eda_features.ipynb``, in a form that can be
used without the notebook. The notebook checks that this module reproduces its saved tables exactly.

Four steps:

- ``clean_installments``          rules 1 to 4: payments of zero, redundant rows, cards, penalty charges
- ``to_instalments``              rule 5: one row per instalment, with what was due and what was paid counted once
- ``prepare_instalments``         both, with a guard that nothing is dated on or after the current application
- ``build_instalment_features``   33 summaries per applicant; ``DROPPED_FEATURES`` lists the 4 removed in screening

Properties the notebook verified, which callers may rely on:

- every feature of an applicant is computed from that applicant's own rows, so the table can be built once and
  used inside cross-validation without being rebuilt in each fold;
- no function uses the target, and none contains a number estimated from the data;
- no due date and no payment date is on or after the current application.

Usage::

    from installments import instalment_feature_table
    features = instalment_feature_table(pd.read_csv('data/raw/installments_payments.csv'), applicant_ids)
"""
import numpy as np
import pandas as pd

CARD_VERSION, EXTRA_ABOVE = 0, 100
UNPAID_MIN_AMOUNT = 200
LATE_DAYS, LONG_LATE_DAYS, TIMING_CAP = 5, 30, 60
RECENT_DAYS, HALF_YEAR_DAYS, PENALTY_RECENT_DAYS = 365, 183, 730
UNPAID_RECENT_DAYS, CARD_ACTIVE_DAYS, PAYMENTS_WINDOW_DAYS, LARGE_PAYMENT_FACTOR = 30, 90, 30, 3
PREFIX = 'INST_'

FEATURES = ['N_LOANS', 'N_INSTALMENTS', 'DAYS_SINCE_FIRST_DUE', 'DAYS_SINCE_LAST_PAYMENT', 'N_PAYMENTS_30D',
            'SHARE_LATE', 'SHARE_LATE_5D', 'MEAN_DAYS_LATE', 'MAX_DAYS_LATE', 'N_LATE_30D', 'MEAN_DAYS_EARLY',
            'SHARE_LATE_5D_12M', 'MEAN_TIMING_12M', 'TIMING_CHANGE', 'N_LATE_5D_6M', 'DAYS_SINCE_LAST_LATE_5D', 'LAST_TIMING',
            'SHARE_PAID_IN_PARTS', 'SHARE_PAID_IN_PARTS_12M', 'N_SHORT', 'N_UNPAID_RECENT', 'N_UNPAID_OLDER',
            'N_PENALTIES', 'N_PENALTIES_24M', 'DAYS_SINCE_LAST_PENALTY',
            'CARD_N_INSTALMENTS', 'CARD_SHARE_LATE_5D', 'CARD_ACTIVE_90D', 'CARD_MEAN_PAYMENT_12M',
            'PAID_12M', 'MEDIAN_AMT_DUE', 'SHARE_LOANS_LARGE_PAYMENT', 'SHARE_SPLIT_DUE']

# removed in stage 8 of the notebook: two near-copies of another candidate, and two that repeat a feature already built from previous_application
DROPPED_FEATURES = ['INST_SHARE_LATE', 'INST_MEAN_DAYS_LATE', 'INST_N_LOANS', 'INST_DAYS_SINCE_FIRST_DUE']


def clean_installments(raw):
    """Apply the cleaning rules to installments_payments rows. No outcome is used and nothing is estimated from the data."""
    df = raw.copy()
    # 1. a payment of zero is no payment
    zero = df['AMT_PAYMENT'] == 0
    df.loc[zero, ['AMT_PAYMENT', 'DAYS_ENTRY_PAYMENT']] = np.nan
    # 2. drop a row with no payment when the same instalment, in the same calendar version, has a payment on another row
    paid_rows = df['AMT_PAYMENT'].notna().groupby([df['SK_ID_PREV'], df['NUM_INSTALMENT_NUMBER'], df['NUM_INSTALMENT_VERSION']]).transform('sum')
    df = df[df['AMT_PAYMENT'].notna() | (paid_rows == 0)].copy()
    # 3. cards
    df['IS_CARD'] = (df['NUM_INSTALMENT_VERSION'] == CARD_VERSION).groupby(df['SK_ID_PREV']).transform('any')
    # 4. extra charges
    df['IS_EXTRA'] = ~df['IS_CARD'] & (df['NUM_INSTALMENT_NUMBER'] > EXTRA_ABOVE)
    return df.reset_index(drop=True)


def to_instalments(clean):
    """One row per instalment (loan and instalment number), from cleaned rows."""
    keys = ['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']
    by_version = clean.groupby(keys + ['NUM_INSTALMENT_VERSION'], sort=False).agg(
        SK_ID_CURR=('SK_ID_CURR', 'first'), IS_CARD=('IS_CARD', 'first'), IS_EXTRA=('IS_EXTRA', 'first'), DUE_DATE=('DAYS_INSTALMENT', 'max'), AMT_DUE=('AMT_INSTALMENT', 'max'),
        AMT_PAID=('AMT_PAYMENT', 'sum'), N_PAYMENTS=('AMT_PAYMENT', 'count'), FIRST_PAYMENT=('DAYS_ENTRY_PAYMENT', 'min'), LAST_PAYMENT=('DAYS_ENTRY_PAYMENT', 'max'))
    out = by_version.groupby(level=keys, sort=False).agg(
        SK_ID_CURR=('SK_ID_CURR', 'first'), IS_CARD=('IS_CARD', 'first'), IS_EXTRA=('IS_EXTRA', 'first'), DUE_DATE=('DUE_DATE', 'max'), N_VERSIONS=('AMT_DUE', 'size'), AMT_DUE=('AMT_DUE', 'sum'),
        AMT_PAID=('AMT_PAID', 'max'), N_PAYMENTS=('N_PAYMENTS', 'max'), FIRST_PAYMENT=('FIRST_PAYMENT', 'min'), LAST_PAYMENT=('LAST_PAYMENT', 'max'))
    out['UNPAID'] = out['N_PAYMENTS'] == 0
    out.loc[out['UNPAID'], 'AMT_PAID'] = 0.0
    return out


def prepare_instalments(raw):
    """Clean the rows, collapse them to instalments, enforce the time rule, and add the derived columns."""
    assert (raw['DAYS_INSTALMENT'] < 0).all() and (raw['DAYS_ENTRY_PAYMENT'].dropna() < 0).all(), 'a date on or after the application'
    out = to_instalments(clean_installments(raw))
    out['DAYS_LATE'] = out['LAST_PAYMENT'] - out['DUE_DATE']          # completion date minus due date; positive is late; missing when nothing was paid
    out['DAYS_OBSERVED'] = -out['DUE_DATE']                           # how long before the application it fell due: no delay longer than this can be on record
    out['SHORTFALL'] = (out['AMT_DUE'] - out['AMT_PAID']).clip(lower=0)
    return out


def build_instalment_features(prepared, ids):
    """One row per applicant in `ids`, from prepared instalments. Uses the applicant's own instalments only, no outcome, and no number estimated from the data."""
    p = prepared.reset_index()
    who = p['SK_ID_CURR']
    late, seen = p['DAYS_LATE'], p['DAYS_OBSERVED']
    is_sched = ~p['IS_CARD'] & ~p['IS_EXTRA']
    recent = seen <= RECENT_DAYS
    total = lambda flag: flag.groupby(who).sum()
    mean_where = lambda values, flag: values.where(flag).groupby(who).mean()
    f = pd.DataFrame(index=pd.Index(sorted(who.unique()), name='SK_ID_CURR'))

    # volume
    f['N_LOANS'] = p.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()
    f['N_INSTALMENTS'] = total(is_sched)
    f['DAYS_SINCE_FIRST_DUE'] = seen.groupby(who).max()
    f['DAYS_SINCE_LAST_PAYMENT'] = -p['LAST_PAYMENT'].groupby(who).max()
    f['N_PAYMENTS_30D'] = total(p['LAST_PAYMENT'] > -(PAYMENTS_WINDOW_DAYS + 1))

    # lateness, on loans with a schedule. An instalment with no payment has no delay and is counted under 'unpaid'.
    n_sched = f['N_INSTALMENTS'].where(f['N_INSTALMENTS'] > 0)
    f['SHARE_LATE'] = total(is_sched & (late > 0)) / n_sched
    f['SHARE_LATE_5D'] = total(is_sched & (late > LATE_DAYS)) / n_sched
    f['MEAN_DAYS_LATE'] = mean_where(late.clip(lower=0, upper=TIMING_CAP), is_sched)
    f['MAX_DAYS_LATE'] = late.where(is_sched).groupby(who).max().clip(lower=0)
    old_enough = is_sched & (seen > LONG_LATE_DAYS)
    f['N_LATE_30D'] = total(old_enough & (late > LONG_LATE_DAYS)).where(total(old_enough) > 0)
    f['MEAN_DAYS_EARLY'] = mean_where((-late).clip(lower=0, upper=TIMING_CAP), is_sched)

    # recent behaviour
    n_recent = total(is_sched & recent)
    f['SHARE_LATE_5D_12M'] = total(is_sched & recent & (late > LATE_DAYS)) / n_recent.where(n_recent > 0)
    capped = late.clip(lower=-TIMING_CAP, upper=TIMING_CAP)
    f['MEAN_TIMING_12M'] = mean_where(capped, is_sched & recent)
    f['TIMING_CHANGE'] = f['MEAN_TIMING_12M'] - mean_where(capped, is_sched & ~recent)
    f['N_LATE_5D_6M'] = total(is_sched & (seen <= HALF_YEAR_DAYS) & (late > LATE_DAYS)).where(f['N_INSTALMENTS'] > 0)
    f['DAYS_SINCE_LAST_LATE_5D'] = seen.where(is_sched & (late > LATE_DAYS)).groupby(who).min()
    last = p[is_sched].sort_values(['SK_ID_CURR', 'DUE_DATE', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']).groupby('SK_ID_CURR').tail(1).set_index('SK_ID_CURR')
    f['LAST_TIMING'] = last['DAYS_LATE']

    # parts and shortfall
    parts = is_sched & (p['N_PAYMENTS'] > 1)
    f['SHARE_PAID_IN_PARTS'] = total(parts) / n_sched
    f['SHARE_PAID_IN_PARTS_12M'] = total(parts & recent) / n_recent.where(n_recent > 0)
    f['N_SHORT'] = total(is_sched & ~p['UNPAID'] & (p['SHORTFALL'] > 0.01)).where(f['N_INSTALMENTS'] > 0)

    # unpaid: schedule loans and cards, amounts of UNPAID_MIN_AMOUNT or more
    unpaid = ~p['IS_EXTRA'] & p['UNPAID'] & (p['AMT_DUE'] >= UNPAID_MIN_AMOUNT)
    f['N_UNPAID_RECENT'] = total(unpaid & (seen <= UNPAID_RECENT_DAYS))
    f['N_UNPAID_OLDER'] = total(unpaid & (seen > UNPAID_RECENT_DAYS))

    # penalties
    f['N_PENALTIES'] = total(p['IS_EXTRA'])
    f['N_PENALTIES_24M'] = total(p['IS_EXTRA'] & (seen <= PENALTY_RECENT_DAYS))
    f['DAYS_SINCE_LAST_PENALTY'] = seen.where(p['IS_EXTRA']).groupby(who).min()

    # cards
    f['CARD_N_INSTALMENTS'] = total(p['IS_CARD'])
    n_card = f['CARD_N_INSTALMENTS'].where(f['CARD_N_INSTALMENTS'] > 0)
    f['CARD_SHARE_LATE_5D'] = total(p['IS_CARD'] & (late > LATE_DAYS)) / n_card
    f['CARD_ACTIVE_90D'] = (total(p['IS_CARD'] & (seen <= CARD_ACTIVE_DAYS)) > 0).astype(float)
    f['CARD_MEAN_PAYMENT_12M'] = mean_where(p['AMT_PAID'], p['IS_CARD'] & recent)

    # amounts
    f['PAID_12M'] = p['AMT_PAID'].where(is_sched & recent).groupby(who).sum(min_count=1)
    f['MEDIAN_AMT_DUE'] = p['AMT_DUE'].where(is_sched).groupby(who).median()
    s = p[is_sched]
    usual = s.groupby('SK_ID_PREV')['AMT_DUE'].transform('median')
    loan_large = (s['AMT_DUE'] > LARGE_PAYMENT_FACTOR * usual).groupby([s['SK_ID_CURR'], s['SK_ID_PREV']]).any()
    f['SHARE_LOANS_LARGE_PAYMENT'] = loan_large.groupby(level=0).mean()
    f['SHARE_SPLIT_DUE'] = total(is_sched & (p['N_VERSIONS'] > 1)) / n_sched

    out = f[FEATURES].astype(float).reindex(pd.Index(ids, name='SK_ID_CURR'))
    out['N_LOANS'] = out['N_LOANS'].fillna(0)
    return out


def instalment_feature_table(raw, ids, screened=True):
    """The saved table: features named INST_, one row per applicant in ``ids``, in that order."""
    table = build_instalment_features(prepare_instalments(raw), ids).add_prefix(PREFIX)
    return table.drop(columns=DROPPED_FEATURES) if screened else table


if __name__ == '__main__':
    import sys
    raw_path, application_path, out_path = sys.argv[1:4]
    ids = pd.read_parquet(application_path, columns=[]).index
    table = instalment_feature_table(pd.read_csv(raw_path), ids)
    table.to_parquet(out_path)
    print(f'{len(table)} applicants, {table.shape[1]} features')
