"""Features from the credit bureau table, one row per applicant.

This is the code developed and verified in ``notebooks/03_bureau_eda_features.ipynb``, in a form that can be used
without the notebook. The notebook checks that this module reproduces its saved tables exactly.

Three steps:

- ``clean_bureau``            rules 1 to 9: repeated rows, grouped status and type, currency, placeholders, impossible values
- ``prepare_bureau``          rule 10 on top: nothing recorded after the application date is used
- ``build_bureau_features``   46 summaries per applicant; ``SCREENED_FEATURES`` lists the 40 that passed screening

Properties the notebook verified, which callers may rely on:

- every feature of an applicant is computed from that applicant's own rows, so the table can be built once and
  used inside cross-validation without being rebuilt in each fold;
- no function uses the target, and none contains a number estimated from the data;
- ``bureau_balance.csv``, ``AMT_ANNUITY``, ``CREDIT_CURRENCY`` and ``CNT_CREDIT_PROLONG`` are deliberately not used.

Usage::

    from bureau import bureau_feature_table
    features = bureau_feature_table(pd.read_csv('data/raw/bureau.csv'), applicant_ids)
"""
import numpy as np
import pandas as pd

MAIN_TYPES = ['Consumer credit', 'Credit card', 'Car loan', 'Mortgage', 'Microloan']
AMOUNT_COLUMNS = ['AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_MAX_OVERDUE']
FAR_FUTURE_DAYS = 365.25 * 30
STATE_AT_LAST_UPDATE = ['DAYS_CREDIT_UPDATE', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_CREDIT_MAX_OVERDUE',
                        'CREDIT_DAY_OVERDUE', 'CNT_CREDIT_PROLONG']

WINDOW_EDGE_DAYS = 7 * 365.25           # an oldest credit beyond this is treated as reaching the edge of the eight-year history
LIVE_WITHIN_DAYS = 365                  # an active credit not updated for longer than this is treated as finished
OVERDUE_FLOOR = 1000                    # a largest-ever overdue amount at or below this is not counted in the "_1000" features
RECENT_WINDOWS = {'3M': 91, '6M': 183, '12M': 365, '24M': 730}

# removed in stage 8 of the notebook: four near-copies of another feature, and two with no signal
DROPPED_FEATURES = ['BUREAU_DAYS_SINCE_LAST_LIVE_OPENED', 'BUREAU_N_EVER_OVERDUE_1000', 'BUREAU_SHARE_EVER_OVERDUE_1000', 'BUREAU_LIVE_DEBT_SUM',
                    'BUREAU_N_CAR_LOANS', 'BUREAU_N_TYPES']


def clean_bureau(raw, report=None):
    # Applies the nine rules of section 3.8. Uses only each row's own values, plus the other rows of the same applicant for rule 1.
    df = raw.copy()
    note = (lambda rule, n: report.__setitem__(rule, int(n))) if report is not None else (lambda rule, n: None)

    # 1. repeated reports of the same credit: keep the one with the lowest SK_ID_BUREAU
    df = df.sort_values('SK_ID_BUREAU')
    repeated = df.duplicated(subset=[c for c in df.columns if c != 'SK_ID_BUREAU'], keep='first')
    note('1. repeated rows removed', repeated.sum())
    df = df[~repeated].copy()

    # 2. status
    df['STATUS'] = df['CREDIT_ACTIVE'].where(df['CREDIT_ACTIVE'].isin(['Active', 'Closed']), 'Sold or bad debt')
    note('2. rows with status grouped as Sold or bad debt', (df['STATUS'] == 'Sold or bad debt').sum())

    # 3. type
    df['TYPE'] = df['CREDIT_TYPE'].where(df['CREDIT_TYPE'].isin(MAIN_TYPES), 'Other')
    note('3. rows with type grouped as Other', (df['TYPE'] == 'Other').sum())

    # 4. amounts in other currencies cannot be compared
    foreign = df['CREDIT_CURRENCY'] != 'currency 1'
    df.loc[foreign, AMOUNT_COLUMNS] = np.nan
    note('4. rows in another currency, amounts set to missing', foreign.sum())

    # 5. supplied on different terms in train and test
    note('5. AMT_ANNUITY values dropped', df['AMT_ANNUITY'].notna().sum())
    df = df.drop(columns=['AMT_ANNUITY'])

    # 6. placeholder for a credit with no fixed end
    far = df['DAYS_CREDIT_ENDDATE'] > FAR_FUTURE_DAYS
    df.loc[far, 'DAYS_CREDIT_ENDDATE'] = np.nan
    note('6. planned end more than 30 years ahead, set to missing', far.sum())

    # 7. dates before the credit was opened
    for c in ['DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'DAYS_CREDIT_UPDATE']:
        impossible = df[c] < df['DAYS_CREDIT']
        df[c] = df[c].astype(float).where(~impossible)
        note(f'7. {c} before the credit was opened, set to missing', impossible.sum())

    # 8. overpaid credit cards owe nothing
    negative = df['AMT_CREDIT_SUM_DEBT'] < 0
    df.loc[negative, 'AMT_CREDIT_SUM_DEBT'] = 0.0
    note('8. negative debt set to zero', negative.sum())

    # 9. only a positive credit limit is information
    no_limit = df['AMT_CREDIT_SUM_LIMIT'] <= 0
    df.loc[no_limit, 'AMT_CREDIT_SUM_LIMIT'] = np.nan
    note('9. credit limit of zero or below, set to missing', no_limit.sum())

    return df.drop(columns=['CREDIT_ACTIVE', 'CREDIT_TYPE', 'CREDIT_CURRENCY']).reset_index(drop=True)


def prepare_bureau(raw, report=None):
    # rules 1 to 9 (section 3.8), then rule 10: nothing recorded after the application is used
    df = clean_bureau(raw, report)
    after_application = df['DAYS_CREDIT_UPDATE'] > 0
    assert (df.loc[after_application, 'STATUS'] == 'Active').all(), 'rule 10 keeps the status only because these credits are all active'
    for c in STATE_AT_LAST_UPDATE:
        df[c] = df[c].astype(float).where(~after_application)
    if report is not None:
        report['10. rows updated after the application, state columns set to missing'] = int(after_application.sum())
    return df


def build_bureau_features(ready, applicant_ids):
    # ready: the output of prepare_bureau. Returns one row per applicant in applicant_ids, in that order.
    # Every feature is computed from the applicant's own credits only.
    r = ready
    since_open, since_update = -r['DAYS_CREDIT'], -r['DAYS_CREDIT_UPDATE']
    active, closed, sold = r['STATUS'] == 'Active', r['STATUS'] == 'Closed', r['STATUS'] == 'Sold or bad debt'
    live = active & ~(since_update > LIVE_WITHIN_DAYS)          # a missing update date counts as live
    finished = ~live & ~sold
    card = r['TYPE'] == 'Credit card'
    debt, amount, limit = r['AMT_CREDIT_SUM_DEBT'], r['AMT_CREDIT_SUM'], r['AMT_CREDIT_SUM_LIMIT']
    overdue_now = (r['CREDIT_DAY_OVERDUE'] > 0) | (r['AMT_CREDIT_SUM_OVERDUE'] > 0)
    ever_overdue = r['AMT_CREDIT_MAX_OVERDUE'] > 0
    ever_overdue_big = r['AMT_CREDIT_MAX_OVERDUE'] > OVERDUE_FLOOR
    live_loan = live & ~card & (amount > 0) & debt.notna()      # loans where a share owed can be computed
    live_card_limit = live & card & limit.notna() & debt.notna()
    closed_dated = closed & r['DAYS_CREDIT_ENDDATE'].notna() & r['DAYS_ENDDATE_FACT'].notna()
    early_by = r['DAYS_CREDIT_ENDDATE'] - r['DAYS_ENDDATE_FACT']

    key = r['SK_ID_CURR']
    count = lambda flag: flag.astype(float).groupby(key).sum()
    total = lambda values, flag: values.where(flag).groupby(key).sum(min_count=1)       # missing when the applicant has no such credit
    smallest = lambda values, flag: values.where(flag).groupby(key).min()
    largest = lambda values, flag: values.where(flag).groupby(key).max()
    everything = pd.Series(True, index=r.index)

    f = pd.DataFrame(index=pd.Index(np.sort(key.unique()), name='SK_ID_CURR'))
    # volume
    f['N_CREDITS'] = count(everything)
    f['N_LIVE'], f['N_FINISHED'], f['N_SOLD_OR_BAD'] = count(live), count(finished), count(sold)
    f['SHARE_LIVE'] = f['N_LIVE'] / f['N_CREDITS']
    # recency
    f['DAYS_SINCE_LAST_OPENED'] = smallest(since_open, everything)
    f['DAYS_SINCE_FIRST_OPENED'] = largest(since_open, everything)
    f['HISTORY_REACHES_WINDOW_EDGE'] = (f['DAYS_SINCE_FIRST_OPENED'] >= WINDOW_EDGE_DAYS).astype(float)
    for label, days in RECENT_WINDOWS.items():
        f[f'N_OPENED_{label}'] = count(since_open <= days)
    f['MEAN_DAYS_BETWEEN_OPENINGS'] = ((f['DAYS_SINCE_FIRST_OPENED'] - f['DAYS_SINCE_LAST_OPENED']) / (f['N_CREDITS'] - 1)).where(f['N_CREDITS'] > 1)
    f['DAYS_SINCE_LAST_LIVE_OPENED'] = smallest(since_open, live)
    f['DAYS_SINCE_LAST_CLOSED'] = smallest(-r['DAYS_ENDDATE_FACT'], closed)
    f['DAYS_SINCE_LAST_UPDATE'] = smallest(since_update, everything)
    # type
    f['N_CREDIT_CARDS'], f['N_MORTGAGES'] = count(card), count(r['TYPE'] == 'Mortgage')
    f['N_CAR_LOANS'], f['N_MICROLOANS'] = count(r['TYPE'] == 'Car loan'), count(r['TYPE'] == 'Microloan')
    f['N_LIVE_CREDIT_CARDS'], f['N_LIVE_LOANS'] = count(live & card), count(live & ~card)
    f['N_LIVE_MICROLOANS'] = count(live & (r['TYPE'] == 'Microloan'))
    f['SHARE_CREDIT_CARDS'] = f['N_CREDIT_CARDS'] / f['N_CREDITS']
    f['N_TYPES'] = r.groupby('SK_ID_CURR')['TYPE'].nunique().astype(float)
    # overdue
    f['N_OVERDUE_NOW'], f['N_EVER_OVERDUE'], f['N_EVER_OVERDUE_1000'] = count(overdue_now), count(ever_overdue), count(ever_overdue_big)
    f['SHARE_EVER_OVERDUE_1000'] = f['N_EVER_OVERDUE_1000'] / f['N_CREDITS']
    f['N_EVER_OVERDUE_1000_24M'] = count(ever_overdue_big & (since_open <= RECENT_WINDOWS['24M']))
    f['DAYS_SINCE_LAST_OVERDUE_CREDIT_OPENED'] = smallest(since_open, ever_overdue_big)
    # debt and amounts
    f['LIVE_DEBT_SUM'], f['LIVE_DEBT_MAX'] = total(debt, live), largest(debt, live)
    f['LIVE_LOAN_AMOUNT_SUM'] = total(amount, live_loan)
    f['LIVE_LOAN_SHARE_OWED'] = total(debt, live_loan) / f['LIVE_LOAN_AMOUNT_SUM']
    f['LIVE_LOAN_MAX_SHARE_OWED'] = largest(debt / amount, live_loan)             # added after the design test in section 6.4
    f['N_LIVE_ZERO_DEBT'], f['N_LIVE_DEBT_BLANK'] = count(live & (debt == 0)), count(live & debt.isna())
    f['CARD_LIMIT_SUM'] = total(limit, live_card_limit)
    f['CARD_USE_OF_LIMIT'] = total(debt, live_card_limit) / (total(debt, live_card_limit) + f['CARD_LIMIT_SUM'])
    f['AMOUNT_MAX'] = largest(amount, everything)
    f['AMOUNT_MEDIAN'] = amount.groupby(key).median()
    # term
    f['LIVE_MAX_DAYS_REMAINING'] = largest(r['DAYS_CREDIT_ENDDATE'], live & (r['DAYS_CREDIT_ENDDATE'] > 0))
    f['N_LIVE_NO_END_DATE'] = count(live & r['DAYS_CREDIT_ENDDATE'].isna())
    f['SHARE_CLOSED_LATE'] = count(closed_dated & (early_by < -30)) / count(closed_dated).where(lambda s: s > 0)
    f['SHARE_CLOSED_EARLY'] = count(closed_dated & (early_by > 30)) / count(closed_dated).where(lambda s: s > 0)

    f = f.add_prefix('BUREAU_').reindex(applicant_ids)
    f['BUREAU_N_CREDITS'] = f['BUREAU_N_CREDITS'].fillna(0)          # no bureau record: a count of 0, everything else missing
    return f


def bureau_feature_table(raw, applicant_ids, screened=True):
    """Raw bureau rows to the feature table. With screened=True only the 40 features that passed screening are returned."""
    features = build_bureau_features(prepare_bureau(raw), applicant_ids)
    return features.drop(columns=DROPPED_FEATURES) if screened else features


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build the screened bureau feature table for the applicants of an application feature table.')
    parser.add_argument('bureau_csv')
    parser.add_argument('application_features', help='parquet file indexed by SK_ID_CURR; only its index is read')
    parser.add_argument('output', help='parquet file to write')
    args = parser.parse_args()

    ids = pd.read_parquet(args.application_features, columns=[]).index
    table = bureau_feature_table(pd.read_csv(args.bureau_csv), ids)
    table.to_parquet(args.output)
    print(f"{table.shape[0]} applicants, {table.shape[1]} features")
