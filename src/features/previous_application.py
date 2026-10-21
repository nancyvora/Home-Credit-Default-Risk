"""Features from the applicant's earlier applications to Home Credit, one row per applicant.

This is the code developed and verified in ``notebooks/05_previous_application_eda_features.ipynb``, in a form that
can be used without the notebook. The notebook checks that this module reproduces its saved tables exactly.

Three steps:

- ``clean_previous``            nine rules: superseded and repeated records, placeholders, zero amounts, unused columns
- ``prepare_previous``          the same, with a guard that nothing is dated on or after the current application
- ``build_previous_features``   53 summaries per applicant; ``DROPPED_FEATURES`` lists the 12 removed in screening

Properties the notebook verified, which callers may rely on:

- every feature of an applicant is computed from that applicant's own rows, so the table can be built once and
  used inside cross-validation without being rebuilt in each fold;
- no function uses the target, and none contains a number estimated from the data;
- the current application is not in the table, and no decision or loan event is dated after it.

Usage::

    from previous_application import previous_feature_table
    features = previous_feature_table(pd.read_csv('data/raw/previous_application.csv'), applicant_ids)
"""
import numpy as np
import pandas as pd

DATE_PLACEHOLDER = 365243
LOAN_DATES = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
DROPPED_COLUMNS = ['RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'AMT_GOODS_PRICE', 'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY']

WINDOW_EDGE_DAYS = 7 * 365.25
YIELD_LEVEL = {'low_action': 0.0, 'low_normal': 1.0, 'middle': 2.0, 'high': 3.0}
MAIN_REASONS = ['HC', 'LIMIT', 'SCO', 'SCOFR']

# removed in stage 8 of the notebook: seven near-copies of another feature, and five with no signal
DROPPED_FEATURES = ['PREV_SHARE_REFUSED', 'PREV_SHARE_REFUSED_24M', 'PREV_SHARE_APPROVED_CASH', 'PREV_SHARE_WALK_IN', 'PREV_COST_OF_CREDIT_MAX', 'PREV_AMT_CREDIT_MAX',
                    'PREV_RUNNING_CREDIT_SUM', 'PREV_N_UNUSED_OFFER', 'PREV_N_REFUSED_SCO', 'PREV_DAYS_SINCE_LAST_APPROVAL', 'PREV_N_APPROVED_12M', 'PREV_N_LOANS_ENDED_EARLY']


def clean_previous(raw, report=None):
    # Applies the nine rules of section 3.7. Uses only each row's own values, plus the other rows of the same applicant for rule 2.
    note = (lambda rule, n: report.__setitem__(rule, int(n))) if report is not None else (lambda rule, n: None)
    df = raw.sort_values('SK_ID_PREV')

    # 1. extra records for a contract that has another record
    superseded = df['FLAG_LAST_APPL_PER_CONTRACT'] == 'N'
    note('1. rows not the last application for their contract, removed', superseded.sum())
    df = df[~superseded]

    # 2. repeated records: keep the one with the lowest SK_ID_PREV
    repeated = df.duplicated(subset=[c for c in df.columns if c != 'SK_ID_PREV'], keep='first')
    note('2. repeated rows removed', repeated.sum())
    df = df[~repeated].copy()

    # 3. placeholders in text columns
    text = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    is_placeholder = df[text].isin(['XNA', 'XAP'])
    note('3. text values that were XNA or XAP, set to missing', is_placeholder.values.sum())
    df[text] = df[text].where(~is_placeholder)

    # 4. the date placeholder, keeping what it says about a loan still running
    df['LAST_DUE_OPEN'] = (df['DAYS_LAST_DUE'] == DATE_PLACEHOLDER).astype(float).where(df['DAYS_LAST_DUE'].notna())
    df['TERMINATION_OPEN'] = (df['DAYS_TERMINATION'] == DATE_PLACEHOLDER).astype(float).where(df['DAYS_TERMINATION'].notna())
    for c in LOAN_DATES:
        placeholder = df[c] == DATE_PLACEHOLDER
        note(f'4. {c} of 365243, set to missing', placeholder.sum())
        df[c] = df[c].where(~placeholder)

    # 5. an amount of zero is no amount
    for c in ['AMT_APPLICATION', 'AMT_CREDIT']:
        zero = df[c] == 0
        note(f'5. {c} of zero, set to missing', zero.sum())
        df[c] = df[c].where(~zero)

    # 6. no fixed term
    no_term = df['CNT_PAYMENT'] == 0
    note('6. CNT_PAYMENT of zero, set to missing', no_term.sum())
    df['CNT_PAYMENT'] = df['CNT_PAYMENT'].where(~no_term)

    # 7. placeholder for the seller's area
    no_area = df['SELLERPLACE_AREA'] == -1
    note('7. SELLERPLACE_AREA of -1, set to missing', no_area.sum())
    df['SELLERPLACE_AREA'] = df['SELLERPLACE_AREA'].astype(float).where(~no_area)

    # 8. impossible down payments
    negative = df['AMT_DOWN_PAYMENT'] < 0
    note('8. negative down payments, set to missing', negative.sum())
    df.loc[negative, ['AMT_DOWN_PAYMENT', 'RATE_DOWN_PAYMENT']] = np.nan

    # 9. columns with nothing left to give
    return df.drop(columns=DROPPED_COLUMNS).reset_index(drop=True)


def prepare_previous(raw, report=None):
    # The nine cleaning rules of section 3.7, then a guard: nothing dated on or after the current application, apart from planned end dates.
    df = clean_previous(raw, report)
    assert (df['DAYS_DECISION'] < 0).all(), 'an earlier application is dated on or after the current one'
    for c in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
        assert not (df[c] > 0).any(), f'{c} has a date after the current application'
    return df


def build_previous_features(ready, applicant_ids):
    # ready: the output of prepare_previous. Returns one row per applicant in applicant_ids, in that order.
    # Every feature is computed from the applicant's own earlier applications only.
    r = ready.sort_values(['SK_ID_CURR', 'DAYS_DECISION', 'SK_ID_PREV']).reset_index(drop=True)       # oldest first; ties on a day by SK_ID_PREV
    key = r['SK_ID_CURR']
    status, ctype, since = r['NAME_CONTRACT_STATUS'], r['NAME_CONTRACT_TYPE'], -r['DAYS_DECISION']
    approved, refused, canceled, unused = status == 'Approved', status == 'Refused', status == 'Canceled', status == 'Unused offer'
    decided = approved | refused
    cost = (r['AMT_ANNUITY'] * r['CNT_PAYMENT'] / r['AMT_CREDIT']).where(approved)
    level = r['NAME_YIELD_GROUP'].map(YIELD_LEVEL).where(approved)
    running, ended = approved & (r['TERMINATION_OPEN'] == 1), approved & (r['TERMINATION_OPEN'] == 0)
    ended_early = ended & ((r['DAYS_LAST_DUE_1ST_VERSION'] - r['DAYS_LAST_DUE']) > 30)

    count = lambda flag: flag.astype(float).groupby(key).sum()
    total = lambda values, flag: values.where(flag).groupby(key).sum(min_count=1)
    mean = lambda values, flag: values.where(flag).groupby(key).mean()
    smallest = lambda values, flag: values.where(flag).groupby(key).min()
    largest = lambda values, flag: values.where(flag).groupby(key).max()
    latest = lambda values, flag: values.where(flag).groupby(key).last()                    # the most recent row that has a value
    everything = pd.Series(True, index=r.index)
    ratio = lambda a, b: a / b.where(b > 0)

    f = pd.DataFrame(index=pd.Index(np.sort(key.unique()), name='SK_ID_CURR'))
    # outcomes
    f['N_APPLICATIONS'] = count(everything)
    f['N_APPROVED'], f['N_REFUSED'], f['N_CANCELED'], f['N_UNUSED_OFFER'] = count(approved), count(refused), count(canceled), count(unused)
    f['SHARE_REFUSED'] = ratio(f['N_REFUSED'], count(decided))
    f['LAST_DECIDED_WAS_REFUSED'] = latest(refused.astype(float), decided)
    # refusals
    f['N_REFUSED_12M'], f['N_REFUSED_24M'] = count(refused & (since <= 365)), count(refused & (since <= 730))
    f['SHARE_REFUSED_24M'] = ratio(f['N_REFUSED_24M'], count(decided & (since <= 730)))
    f['DAYS_SINCE_LAST_REFUSAL'] = smallest(since, refused)
    for reason in MAIN_REASONS:
        f[f'N_REFUSED_{reason}'] = count(refused & (r['CODE_REJECT_REASON'] == reason))
    f['N_REFUSED_OTHER'] = count(refused & ~r['CODE_REJECT_REASON'].isin(MAIN_REASONS))
    # timing
    f['DAYS_SINCE_LAST_DECISION'], f['DAYS_SINCE_FIRST_DECISION'] = smallest(since, everything), largest(since, everything)
    f['HISTORY_REACHES_WINDOW_EDGE'] = (f['DAYS_SINCE_FIRST_DECISION'] >= WINDOW_EDGE_DAYS).astype(float)
    f['DAYS_SINCE_LAST_APPROVAL'] = smallest(since, approved)
    f['N_DECIDED_12M'], f['N_APPROVED_12M'] = count(decided & (since <= 365)), count(approved & (since <= 365))
    f['MEAN_DAYS_BETWEEN_DECISIONS'] = ((f['DAYS_SINCE_FIRST_DECISION'] - f['DAYS_SINCE_LAST_DECISION']) / (f['N_APPLICATIONS'] - 1)).where(f['N_APPLICATIONS'] > 1)
    f['HOUR_MEAN'] = mean(r['HOUR_APPR_PROCESS_START'].astype(float), decided)
    # product and channel
    f['N_APPROVED_CASH'], f['N_APPROVED_CONSUMER'] = count(approved & (ctype == 'Cash loans')), count(approved & (ctype == 'Consumer loans'))
    f['N_APPROVED_REVOLVING'] = count(approved & (ctype == 'Revolving loans'))
    f['SHARE_APPROVED_CASH'] = ratio(f['N_APPROVED_CASH'], f['N_APPROVED'])
    f['N_WALK_IN'], f['N_X_SELL'] = count(decided & (r['NAME_PRODUCT_TYPE'] == 'walk-in')), count(decided & (r['NAME_PRODUCT_TYPE'] == 'x-sell'))
    f['SHARE_WALK_IN'] = ratio(f['N_WALK_IN'], count(decided))
    f['N_CHANNEL_AP_CASH'] = count(decided & (r['CHANNEL_TYPE'] == 'AP+ (Cash loan)'))
    f['N_CLIENT_TYPE_NEW'] = count(decided & (r['NAME_CLIENT_TYPE'] == 'New'))
    f['N_REFUSED_CASH'], f['N_REFUSED_REVOLVING'] = count(refused & (ctype == 'Cash loans')), count(refused & (ctype == 'Revolving loans'))
    # pricing and terms, approved applications only
    f['COST_OF_CREDIT_MEAN'], f['COST_OF_CREDIT_MAX'], f['COST_OF_CREDIT_LAST'] = mean(cost, approved), largest(cost, approved), latest(cost, approved)
    f['RATE_PER_PAYMENT_MEAN'] = mean((cost - 1) / r['CNT_PAYMENT'], approved)          # added after the design test in section 6.4
    f['YIELD_LEVEL_MEAN'], f['YIELD_LEVEL_LAST'] = mean(level, approved), latest(level, approved)
    f['N_APPROVED_HIGH_YIELD'] = count(approved & (r['NAME_YIELD_GROUP'] == 'high'))
    f['AMT_CREDIT_MEAN'], f['AMT_CREDIT_MAX'] = mean(r['AMT_CREDIT'], approved), largest(r['AMT_CREDIT'], approved)
    f['AMT_ANNUITY_MEAN'] = mean(r['AMT_ANNUITY'], approved)
    f['GRANTED_TO_REQUESTED_MEAN'] = mean(r['AMT_CREDIT'] / r['AMT_APPLICATION'], approved)
    f['DOWN_PAYMENT_RATE_MEAN'] = mean(r['RATE_DOWN_PAYMENT'], approved)
    # loan state
    f['N_LOANS_ENDED'], f['N_LOANS_RUNNING'], f['N_LOANS_ENDED_EARLY'] = count(ended), count(running), count(ended_early)
    f['RUNNING_ANNUITY_SUM'], f['RUNNING_CREDIT_SUM'] = total(r['AMT_ANNUITY'], running), total(r['AMT_CREDIT'], running)
    f['RUNNING_MAX_DAYS_REMAINING'] = largest(r['DAYS_LAST_DUE_1ST_VERSION'], running & (r['DAYS_LAST_DUE_1ST_VERSION'] > 0))

    f = f.add_prefix('PREV_').reindex(applicant_ids)
    f['PREV_N_APPLICATIONS'] = f['PREV_N_APPLICATIONS'].fillna(0)          # no earlier application: a count of 0, everything else missing
    return f


def previous_feature_table(raw, applicant_ids, screened=True):
    """Raw previous_application rows to the feature table. With screened=True only the 41 features that passed screening are returned."""
    features = build_previous_features(prepare_previous(raw), applicant_ids)
    return features.drop(columns=DROPPED_FEATURES) if screened else features


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build the screened previous-application feature table for the applicants of an application feature table.')
    parser.add_argument('previous_application_csv')
    parser.add_argument('application_features', help='parquet file indexed by SK_ID_CURR; only its index is read')
    parser.add_argument('output', help='parquet file to write')
    args = parser.parse_args()

    ids = pd.read_parquet(args.application_features, columns=[]).index
    table = previous_feature_table(pd.read_csv(args.previous_application_csv), ids)
    table.to_parquet(args.output)
    print(f"{table.shape[0]} applicants, {table.shape[1]} features")
