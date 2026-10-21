"""A made-up credit history from a few plain answers, for the interface.

The model reads an applicant's history as records: credit bureau rows, earlier Home Credit applications, instalment
payments and monthly credit card balances. A visitor cannot be asked for those, so ``make_history`` builds a small,
plausible history from a handful of answers (how many loans, how much still owed, how often late, ...). The records
have the columns of the Kaggle files and every date is before the application, so they go through exactly the same
preprocessing, features and model as real records.

It is an illustration: a real credit file holds far more detail than these answers describe.
"""
APPLICANT = 0            # the interface's applicant id
LOAN = 54_000.0          # an earlier Home Credit consumer loan
TERM = 12                # its monthly instalments
LATE_DAYS = 15           # how late a late instalment is paid
CARD_LIMIT = 135_000.0


def _bureau(closed, open_, share_owed, overdue_now):
    rows = []
    for i in range(int(closed)):
        opened = -2400 + 300 * i
        rows.append({'SK_ID_BUREAU': 9_100_000 + i, 'CREDIT_ACTIVE': 'Closed', 'CREDIT_CURRENCY': 'currency 1', 'DAYS_CREDIT': opened,
                     'CREDIT_DAY_OVERDUE': 0, 'DAYS_CREDIT_ENDDATE': opened + 365, 'DAYS_ENDDATE_FACT': opened + 360, 'AMT_CREDIT_MAX_OVERDUE': 0.0,
                     'CNT_CREDIT_PROLONG': 0, 'AMT_CREDIT_SUM': 120_000.0, 'AMT_CREDIT_SUM_DEBT': 0.0, 'AMT_CREDIT_SUM_LIMIT': 0.0,
                     'AMT_CREDIT_SUM_OVERDUE': 0.0, 'CREDIT_TYPE': 'Consumer credit', 'DAYS_CREDIT_UPDATE': opened + 365, 'AMT_ANNUITY': None})
    open_ = max(int(open_), 1 if overdue_now else 0)          # being overdue needs a loan that is still open
    for i in range(open_):
        opened = -200 - 250 * i
        overdue = overdue_now and i == 0
        rows.append({'SK_ID_BUREAU': 9_150_000 + i, 'CREDIT_ACTIVE': 'Active', 'CREDIT_CURRENCY': 'currency 1', 'DAYS_CREDIT': opened,
                     'CREDIT_DAY_OVERDUE': 45 if overdue else 0, 'DAYS_CREDIT_ENDDATE': opened + 1095, 'DAYS_ENDDATE_FACT': None,
                     'AMT_CREDIT_MAX_OVERDUE': 8_000.0 if overdue else 0.0, 'CNT_CREDIT_PROLONG': 0, 'AMT_CREDIT_SUM': 200_000.0,
                     'AMT_CREDIT_SUM_DEBT': 200_000.0 * share_owed / 100, 'AMT_CREDIT_SUM_LIMIT': 0.0,
                     'AMT_CREDIT_SUM_OVERDUE': 8_000.0 if overdue else 0.0, 'CREDIT_TYPE': 'Consumer credit', 'DAYS_CREDIT_UPDATE': -5, 'AMT_ANNUITY': None})
    return rows


def _previous(loans, refused, has_card):
    rows, starts = [], []
    for i in range(int(loans)):
        decided = -2200 + 365 * i if loans > 1 else -900
        starts.append(decided + 30)
        rows.append({'SK_ID_PREV': 9_200_000 + i, 'NAME_CONTRACT_TYPE': 'Consumer loans', 'AMT_ANNUITY': LOAN / TERM * 1.1, 'AMT_APPLICATION': LOAN,
                     'AMT_CREDIT': LOAN, 'AMT_DOWN_PAYMENT': 0.0, 'AMT_GOODS_PRICE': LOAN, 'WEEKDAY_APPR_PROCESS_START': 'SATURDAY',
                     'HOUR_APPR_PROCESS_START': 12, 'FLAG_LAST_APPL_PER_CONTRACT': 'Y', 'NFLAG_LAST_APPL_IN_DAY': 1, 'RATE_DOWN_PAYMENT': 0.0,
                     'NAME_CASH_LOAN_PURPOSE': 'XAP', 'NAME_CONTRACT_STATUS': 'Approved', 'DAYS_DECISION': decided, 'NAME_PAYMENT_TYPE': 'Cash through the bank',
                     'CODE_REJECT_REASON': 'XAP', 'NAME_TYPE_SUITE': 'Unaccompanied', 'NAME_CLIENT_TYPE': 'Repeater' if i else 'New',
                     'NAME_GOODS_CATEGORY': 'Mobile', 'NAME_PORTFOLIO': 'POS', 'NAME_PRODUCT_TYPE': 'XNA', 'CHANNEL_TYPE': 'Country-wide',
                     'SELLERPLACE_AREA': 50, 'NAME_SELLER_INDUSTRY': 'Connectivity', 'CNT_PAYMENT': float(TERM), 'NAME_YIELD_GROUP': 'middle',
                     'PRODUCT_COMBINATION': 'POS mobile with interest', 'DAYS_FIRST_DRAWING': 365243.0, 'DAYS_FIRST_DUE': decided + 30,
                     'DAYS_LAST_DUE_1ST_VERSION': decided + 30 * TERM, 'DAYS_LAST_DUE': decided + 30 * TERM, 'DAYS_TERMINATION': decided + 30 * TERM + 10,
                     'NFLAG_INSURED_ON_APPROVAL': 0})
    for i in range(int(refused)):
        rows.append({'SK_ID_PREV': 9_250_000 + i, 'NAME_CONTRACT_TYPE': 'Cash loans', 'AMT_ANNUITY': 15_000.0, 'AMT_APPLICATION': 270_000.0,
                     'AMT_CREDIT': 270_000.0, 'WEEKDAY_APPR_PROCESS_START': 'MONDAY', 'HOUR_APPR_PROCESS_START': 11,
                     'FLAG_LAST_APPL_PER_CONTRACT': 'Y', 'NFLAG_LAST_APPL_IN_DAY': 1, 'NAME_CASH_LOAN_PURPOSE': 'XNA', 'NAME_CONTRACT_STATUS': 'Refused',
                     'DAYS_DECISION': -60 - 90 * i, 'NAME_PAYMENT_TYPE': 'Cash through the bank', 'CODE_REJECT_REASON': 'HC', 'NAME_CLIENT_TYPE': 'Repeater',
                     'NAME_GOODS_CATEGORY': 'XNA', 'NAME_PORTFOLIO': 'Cash', 'NAME_PRODUCT_TYPE': 'x-sell', 'CHANNEL_TYPE': 'Credit and cash offices',
                     'SELLERPLACE_AREA': -1, 'NAME_SELLER_INDUSTRY': 'XNA', 'CNT_PAYMENT': 24.0, 'NAME_YIELD_GROUP': 'high',
                     'PRODUCT_COMBINATION': 'Cash X-Sell: high'})
    if has_card:
        rows.append({'SK_ID_PREV': 9_300_000, 'NAME_CONTRACT_TYPE': 'Revolving loans', 'AMT_ANNUITY': 6_750.0, 'AMT_APPLICATION': 0.0,
                     'AMT_CREDIT': CARD_LIMIT, 'WEEKDAY_APPR_PROCESS_START': 'FRIDAY', 'HOUR_APPR_PROCESS_START': 14,
                     'FLAG_LAST_APPL_PER_CONTRACT': 'Y', 'NFLAG_LAST_APPL_IN_DAY': 1, 'NAME_CASH_LOAN_PURPOSE': 'XAP', 'NAME_CONTRACT_STATUS': 'Approved',
                     'DAYS_DECISION': -420, 'NAME_PAYMENT_TYPE': 'XNA', 'CODE_REJECT_REASON': 'XAP', 'NAME_CLIENT_TYPE': 'Repeater',
                     'NAME_GOODS_CATEGORY': 'XNA', 'NAME_PORTFOLIO': 'Cards', 'NAME_PRODUCT_TYPE': 'x-sell', 'CHANNEL_TYPE': 'Credit and cash offices',
                     'SELLERPLACE_AREA': -1, 'NAME_SELLER_INDUSTRY': 'XNA', 'CNT_PAYMENT': 0.0, 'NAME_YIELD_GROUP': 'XNA', 'PRODUCT_COMBINATION': 'Card X-Sell',
                     'DAYS_FIRST_DRAWING': -400.0, 'DAYS_FIRST_DUE': -390.0, 'DAYS_LAST_DUE_1ST_VERSION': 365243.0, 'DAYS_LAST_DUE': 365243.0,
                     'DAYS_TERMINATION': 365243.0, 'NFLAG_INSURED_ON_APPROVAL': 0})
    return rows, starts


def _instalments(starts, share_late):
    # the late ones are spread evenly over all instalments, the most recent included, so the share is what was asked for
    total = len(starts) * TERM
    n_late = round(total * share_late / 100)
    late = {round((k + 0.5) * total / n_late - 0.5) for k in range(n_late)} if n_late else set()
    rows, k = [], 0
    for loan, start in enumerate(starts):
        for n in range(1, TERM + 1):
            due = start + 30 * (n - 1)
            paid = due + (LATE_DAYS if k in late else -2)
            rows.append({'SK_ID_PREV': 9_200_000 + loan, 'NUM_INSTALMENT_VERSION': 1, 'NUM_INSTALMENT_NUMBER': n, 'DAYS_INSTALMENT': due,
                         'DAYS_ENTRY_PAYMENT': min(paid, -1), 'AMT_INSTALMENT': LOAN / TERM * 1.1, 'AMT_PAYMENT': LOAN / TERM * 1.1})
            k += 1
    return rows


def _card(balance_pct, atm_12m):
    balance = CARD_LIMIT * balance_pct / 100
    rows = []
    for m in range(-12, 0):                                 # the last twelve months
        atm = int(atm_12m) // 12 + (1 if (m + 12) < int(atm_12m) % 12 else 0)
        min_due = max(balance * 0.05, 0.0)
        rows.append({'SK_ID_PREV': 9_300_000, 'MONTHS_BALANCE': m, 'AMT_BALANCE': balance, 'AMT_CREDIT_LIMIT_ACTUAL': CARD_LIMIT,
                     'AMT_DRAWINGS_ATM_CURRENT': 4_500.0 * atm, 'AMT_DRAWINGS_CURRENT': 4_500.0 * atm, 'AMT_DRAWINGS_OTHER_CURRENT': 0.0,
                     'AMT_DRAWINGS_POS_CURRENT': 0.0, 'AMT_INST_MIN_REGULARITY': min_due, 'AMT_PAYMENT_CURRENT': min_due,
                     'AMT_PAYMENT_TOTAL_CURRENT': min_due, 'AMT_RECEIVABLE_PRINCIPAL': balance, 'AMT_RECIVABLE': balance, 'AMT_TOTAL_RECEIVABLE': balance,
                     'CNT_DRAWINGS_ATM_CURRENT': float(atm), 'CNT_DRAWINGS_CURRENT': atm, 'CNT_DRAWINGS_OTHER_CURRENT': 0.0, 'CNT_DRAWINGS_POS_CURRENT': 0.0,
                     'CNT_INSTALMENT_MATURE_CUM': float(m + 13), 'NAME_CONTRACT_STATUS': 'Active', 'SK_DPD': 0, 'SK_DPD_DEF': 0})
    return rows


def make_history(bureau_closed=0, bureau_open=0, share_owed=50, overdue_now=False, hc_loans=0, hc_refused=0, share_late=0,
                 has_card=False, card_balance_pct=30, card_atm_12m=0):
    """The answers of the interface's history section as records of each table (empty tables are left out)."""
    clip = lambda v, low, high: min(max(float(v or 0), low), high)
    previous, starts = _previous(clip(hc_loans, 0, 10), clip(hc_refused, 0, 10), bool(has_card))
    history = {
        'bureau': _bureau(clip(bureau_closed, 0, 15), clip(bureau_open, 0, 10), clip(share_owed, 0, 100), bool(overdue_now)),
        'previous': previous,
        'installments': _instalments(starts, clip(share_late, 0, 100)),
        'credit_card': _card(clip(card_balance_pct, 0, 150), clip(card_atm_12m, 0, 120)) if has_card else [],
    }
    return {table: [{'SK_ID_CURR': APPLICANT, **r} for r in rows] for table, rows in history.items() if rows}
