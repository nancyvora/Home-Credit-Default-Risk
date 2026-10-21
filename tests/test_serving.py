"""Serving and the API: the model files are guarded, requests are checked, and live scores equal batch scores."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import MODEL_DIR, PROCESSED, RAW, make_applications
from src.app.app import app, score_form
from src.serving.inference import HISTORY_TABLES, applicant_features, load_model, predict, predict_table


@pytest.fixture(scope='module')
def client():
    return TestClient(app)


def record(row):
    return {k: (None if isinstance(v, float) and np.isnan(v) else (v.item() if isinstance(v, np.generic) else v)) for k, v in row.items()}


def test_served_model_loads_and_matches_its_checksum():
    model = load_model()
    assert len(model.features) == 111 and model.threshold == pytest.approx(1 / 11)


def test_a_changed_model_file_is_refused(tmp_path):
    for f in ('model.txt', 'metadata.json'):
        shutil.copy(MODEL_DIR / f, tmp_path / f)
    with open(tmp_path / 'model.txt', 'a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='checksum'):
        load_model.__wrapped__(tmp_path)


def test_unseen_text_value_is_treated_as_missing():
    model = load_model()
    X = applicant_features(record(make_applications(n=1).iloc[0]))
    unseen = X.assign(ORGANIZATION_TYPE='A company that did not exist')
    missing = X.assign(ORGANIZATION_TYPE=np.nan)
    assert model.predict_proba(unseen)[0] == model.predict_proba(missing)[0]


def test_prediction_has_probability_decision_and_reasons():
    result = predict(record(make_applications(n=1).iloc[0]), top_reasons=3)
    assert 0 < result['probability'] < 1 and result['decision'] in ('approve', 'decline') and len(result['reasons']) == 3
    assert (result['decision'] == 'decline') == (result['probability'] >= result['threshold'])


def test_contributions_rebuild_the_probability():
    model = load_model()
    X = applicant_features(record(make_applications(n=1).iloc[0]))
    reasons = model.explain(X, top=111)[0]
    raw = model.booster.predict(model.prepare(X), pred_contrib=True)[0]
    log_odds = sum(np.log(r['odds_multiplier']) for r in reasons) + model.slope * raw[-1] + model.intercept
    assert 1 / (1 + np.exp(-log_odds)) == pytest.approx(model.predict_proba(X)[0], abs=1e-3)      # odds rounded to 4 decimals


def test_bad_history_is_refused():
    application = record(make_applications(n=1).iloc[0])
    with pytest.raises(ValueError):
        predict(application, {'bureau_balance': []})
    with pytest.raises(ValueError):
        predict(application, {'bureau': [{'SK_ID_CURR': application['SK_ID_CURR'] + 1}]})


def test_api_health_and_model_card(client):
    assert client.get('/').json()['features'] == 111
    card = client.get('/model').json()
    assert card['evaluation']['holdout']['roc_auc'] == pytest.approx(0.7947, abs=1e-4)


def test_api_predict(client):
    body = {'application': record(make_applications(n=1).iloc[0])}
    response = client.post('/predict', content=json.dumps(body), headers={'content-type': 'application/json'})
    assert response.status_code == 200 and set(response.json()) == {'SK_ID_CURR', 'probability', 'decision', 'threshold', 'reasons'}
    assert response.json()['probability'] == pytest.approx(predict(body['application'])['probability'])


@pytest.mark.parametrize('body, status', [
    ({'application': {'AMT_CREDIT': 1.0}}, 422),                                           # no applicant id
    ({'application': {'SK_ID_CURR': 1}, 'threshold': 2}, 422),                              # threshold outside (0, 1)
    ({'application': {'SK_ID_CURR': 1}, 'history': {'bureau_balance': []}}, 400),           # a table the model does not use
    ({'application': {'SK_ID_CURR': 1}, 'history': {'bureau': [{'SK_ID_CURR': 2}]}}, 400),  # another applicant's records
])
def test_api_refuses_bad_requests(client, body, status):
    assert client.post('/predict', json=body).status_code == status


def test_form_scores_like_the_api():
    summary, reasons = score_form(45, 6, 'F', 'Higher education', 'Cash loans', 180000, 600000, 27000, 500000, 0.5, 0.6, 0.5)
    assert 'APPROVE' in summary or 'DECLINE' in summary
    assert all(r[0] not in ('DAYS_REGISTRATION', 'DAYS_ID_PUBLISH') or r[1] is not None for r in reasons)   # filled with typical values


def test_form_sends_an_unavailable_external_score_as_missing():
    args = (45, 6, 'F', 'Higher education', 'Cash loans', 180000, 600000, 27000, 500000, 0.5, 0.6, 0.5)
    with_all, _ = score_form(*args)
    without_1, reasons = score_form(*args, True, False, False)
    assert with_all != without_1                                                   # the score changes
    shown = {r[0]: r[1] for r in reasons}
    assert shown.get('EXT_SOURCE_1', None) is None                                 # and score 1 is missing, not 0.5


@pytest.mark.data
@pytest.mark.slow
def test_live_scores_equal_batch_scores():
    # training / serving consistency: raw records scored one at a time give the batch scores from the saved feature tables
    tables = ['application', 'bureau', 'previous', 'installments', 'credit_card']
    from src.data.load_data import load_joined
    X = load_joined(PROCESSED, 'test', tables)
    ids = list(X.index[::4000][:12])
    raw_app = pd.read_csv(RAW / 'application_test.csv').set_index('SK_ID_CURR', drop=False)
    names = {'bureau': 'bureau.csv', 'previous': 'previous_application.csv', 'installments': 'installments_payments.csv',
             'pos_cash': 'POS_CASH_balance.csv', 'credit_card': 'credit_card_balance.csv'}
    history = {i: {} for i in ids}
    for t in HISTORY_TABLES:
        raw = pd.read_csv(RAW / names[t])
        for i, g in raw[raw['SK_ID_CURR'].isin(ids)].groupby('SK_ID_CURR'):
            history[i][t] = [record(r) for r in g.to_dict(orient='records')]
    batch = predict_table(X.loc[ids])
    live = [predict(record(raw_app.loc[i].to_dict()), history[i])['probability'] for i in ids]
    assert np.allclose(live, batch, rtol=0, atol=1e-6)


@pytest.mark.data
def test_batch_scores_reproduce_the_submission():
    from src.data.load_data import load_joined
    X = load_joined(PROCESSED, 'test', ['application', 'bureau', 'previous', 'installments', 'credit_card'])
    submission = pd.read_csv(PROCESSED.parent.parent / 'artifacts' / 'submissions' / 'submission_with_credit_cards.csv').set_index('SK_ID_CURR')['TARGET']
    assert np.allclose(predict_table(X), submission.loc[X.index].values, rtol=0, atol=1e-12)
