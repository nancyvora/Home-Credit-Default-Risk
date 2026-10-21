"""Train, evaluate, tune: the recipe from the config, the measures, the change rule, and the saved-model round trip."""
import copy

import numpy as np
import pandas as pd
import pytest

from conftest import PROCESSED, make_applications
from src.data.preprocess import preprocess
from src.features.build_features import build_features
from src.models.evaluate import calibration_line, decisions, evaluate_model, holdout_report
from src.models.train import build_model, save_model, text_columns, train_model
from src.models.tune import four_conditions
from src.serving.inference import load_model
from src.utils.utils import load_config


@pytest.fixture(scope='module')
def cfg():
    return load_config()


def test_config_describes_the_final_model(cfg):
    assert len(cfg.features) == 111 and cfg['model']['trees'] == 3500 and cfg['model']['learning_rate'] == 0.02
    assert cfg.model_tables == ['application', 'bureau', 'previous', 'installments', 'credit_card']
    assert cfg['decision']['reference_threshold'] == pytest.approx(1 / 11)


def test_config_matches_the_served_model(cfg):
    served = load_model().metadata
    assert served['features'] == cfg.features and served['training']['lightgbm_settings'] == cfg['model']['lightgbm']


def test_a_different_seed_is_refused(cfg):
    other = copy.deepcopy(cfg)
    other['model']['seed'] = 7
    with pytest.raises(ValueError):
        build_model(other, [])


def test_four_conditions():
    assert four_conditions([0.001, 0.0012, 0.0009, 0.0011, 0.001])['passed']
    assert not four_conditions([0.0003, 0.0004, 0.0002, 0.0003, 0.0004])['passed']          # under the noise floor
    assert not four_conditions([0.003, 0.003, 0.003, -0.001, -0.001])['passed']             # better in only 3 folds
    result = four_conditions([0.003, -0.001, 0.003, 0.0001, 0.002])                       # mean above the floor, but noisy
    assert result['folds_better'] == 4 and not result['conditions']['3. gain is more than twice its standard error']


def test_measures_on_known_predictions():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.05, 0.10, 0.20, 0.60, 0.70, 0.08])
    d = decisions(y, p, threshold=0.5)
    assert d['confusion matrix'] == {'approved, repaid': 3, 'declined, would have repaid': 1, 'approved, defaulted': 1, 'declined, would have defaulted': 1}
    assert d['declined %'] == pytest.approx(100 * 2 / 6) and d['defaults caught %'] == pytest.approx(50)
    assert evaluate_model(y, p, 0.5)['ROC AUC'] == pytest.approx(5 / 8)


def test_calibrated_predictions_have_slope_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.6, 200_000)
    y = (rng.random(200_000) < p).astype(int)
    slope, intercept = calibration_line(y, p)
    assert slope == pytest.approx(1, abs=0.03) and intercept == pytest.approx(0, abs=0.05)


def test_holdout_report_intervals():
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.1).astype(int)
    good, weak = np.clip(0.1 + 0.3 * y + rng.normal(0, 0.15, 4000), 0.001, 0.999), rng.uniform(0.01, 0.99, 4000)
    r = holdout_report(y, good, 0.3, p_reference=weak, n_boot=200)
    low, high = r['ROC AUC 95% interval']
    assert low < r['ROC AUC'] < high and r['gain 95% interval'][0] > 0


@pytest.mark.slow
def test_train_save_and_load_give_the_same_predictions(cfg, tmp_path):
    # a small, quick version of the recipe on synthetic applicants: the saved files reproduce the fitted pipeline
    small = copy.deepcopy(cfg)
    small['model']['trees'] = 30
    small['model']['lightgbm']['min_child_samples'] = 20
    small['features'] = {'application': cfg['features']['application']}
    rows = make_applications(n=600, target=True, seed=3)
    X = build_features('application', preprocess('application', rows))
    y = rows['TARGET'].to_numpy()
    model = train_model(X, y, small)
    meta = save_model(model, small, tmp_path, rows=len(X), defaults=int(y.sum()))
    assert meta['features'] == small.features and len(meta['text_feature_levels']) == len([c for c in small.features if c in text_columns(X)])
    loaded = load_model.__wrapped__(tmp_path)
    assert np.allclose(loaded.predict_proba(X), model.predict_proba(X)[:, 1], rtol=0, atol=1e-12)


@pytest.mark.data
def test_saved_feature_tables_join_in_order(cfg):
    from src.data.load_data import load_joined
    X = load_joined(PROCESSED, 'test', cfg.model_tables)
    assert len(X) == 48_744 and set(cfg.features) <= set(X.columns) and X.index.is_unique
