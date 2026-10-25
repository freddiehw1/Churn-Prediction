import numpy as np
import pandas as pd
import pytest

from data_utils import FEATURES, TARGET, add_features, load_data
from model_utils import (breakeven_threshold, choose_threshold, make_model, net_benefit, split_data,
                         MODEL_NAMES, FEATURE_SETS)


@pytest.fixture(scope='module')
def df():
    return load_data()


def test_data_is_valid(df):
    assert len(df) == 10_000
    assert set(df[TARGET].unique()) == {0, 1}
    assert df[FEATURES].isna().sum().sum() == 0


def test_split_is_stratified_and_disjoint(df):
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_test) == 2000
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.005


def test_features_do_not_use_target(df):
    out = add_features(df[FEATURES])
    assert TARGET not in out.columns
    assert out.loc[df['NumOfProducts'] >= 3, 'Products3Plus'].eq(1).all()
    assert out.loc[df['Balance'] == 0, 'ZeroBalance'].eq(1).all()


def test_no_gender_model_ignores_gender(df):
    X = df[FEATURES].head(2000)
    y = df[TARGET].head(2000)
    model = make_model('Logistic regression', 'Engineered, no gender').fit(X, y)
    flipped = X.assign(Gender=X['Gender'].map({'Male': 'Female', 'Female': 'Male'}))
    np.testing.assert_allclose(model.predict_proba(X), model.predict_proba(flipped))


@pytest.mark.parametrize('name', MODEL_NAMES)
@pytest.mark.parametrize('features', list(FEATURE_SETS))
def test_every_pipeline_scores_a_raw_record(df, name, features):
    model = make_model(name, features).fit(df[FEATURES].head(1500), df[TARGET].head(1500))
    p = model.predict_proba(df[FEATURES].tail(5))[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_net_benefit_arithmetic():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.1, 0.05])
    # Contact the two customers above 0.5: one churner reached -> 1 * 0.5 * 100 - 2 * 10 = 30
    assert net_benefit(y, p, 0.5, cost_per_contact=10, value_per_save=100, save_rate=0.5) == 30
    assert net_benefit(y, p, 0.99, cost_per_contact=10, value_per_save=100, save_rate=0.5) == 0


def test_chosen_threshold_is_near_breakeven_for_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50_000)
    y = (rng.uniform(0, 1, 50_000) < p).astype(int)  # perfectly calibrated by construction
    econ = {'cost_per_contact': 5, 'value_per_save': 100, 'save_rate': 0.2}
    t, _ = choose_threshold(y, p, econ)
    assert abs(t - breakeven_threshold(econ)) <= 0.03


def test_saved_results_load_for_the_app():
    from model_utils import load_results
    r = load_results()
    assert r is not None, 'Run `python train.py` to create artifacts/results.joblib'
    assert 0 <= r['threshold'] <= 1
    p = r['final_model'].predict_proba(r['X_test'].head(3))[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
