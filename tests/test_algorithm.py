"""XGBoost gegen unabhängige Referenzen: die Gain-Formel gegen Brute-Force über alle Schwellen, die Vorhersagen für wenige Runden fast exakt (Rundungsrauschen ~1e-7) gegen die echte `xgboost`-Bibliothek
(`tree_method="exact"`, gleiche Parameter), mit lambda=0/gamma=0 identisch mit gradient-boosting-demo bis unabhängige Fließkomma-Gleichstände bei sehr kleinen Residuen nach vielen Runden auseinanderlaufen
(siehe README) - deshalb wird der exakte Vergleich auf wenige Runden begrenzt, danach nur noch Korrelation/Toleranz geprüft."""

import sys
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

import xgb_algorithm as xgm
import xgb_scenario as S
import xgb_tree as T

GB_DIR = Path(__file__).resolve().parents[2] / "gradient-boosting-demo"


def _reg(n=300, d=5, seed=0, noise=0.4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] * X[:, 3] + rng.normal(0, noise, n)
    return X, y


def _cls(n=300, d=5, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * np.sin(X[:, 1]) + rng.normal(0, noise, n) > 0).astype(float)
    return X, y


# --- Gain-Formel gegen Brute-Force -----------------------------------------------------------------------------------------------------------------

def test_gain_matrix_matches_brute_force_search():
    rng = np.random.default_rng(1)
    n = 40
    X = rng.normal(size=(n, 3))
    grad = rng.normal(size=n)
    hess = rng.uniform(0.2, 2.0, n)
    lam, gamma, mcw = 1.5, 0.3, 0.5
    f, thr, gain = T.best_split(X, grad, hess, lam, gamma, mcw)
    Gtot, Htot = grad.sum(), hess.sum()
    best = None
    for ff in range(3):
        order = np.argsort(X[:, ff], kind="stable")
        xs = X[order, ff]
        for i in range(1, n):
            if xs[i - 1] == xs[i]:
                continue
            left, right = order[:i], order[i:]
            Gl, Hl = grad[left].sum(), hess[left].sum()
            Gr, Hr = grad[right].sum(), hess[right].sum()
            if Hl < mcw or Hr < mcw:
                continue
            g = 0.5 * (Gl ** 2 / (Hl + lam) + Gr ** 2 / (Hr + lam) - Gtot ** 2 / (Htot + lam)) - gamma
            if best is None or g > best[2]:
                best = (ff, (xs[i - 1] + xs[i]) / 2.0, g)
    assert f == best[0] and thr == pytest.approx(best[1]) and gain == pytest.approx(best[2], abs=1e-9)


def test_split_is_rejected_when_no_gain_is_positive():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 2))
    grad = rng.normal(0, 0.01, 30)                                        # winzige Gradienten: kein Split lohnt sich gegen ein hohes gamma
    hess = np.ones(30)
    assert T.best_split(X, grad, hess, lam=1.0, gamma=50.0, min_child_weight=1.0) is None


# --- Fast exakt gegen die echte xgboost-Bibliothek (wenige Runden) ---------------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [1, 2, 3])
def test_regression_matches_xgboost_library_closely_for_few_rounds(depth):
    X, y = _reg(300, 5, 0)
    Xt, _ = _reg(60, 5, 1)
    lam, gamma, mcw, lr = 1.5, 0.5, 2.0, 0.2
    ens = xgm.fit(X, y, "reg", depth=depth, lam=lam, gamma=gamma, min_child_weight=mcw, n_rounds=8, learning_rate=lr, subsample=1.0, seed=0)
    ref = xgb.XGBRegressor(n_estimators=8, max_depth=depth, learning_rate=lr, reg_lambda=lam, gamma=gamma, min_child_weight=mcw, reg_alpha=0,
                            subsample=1.0, tree_method="exact", objective="reg:squarederror", base_score=ens.f0, random_state=0).fit(X, y)
    assert np.max(np.abs(xgm.predict_value(ens, Xt) - ref.predict(Xt))) < 1e-4


def test_classification_matches_xgboost_library_closely_for_few_rounds():
    X, y = _cls(300, 5, 0)
    Xt, _ = _cls(60, 5, 1)
    lam, gamma, mcw, lr, depth = 1.0, 0.0, 1.0, 0.3, 3
    ens = xgm.fit(X, y, "class", depth=depth, lam=lam, gamma=gamma, min_child_weight=mcw, n_rounds=10, learning_rate=lr, subsample=1.0, seed=0)
    p0 = 1.0 / (1.0 + np.exp(-ens.f0))
    ref = xgb.XGBClassifier(n_estimators=10, max_depth=depth, learning_rate=lr, reg_lambda=lam, gamma=gamma, min_child_weight=mcw, reg_alpha=0,
                             subsample=1.0, tree_method="exact", objective="binary:logistic", base_score=p0, random_state=0, eval_metric="logloss").fit(X, y.astype(int))
    assert np.max(np.abs(xgm.predict_value(ens, Xt) - ref.predict_proba(Xt)[:, 1])) < 1e-4


# --- lambda=0, gamma=0 = gradient-boosting-demo (quadratischer Verlust) -----------------------------------------------------------------------------

def test_zero_regularization_matches_gradient_boosting_demo_for_moderate_rounds():
    if not GB_DIR.exists():
        pytest.skip("gradient-boosting-demo nicht neben diesem Repo gefunden")
    sys.path.insert(0, str(GB_DIR))
    import gb_algorithm as gb

    X, y = _reg(300, 5, 0)
    Xt, _ = _reg(60, 5, 1)
    ens_x = xgm.fit(X, y, "reg", depth=2, lam=0.0, gamma=0.0, min_child_weight=0.0, n_rounds=20, learning_rate=0.3, subsample=1.0, seed=0)
    ens_g = gb.fit(X, y, "reg", "squared", depth=2, min_leaf=1, n_rounds=20, learning_rate=0.3, subsample=1.0, seed=0)
    assert np.max(np.abs(xgm.predict_value(ens_x, Xt) - gb.predict_value(ens_g, Xt))) < 1e-9


def test_zero_regularization_eventually_diverges_from_gb_only_via_float_tie_breaks():
    """Nach vielen Runden können winzige, sich unabhängig aufbauende Fließkomma-Unterschiede zwischen den zwei verschieden geschriebenen (aber mathematisch identischen) Gain-Formeln einen Gleichstand
    unterschiedlich auflösen - dokumentiert hier als bewusst NICHT als exakter Test, nur als Plausibilitätsprüfung (die Vorhersagen bleiben nahe beieinander, nicht mehr exakt gleich)."""
    if not GB_DIR.exists():
        pytest.skip("gradient-boosting-demo nicht neben diesem Repo gefunden")
    sys.path.insert(0, str(GB_DIR))
    import gb_algorithm as gb

    X, y = _reg(300, 5, 0)
    Xt, _ = _reg(60, 5, 1)
    ens_x = xgm.fit(X, y, "reg", depth=2, lam=0.0, gamma=0.0, min_child_weight=0.0, n_rounds=60, learning_rate=0.3, subsample=1.0, seed=0)
    ens_g = gb.fit(X, y, "reg", "squared", depth=2, min_leaf=1, n_rounds=60, learning_rate=0.3, subsample=1.0, seed=0)
    diff = np.abs(xgm.predict_value(ens_x, Xt) - gb.predict_value(ens_g, Xt))
    assert diff.mean() < 1.0                                              # nah beieinander, aber kein exakter Test mehr über so viele Runden


# --- Grenzfälle --------------------------------------------------------------------------------------------------------------------------------------

def test_min_child_weight_is_respected():
    X, y = _reg(200, 4, 1)
    ens = xgm.fit(X, y, "reg", depth=4, lam=1.0, gamma=0.0, min_child_weight=15.0, n_rounds=5, learning_rate=0.2, subsample=1.0, seed=0)
    for tree in ens.trees:
        leaves = tree.feature < 0
        assert (tree.h_sum[leaves] >= 15.0 - 1e-9).all()


def test_larger_gamma_prunes_more_aggressively():
    X, y = _reg(300, 5, 2)
    leaves_low = sum(t.n_leaves for t in xgm.fit(X, y, "reg", depth=4, lam=1.0, gamma=0.0, n_rounds=15, learning_rate=0.2, seed=0).trees)
    leaves_high = sum(t.n_leaves for t in xgm.fit(X, y, "reg", depth=4, lam=1.0, gamma=10.0, n_rounds=15, learning_rate=0.2, seed=0).trees)
    assert leaves_high < leaves_low


def test_larger_lambda_shrinks_leaf_weights_towards_zero():
    X, y = _reg(300, 5, 3)
    Xt, _ = _reg(60, 5, 4)
    ens_low = xgm.fit(X, y, "reg", depth=3, lam=0.1, gamma=0.0, n_rounds=1, learning_rate=1.0, seed=0)
    ens_high = xgm.fit(X, y, "reg", depth=3, lam=50.0, gamma=0.0, n_rounds=1, learning_rate=1.0, seed=0)
    v_low = xgm.predict_raw(ens_low, Xt) - ens_low.f0
    v_high = xgm.predict_raw(ens_high, Xt) - ens_high.f0
    assert np.abs(v_high).mean() < np.abs(v_low).mean()


def test_a_single_round_equals_f0_plus_learning_rate_times_the_first_tree():
    X, y = _reg(150, 4, 0)
    ens = xgm.fit(X, y, "reg", depth=2, lam=1.0, gamma=0.0, n_rounds=1, learning_rate=0.5, subsample=1.0, seed=0)
    expected = ens.f0 + 0.5 * T.predict_value(ens.trees[0], X)
    assert np.allclose(xgm.predict_value(ens, X), expected)


def test_subsample_is_reproducible_with_the_same_seed_and_varies_with_a_different_one():
    X, y = _reg(300, 5, 2)
    ens1 = xgm.fit(X, y, "reg", depth=2, lam=1.0, n_rounds=20, learning_rate=0.2, subsample=0.5, seed=7)
    ens2 = xgm.fit(X, y, "reg", depth=2, lam=1.0, n_rounds=20, learning_rate=0.2, subsample=0.5, seed=7)
    ens3 = xgm.fit(X, y, "reg", depth=2, lam=1.0, n_rounds=20, learning_rate=0.2, subsample=0.5, seed=8)
    assert np.array_equal(xgm.predict_value(ens1, X), xgm.predict_value(ens2, X))
    assert not np.array_equal(xgm.predict_value(ens1, X), xgm.predict_value(ens3, X))


def test_generator_matches_cart_demo_conventions():
    ds = S.generate_dataset(500, 3, 0, 7)
    assert ds.X.shape == (500, 11) and ds.names[:2] == ("Distanz", "Ladegewicht")
    Xtr, ytr, Xte, yte = S.split(ds, "class")
    assert len(Xtr) == 350 and len(Xte) == 150
