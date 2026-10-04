"""Unabhängige Orakel für den XGBoost-Baumkern (Gain-Formel, Newton-Blattwerte, Boosting-Schleife).

1. Handrechnung an einer Mini-Instanz (Gain und Blattgewichte aus der Formel von Chen & Guestrin 2016, ohne den Code der Demo).
2. Zufällige Instanzen gegen die echte `xgboost`-Bibliothek (`tree_method="exact"`), Vorhersagen auf Trainings- UND neuen Daten. Instanzen, in denen irgendein Knoten
   einen exakten Gain-Gleichstand zwischen zwei Splits hat (bei Log-Loss in der ersten Runde häufig), werden übersprungen: Bibliothek und Demo lösen Gleichstände verschieden auf.
3. Die γ-Skala: die Bibliothek vergleicht den Gain OHNE den Faktor ½ mit γ, die Demo folgt der Formel des Papiers (mit ½). Bei Tiefe 1 gilt deshalb Demo(γ) = Bibliothek(2γ) exakt;
   bei größerer Tiefe beschneidet die Bibliothek zusätzlich erst nachträglich von unten, die Demo vorwärts - dort ist ein Vergleich mit γ > 0 nicht sinnvoll.
"""

import numpy as np
import pytest

import xgb_algorithm as xgm
import xgb_tree as T

xgb = pytest.importorskip("xgboost")


def test_gain_and_leaf_weights_by_hand():
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    g = np.array([-1.0, -1.0, 1.0, 1.0])
    h = np.ones(4)
    lam, gamma = 1.0, 0.1
    f, thr, gain = T.best_split(x, g, h, lam, gamma, 1.0)
    # beste Teilung zwischen 2 und 3: GL = -2, HL = 2, GR = 2, HR = 2, G = 0, H = 4  ->  0.5 * (4/3 + 4/3 - 0) - 0.1
    assert f == 0 and thr == pytest.approx(2.5) and gain == pytest.approx(4.0 / 3.0 - 0.1)
    tree = T.grow(x, g, h, 1, lam, gamma, 1.0)
    leaves = tree.feature < 0
    assert sorted(np.round(tree.value[leaves], 12)) == pytest.approx([-2.0 / 3.0, 2.0 / 3.0])    # -G/(H+lambda): +2/3 links (G = -2), -2/3 rechts


def _instance(rng, it):
    task = "reg" if it % 2 == 0 else "class"
    n, d = int(rng.integers(10, 120)), int(rng.integers(1, 5))
    X = rng.normal(size=(n, d))
    if it % 7 == 0:
        X = np.round(X, 1)                                                # Gleichstände in den Merkmalswerten
    if task == "reg":
        y = 2.0 * X[:, 0] + np.sin(X[:, -1]) + rng.normal(size=n)
    else:
        y = (X[:, 0] + rng.normal(0, 0.7, n) > 0).astype(float)
        if y.min() == y.max():
            y[0] = 1.0 - y[0]
    return task, X, y


def _library(task, X, y, f0, depth, lam, gamma_lib, mcw, lr, rounds):
    kw = dict(n_estimators=rounds, max_depth=depth, learning_rate=lr, reg_lambda=lam, gamma=gamma_lib, min_child_weight=mcw, reg_alpha=0.0, tree_method="exact", random_state=0, n_jobs=1)
    if task == "reg":
        m = xgb.XGBRegressor(objective="reg:squarederror", base_score=f0, **kw).fit(X, y)
        return lambda Z: m.predict(Z)
    m = xgb.XGBClassifier(objective="binary:logistic", base_score=1.0 / (1.0 + np.exp(-f0)), eval_metric="logloss", **kw).fit(X, y.astype(int))
    return lambda Z: m.predict_proba(Z)[:, 1]


def test_random_instances_match_xgboost_library(monkeypatch):
    ties = []
    original = T.best_split

    def spy(X, grad, hess, lam, gamma, mcw):
        if len(grad) >= 2:
            gm, _ = T.gain_matrix(X, grad, hess, lam, gamma, mcw)
            v = np.sort(gm[np.isfinite(gm)])[::-1]
            if len(v) >= 2 and v[0] > 0 and abs(v[0] - v[1]) <= 1e-9 * max(1.0, abs(v[0])):
                ties.append(1)
        return original(X, grad, hess, lam, gamma, mcw)

    monkeypatch.setattr(T, "best_split", spy)
    rng = np.random.default_rng(777)
    compared = 0
    for it in range(160):
        task, X, y = _instance(rng, it)
        depth = int(rng.integers(1, 5))
        gamma = float(rng.choice([0.0, 0.05, 0.3, 1.0])) if depth == 1 else 0.0
        lam, mcw, lr, rounds = float(rng.choice([0.1, 0.3, 1.0, 5.0])), float(rng.choice([0.2, 0.5, 1.0, 3.0])), float(rng.choice([0.1, 0.3, 1.0])), int(rng.integers(1, 5))
        ties.clear()
        ens = xgm.fit(X, y, task, depth=depth, lam=lam, gamma=gamma, min_child_weight=mcw, n_rounds=rounds, learning_rate=lr, subsample=1.0, seed=0)
        if ties:
            continue
        ref = _library(task, X, y, ens.f0, depth, lam, 2.0 * gamma, mcw, lr, rounds)               # Bibliotheks-gamma = 2 * gamma der Demo (kein Faktor 1/2 im Gain)
        Xt = rng.normal(size=(40, X.shape[1]))
        assert np.max(np.abs(ref(X) - xgm.predict_value(ens, X))) < 1e-4, (it, task, depth, lam, gamma, mcw)
        assert np.max(np.abs(ref(Xt) - xgm.predict_value(ens, Xt))) < 1e-4, (it, task, depth, lam, gamma, mcw)
        compared += 1
    assert compared >= 80                                                  # genug tie-freie Instanzen blieben übrig


def test_gamma_scale_difference_to_library():
    """Dokumentiert die γ-Skala: ein Split mit Gain (mit ½) knapp unter γ wird von der Demo verworfen, von der Bibliothek erst, wenn γ_Bib > 2 * Gain."""
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(40):
        n = int(rng.integers(8, 40))
        X = rng.normal(size=(n, 2))
        y = X[:, 0] + rng.normal(size=n)
        g, h = y.mean() - y, np.ones(n)
        s = T.best_split(X, g, h, 1.0, 0.0, 1.0)
        if s is None:
            continue
        half_gain = s[2]
        for gamma_demo, expect_split in ((half_gain * 0.99, True), (half_gain * 1.01, False)):
            demo_split = T.grow(X, g, h, 1, 1.0, gamma_demo, 1.0).n_leaves == 2
            m = xgb.XGBRegressor(n_estimators=1, max_depth=1, learning_rate=1.0, reg_lambda=1.0, gamma=2.0 * gamma_demo, min_child_weight=1.0, tree_method="exact",
                                 base_score=float(y.mean()), reg_alpha=0.0).fit(X, y)
            lib_split = len(m.get_booster().trees_to_dataframe()) > 1
            assert demo_split == lib_split == expect_split
        checked += 1
    assert checked >= 20
