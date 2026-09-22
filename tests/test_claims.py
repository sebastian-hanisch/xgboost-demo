"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-22, Toleranzen fangen Rundung ab). `analyse()` und die Experiment-Funktionen sind deterministisch (kein Zufall außer im
Datenerzeuger und der - festen, mit `seed` reproduzierbaren - Teilstichprobe)."""

import functools
import sys
from pathlib import Path

import numpy as np
import pytest

import xgb_algorithm as xgm
import xgb_constants as C
import xgb_evaluation as ev
import xgb_scenario as S

GB_DIR = Path(__file__).resolve().parents[2] / "gradient-boosting-demo"

PRESET = {"standard": "🌳 Standard", "stump": "🪓 Ein Schritt (kein Boosting)", "nogamma": "🌲 Ohne Bremse (gamma = 0)", "gamma": "✂️ Mit Gamma-Beschneidung", "reg": "📈 Regression Standard"}


@functools.lru_cache(maxsize=None)
def _preset(key):
    p = C.PRESETS[PRESET[key]]
    return ev.analyse(p["task"], p["depth"], p["lam"], p["gamma"], p["min_child_weight"], p["n_rounds"], p["lr"], p["subsample"], p["n"], p["n_noise"], p["label_noise"], p["seed"])


def _help(key, *needles):
    text = C.PRESET_HELP[PRESET[key]]
    for n in needles:
        assert n in text, (key, n)


@functools.lru_cache(maxsize=None)
def _default():
    return ev.analyse("class", C.DEFAULT_DEPTH, C.DEFAULT_LAM, C.DEFAULT_GAMMA, C.DEFAULT_MIN_CHILD_WEIGHT, C.DEFAULT_N_ROUNDS, C.DEFAULT_LR, C.DEFAULT_SUBSAMPLE,
                      C.DEFAULT_N, C.DEFAULT_NOISE, 0, C.DEFAULT_SEED)


# --- Preset-Hilfen --------------------------------------------------------------------------------------------------------------------------------

def test_standard_preset():
    a = _preset("standard")
    assert (a.train["error"], a.test["error"], a.n_leaves) == pytest.approx((0.019, 0.1556, 453), abs=0.0015)
    _help("standard", "1.9 %", "15.6 %", "453")


def test_single_step_preset_is_no_better_than_a_shallow_stump():
    a = _preset("stump")
    assert len(a.ensemble.trees) == 1 and a.verdict == "stump"
    assert (a.test["error"], a.n_leaves) == pytest.approx((0.2444, 2), abs=0.0015)
    _help("stump", "24.4 %", "2 Blätter")


def test_no_gamma_grows_a_huge_ensemble_without_beating_the_pruned_one():
    a_no = _preset("nogamma")
    a_yes = _preset("gamma")
    assert a_no.gamma == 0.0 and a_yes.gamma == 0.5
    assert (a_no.train["error"], a_no.test["error"], a_no.n_leaves) == pytest.approx((0.0, 0.1667, 1963), abs=0.0015)
    assert (a_yes.train["error"], a_yes.test["error"], a_yes.n_leaves) == pytest.approx((0.0202, 0.1472, 492), abs=0.0015)
    assert a_yes.n_leaves < a_no.n_leaves * 0.3                                                 # deutlich kleiner ...
    assert a_yes.test["error"] < a_no.test["error"]                                             # ... und trotzdem (hier: gerade deshalb) genauer
    _help("nogamma", "1963", "16.7 %")
    _help("gamma", "492", "75 %", "2.0 %", "14.7 %", "16.7 %")


def test_regression_standard_preset():
    a = _preset("reg")
    assert a.task == "reg"
    assert (a.test["rmse"], a.n_leaves) == pytest.approx((9.044, 464), abs=0.02)
    _help("reg", "9.0", "464")


def test_every_preset_is_a_valid_setting():
    for name, p in C.PRESETS.items():
        assert p["task"] in C.TASKS
        assert C.DEPTH_MIN <= p["depth"] <= C.DEPTH_MAX and C.N_ROUNDS_MIN <= p["n_rounds"] <= C.N_ROUNDS_MAX
        assert C.LR_MIN <= p["lr"] <= C.LR_MAX and C.LAM_MIN <= p["lam"] <= C.LAM_MAX and C.GAMMA_MIN <= p["gamma"] <= C.GAMMA_MAX
        assert C.MIN_CHILD_WEIGHT_MIN <= p["min_child_weight"] <= C.MIN_CHILD_WEIGHT_MAX and C.SUBSAMPLE_MIN <= p["subsample"] <= C.SUBSAMPLE_MAX
        d = C.N_BASE + p["n_noise"]
        assert C.N_MIN <= p["n"] <= C.N_MAX and 0 <= p["fx"] < d and 0 <= p["fy"] < d and name in C.PRESET_HELP


# --- Standardansicht --------------------------------------------------------------------------------------------------------------------------------

def test_default_view_numbers():
    a = _default()
    assert (a.train["error"], a.test["error"], a.baseline) == pytest.approx((0.019, 0.1556, 0.4639), abs=0.0015)
    assert a.verdict in ("ok", "overfit")                                                        # je nach Ausgangslage - siehe round_curve für das eigentliche Bild


def test_round_curve_reaches_a_minimum():
    a = _default()
    rows = ev.round_rows(a)
    best = ev.best_round(rows)
    assert best["k"] <= a.n_rounds and best["test"] <= rows[-1]["test"] + 1e-9


# --- Konvergenz gegen gradient-boosting-demo -------------------------------------------------------------------------------------------------------

def test_convergence_experiment_numbers():
    if not GB_DIR.exists():
        pytest.skip("gradient-boosting-demo nicht neben diesem Repo gefunden")
    sys.path.insert(0, str(GB_DIR))
    import gb_algorithm as gb

    rows = ev.convergence_rows(2, 1.0, 0.3, 0.1, C.DEFAULT_N, C.DEFAULT_NOISE, 0, gb_fit=gb.fit, gb_predict=gb.predict)
    x15 = ev.rounds_to_threshold(rows, "xgb", 0.15)
    g15 = ev.rounds_to_threshold(rows, "gb", 0.15)
    assert x15 == 15
    assert g15 == 30                                                                              # GB braucht doppelt so viele Runden für denselben Testfehler
    vals = {r["n_rounds"]: (r["xgb"], r["gb"]) for r in rows}
    assert vals[15] == pytest.approx((0.1478, 0.1667), abs=0.002)
    assert vals[30] == pytest.approx((0.1411, 0.1494), abs=0.002)


# --- Wirkung von lambda --------------------------------------------------------------------------------------------------------------------------------

def test_lambda_mainly_shrinks_weights_not_tree_size():
    rows = ev.lambda_rows(4, 0.0, 150, 0.3, C.DEFAULT_N, C.DEFAULT_NOISE, 0)
    leaves = [r["leaves"] for r in rows]
    assert leaves == pytest.approx([1316.8, 1444.6, 1535.6, 1716.0, 1918.8, 2026.2], abs=5)
    assert leaves[-1] > leaves[0]                                                                  # Blätterzahl WÄCHST mit lambda statt zu schrumpfen
    test_vals = [r["test"] for r in rows]
    assert max(test_vals[:-1]) - min(test_vals[:-1]) < 0.006                                       # bis auf den letzten (sehr großen) Wert kaum Bewegung
    assert test_vals[-1] < test_vals[0]                                                             # nur bei sehr großem lambda ein sichtbarer Effekt


# --- Wirkung von gamma ---------------------------------------------------------------------------------------------------------------------------------

def test_gamma_directly_controls_tree_size_and_overfitting_gap():
    rows = ev.gamma_rows(4, 1.0, 150, 0.3, C.DEFAULT_N, C.DEFAULT_NOISE, 0)
    leaves = [r["leaves"] for r in rows]
    gaps = [r["test"] - r["train"] for r in rows]
    assert leaves == pytest.approx([1535.6, 417.8, 313.4, 211.4, 165.4, 157.2], abs=5)
    assert all(leaves[i] > leaves[i + 1] for i in range(len(leaves) - 1))                          # streng monoton fallend
    assert all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1))                                # Lücke schrumpft streng monoton
    best = min(rows, key=lambda r: r["test"])
    assert best["gamma"] in (0.5, 1.0)                                                              # bestes gamma liegt in der Mitte, nicht an den Rändern
    assert rows[-1]["test"] > best["test"]                                                          # zu viel gamma unterpasst wieder


# --- Grenzfälle --------------------------------------------------------------------------------------------------------------------------------------

def test_zero_lambda_zero_gamma_matches_gradient_boosting_demo_squared_loss():
    if not GB_DIR.exists():
        pytest.skip("gradient-boosting-demo nicht neben diesem Repo gefunden")
    sys.path.insert(0, str(GB_DIR))
    import gb_algorithm as gb

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(0, 0.4, 300)
    Xt = np.random.default_rng(1).normal(size=(60, 5))
    ens_x = xgm.fit(X, y, "reg", depth=2, lam=0.0, gamma=0.0, min_child_weight=0.0, n_rounds=20, learning_rate=0.3, subsample=1.0, seed=0)
    ens_g = gb.fit(X, y, "reg", "squared", depth=2, min_leaf=1, n_rounds=20, learning_rate=0.3, subsample=1.0, seed=0)
    assert np.max(np.abs(xgm.predict_value(ens_x, Xt) - gb.predict_value(ens_g, Xt))) < 1e-9


def test_generator_matches_cart_demo_conventions():
    ds = S.generate_dataset(500, 3, 0, 7)
    assert ds.X.shape == (500, 11) and ds.names[:2] == ("Distanz", "Ladegewicht")
    Xtr, ytr, Xte, yte = S.split(ds, "class")
    assert len(Xtr) == 350 and len(Xte) == 150
