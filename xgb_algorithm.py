"""XGBoost (Chen & Guestrin 2016): dieselbe additive Boosting-Idee wie gradient-boosting-demo (F_m = F_{m-1} + Lernrate * Baum_m), aber jede Runde optimiert direkt das REGULARISIERTE Ziel
`Σ l(y_i, F_{m-1}(x_i) + Baum_m(x_i)) + Ω(Baum_m)` mit `Ω(T) = γ*Blätter + 0.5*λ*Σ Blattwerte²` über eine Newton-Näherung zweiter Ordnung (Gradient UND Hessematrix, nicht nur der Gradient wie in
gradient-boosting-demo). Der Baumkern (`xgb_tree.py`) sucht Splits direkt über diese Formel - der Blattwert ist der Newton-Schritt `-G/(H+λ)`, keine nachträgliche Korrektur nötig."""

from dataclasses import dataclass

import numpy as np

import xgb_tree as T

EPS = 1e-12


@dataclass(frozen=True)
class Ensemble:
    trees: tuple
    f0: float
    learning_rate: float
    task: str
    lam: float
    gamma: float
    n_train: int


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


# --- Gradient und Hessematrix je Aufgabe (fest: quadratisch bei Regression, Log-Loss bei Klassifikation - siehe README) --------------------------------

def grad_hess(y, F, task):
    """(g, h) = erste und zweite Ableitung des Verlusts nach F, zeilenweise - der Baum wächst direkt auf diesen beiden Summen, nicht auf einem Pseudo-Residuum wie in gradient-boosting-demo."""
    if task == "reg":
        return F - y, np.ones_like(F)
    p = _sigmoid(F)
    return p - y, p * (1.0 - p)


def loss_value(y, F, task):
    if task == "reg":
        return 0.5 * (y - F) ** 2
    p = np.clip(_sigmoid(F), EPS, 1.0 - EPS)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def init_value(y, task):
    """Der beste konstante Start F0 (nicht durch lambda regularisiert - lambda wirkt nur auf Blattgewichte, wie im echten XGBoost)."""
    if task == "reg":
        return float(np.mean(y))
    p = np.clip(float(np.mean(y)), 1e-6, 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


# --- Fit / Vorhersage -------------------------------------------------------------------------------------------------------------------------------

def fit(X, y, task, depth=3, lam=1.0, gamma=0.0, min_child_weight=1.0, n_rounds=60, learning_rate=0.3, subsample=1.0, seed=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    f0 = init_value(y, task)
    F = np.full(n, f0)
    rng = np.random.default_rng(seed)
    trees = []
    for _ in range(n_rounds):
        if subsample < 1.0:
            k = max(2, int(round(subsample * n)))
            idx = np.sort(rng.choice(n, k, replace=False))
        else:
            idx = np.arange(n)
        g, h = grad_hess(y[idx], F[idx], task)
        tree = T.grow(X[idx], g, h, depth, lam, gamma, min_child_weight)
        F = F + learning_rate * T.predict_value(tree, X)
        trees.append(tree)
    return Ensemble(tuple(trees), f0, learning_rate, task, lam, gamma, n)


def predict_raw(ensemble, X, upto=None):
    trees = ensemble.trees[:upto] if upto else ensemble.trees
    F = np.full(len(X), ensemble.f0)
    for t in trees:
        F = F + ensemble.learning_rate * T.predict_value(t, X)
    return F


def predict_value(ensemble, X, upto=None):
    F = predict_raw(ensemble, X, upto)
    return F if ensemble.task == "reg" else _sigmoid(F)


def predict(ensemble, X, upto=None):
    v = predict_value(ensemble, X, upto)
    return (v > 0.5).astype(int) if ensemble.task == "class" else v


def n_leaves_total(ensemble):
    return int(sum(t.n_leaves for t in ensemble.trees))
