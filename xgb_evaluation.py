"""Messungen an XGBoost: Konvergenz (Runden bis zu einer Testfehlerschwelle) gegen gradient-boosting-demo bei den jeweils typischen Einstellungen, Wirkung von lambda und gamma auf Überanpassung und
Blätterzahl."""

from dataclasses import dataclass

import numpy as np

import xgb_algorithm as xgm
import xgb_constants as C
import xgb_scenario as S
import xgb_tree as T


def baseline_error(ds, task):
    _, ytr, _, yte = S.split(ds, task)
    if task == "class":
        return float(np.mean(yte != int(ytr.mean() > 0.5)))
    return float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))


def _metrics(ensemble, X, y, task):
    if task == "class":
        return {"error": float(np.mean(xgm.predict(ensemble, X) != y))}
    v = xgm.predict_value(ensemble, X)
    return {"rmse": float(np.sqrt(np.mean((v - y) ** 2))), "mae": float(np.mean(np.abs(v - y)))}


def primary(metrics, task):
    return metrics["error"] if task == "class" else metrics["rmse"]


def ensemble_importances(ensemble):
    imps = [T.importances(t) for t in ensemble.trees]
    return np.mean(imps, axis=0) if imps else np.zeros(0)


@dataclass
class Analysis:
    ds: object
    task: str
    depth: int
    lam: float
    gamma: float
    min_child_weight: float
    n_rounds: int
    lr: float
    subsample: float
    ensemble: object
    train: dict
    test: dict
    baseline: float
    verdict: str
    n_leaves: int
    imp: np.ndarray


def analyse(task, depth, lam, gamma, min_child_weight, n_rounds, lr, subsample, n, n_noise, label_noise, seed):
    ds = S.generate_dataset(n, n_noise, label_noise if task == "class" else 0, seed)
    Xtr, ytr, Xte, yte = S.split(ds, task)
    ensemble = xgm.fit(Xtr, ytr.astype(float), task, depth=depth, lam=lam, gamma=gamma, min_child_weight=min_child_weight,
                        n_rounds=n_rounds, learning_rate=lr, subsample=subsample / 100.0, seed=0)
    train = _metrics(ensemble, Xtr, ytr, task)
    test = _metrics(ensemble, Xte, yte, task)
    baseline = baseline_error(ds, task)
    a = Analysis(ds, task, depth, lam, gamma, min_child_weight, n_rounds, lr, subsample, ensemble, train, test, baseline, "",
                 xgm.n_leaves_total(ensemble), ensemble_importances(ensemble))
    a.verdict = verdict(a)
    return a


def verdict(a):
    tr, te = primary(a.train, a.task), primary(a.test, a.task)
    if a.n_rounds <= 1:
        return "stump"
    over = (te - tr > C.OVERFIT_GAP_CLASS) if a.task == "class" else (te > C.OVERFIT_RATIO_REG * max(tr, 1e-9))
    if over:
        return "overfit"
    return "underfit" if te > C.UNDERFIT_SHARE * a.baseline else "ok"


# --- Testfehler gegen die Rundenzahl ----------------------------------------------------------------------------------------------------------------

def round_rows(a, ks=None):
    ks = ks or sorted(set(np.unique(np.round(np.geomspace(1, a.n_rounds, min(24, a.n_rounds))).astype(int))))
    Xtr, ytr, Xte, yte = S.split(a.ds, a.task)
    rows = []
    for k in ks:
        tr = primary(_metrics_upto(a.ensemble, Xtr, ytr, a.task, k), a.task)
        te = primary(_metrics_upto(a.ensemble, Xte, yte, a.task, k), a.task)
        rows.append({"k": int(k), "train": tr, "test": te})
    return rows


def _metrics_upto(ensemble, X, y, task, upto):
    if task == "class":
        return {"error": float(np.mean(xgm.predict(ensemble, X, upto=upto) != y))}
    v = xgm.predict_value(ensemble, X, upto=upto)
    return {"rmse": float(np.sqrt(np.mean((v - y) ** 2))), "mae": float(np.mean(np.abs(v - y)))}


def best_round(rows):
    return min(rows, key=lambda r: (r["test"], r["k"]))


# --- Konvergenz gegen gradient-boosting-demo (jeweils typische Einstellungen) -----------------------------------------------------------------------

ROUNDS_GRID = (5, 10, 15, 20, 30, 40, 60, 80, 120)


def convergence_rows(depth, lam, xgb_lr, gb_lr, n, n_noise, label_noise, gb_fit=None, gb_predict=None, seeds=C.SWEEP_SEEDS, grid=ROUNDS_GRID):
    """Testfehler von XGBoost (eigene Standard-Lernrate) gegen gradient-boosting-demo (dessen eigene Standard-Lernrate) je Rundenzahl, gemittelt über mehrere Datensätze.
    `gb_fit`/`gb_predict` werden von der App übergeben (Import von gradient-boosting-demo, optional - siehe app.py)."""
    rows = []
    for n_rounds in grid:
        xgb_errs, gb_errs = [], []
        for sd in seeds:
            ds = S.generate_dataset(n, n_noise, label_noise, sd)
            Xtr, ytr, Xte, yte = S.split(ds, "class")
            ens = xgm.fit(Xtr, ytr.astype(float), "class", depth=depth, lam=lam, gamma=0.0, min_child_weight=1.0, n_rounds=n_rounds, learning_rate=xgb_lr, subsample=1.0, seed=0)
            xgb_errs.append(float(np.mean(xgm.predict(ens, Xte) != yte)))
            if gb_fit is not None:
                gens = gb_fit(Xtr, ytr.astype(float), "class", "logloss", depth=depth, min_leaf=5, n_rounds=n_rounds, learning_rate=gb_lr, subsample=1.0, seed=0)
                gb_errs.append(float(np.mean(gb_predict(gens, Xte) != yte)))
        row = {"n_rounds": n_rounds, "xgb": float(np.mean(xgb_errs))}
        if gb_errs:
            row["gb"] = float(np.mean(gb_errs))
        rows.append(row)
    return rows


def rounds_to_threshold(rows, key, threshold):
    """Die kleinste Rundenzahl, bei der der Testfehler (Spalte `key`) unter `threshold` fällt (None, wenn nie erreicht)."""
    for r in rows:
        if key in r and r[key] < threshold:
            return r["n_rounds"]
    return None


# --- Wirkung von lambda und gamma auf Überanpassung und Blätterzahl ----------------------------------------------------------------------------------

LAM_GRID = (0.0, 0.5, 1.0, 3.0, 10.0, 20.0)
GAMMA_GRID = (0.0, 0.5, 1.0, 3.0, 10.0, 20.0)


def lambda_rows(depth, gamma, n_rounds, lr, n, n_noise, label_noise, seeds=C.SWEEP_SEEDS, grid=LAM_GRID):
    rows = []
    for lam in grid:
        tr_errs, te_errs, leaves = [], [], []
        for sd in seeds:
            ds = S.generate_dataset(n, n_noise, label_noise, sd)
            Xtr, ytr, Xte, yte = S.split(ds, "class")
            ens = xgm.fit(Xtr, ytr.astype(float), "class", depth=depth, lam=lam, gamma=gamma, min_child_weight=1.0, n_rounds=n_rounds, learning_rate=lr, subsample=1.0, seed=0)
            tr_errs.append(float(np.mean(xgm.predict(ens, Xtr) != ytr)))
            te_errs.append(float(np.mean(xgm.predict(ens, Xte) != yte)))
            leaves.append(xgm.n_leaves_total(ens))
        rows.append({"lam": lam, "train": float(np.mean(tr_errs)), "test": float(np.mean(te_errs)), "leaves": float(np.mean(leaves))})
    return rows


def gamma_rows(depth, lam, n_rounds, lr, n, n_noise, label_noise, seeds=C.SWEEP_SEEDS, grid=GAMMA_GRID):
    rows = []
    for gamma in grid:
        tr_errs, te_errs, leaves = [], [], []
        for sd in seeds:
            ds = S.generate_dataset(n, n_noise, label_noise, sd)
            Xtr, ytr, Xte, yte = S.split(ds, "class")
            ens = xgm.fit(Xtr, ytr.astype(float), "class", depth=depth, lam=lam, gamma=gamma, min_child_weight=1.0, n_rounds=n_rounds, learning_rate=lr, subsample=1.0, seed=0)
            tr_errs.append(float(np.mean(xgm.predict(ens, Xtr) != ytr)))
            te_errs.append(float(np.mean(xgm.predict(ens, Xte) != yte)))
            leaves.append(xgm.n_leaves_total(ens))
        rows.append({"gamma": gamma, "train": float(np.mean(tr_errs)), "test": float(np.mean(te_errs)), "leaves": float(np.mean(leaves))})
    return rows
