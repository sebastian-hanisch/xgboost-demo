"""Der XGBoost-Baumkern (Chen & Guestrin 2016, "XGBoost: A Scalable Tree Boosting System"): anders als der Regressionsbaum aus cart-demo/gradient-boosting-demo (Varianz-Kriterium, Blattwert = Mittelwert)
sucht dieser Kern Schnitte über das REGULARISIERTE Ziel direkt - jeder Knoten trägt die Summen G = Summe der Gradienten, H = Summe der Hessematrix-Diagonale (zweite Ableitung) seiner Zeilen, der beste
Schnitt maximiert den Gewinn `0.5*[GL²/(HL+λ) + GR²/(HR+λ) - G²/(H+λ)] - γ` und der Blattwert ist direkt der Newton-Schritt `-G/(H+λ)` - keine nachträgliche Blattwert-Korrektur wie in gradient-boosting-demo
nötig, weil der Gewinn schon der (mit λ regularisierte) Newton-Schritt ist. `γ` bestraft jeden Schnitt einzeln: ein Schnitt wird nur gemacht, wenn der Gewinn (nach Abzug von γ) positiv ist - das ist die
eingebaute Vorwärts-Beschneidung, die cart-demo erst nachträglich über Kosten-Komplexität nachrüsten musste.

Der Baum liegt in parallelen Feldern wie in cart-demo (Breitenreihenfolge, Schwelle = Mitte zwischen zwei benachbarten Werten). Die Schnittsuche ist vektorisiert (sortieren, kumulative Summen von G und H
je Merkmal in einem Zug), aber neu geschrieben (die Gewinnformel unterscheidet sich fundamental von Varianz/Gini/Entropie)."""

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class Tree:
    feature: np.ndarray          # -1 = Blatt
    threshold: np.ndarray
    left: np.ndarray             # -1 bei Blättern
    right: np.ndarray
    value: np.ndarray            # Newton-Schritt -G/(H+lambda), für jeden Knoten (auch innere - "wäre dieser ein Blatt")
    gain: np.ndarray             # Gewinn des Schnitts an diesem Knoten (0 bei Blättern)
    g_sum: np.ndarray
    h_sum: np.ndarray
    n: np.ndarray
    depth: np.ndarray
    lam: float
    gamma: float
    n_total: int
    n_features: int

    @property
    def n_nodes(self):
        return len(self.feature)

    @property
    def n_leaves(self):
        return int((self.feature < 0).sum())

    @property
    def max_depth(self):
        return int(self.depth.max())

    def internal_nodes(self):
        return np.nonzero(self.feature >= 0)[0]


# --- Schnittsuche ------------------------------------------------------------------------------------------------------------------------------------

def gain_matrix(X, grad, hess, lam, gamma, min_child_weight):
    """Gewinn für jede Schwelle jedes Merkmals. Rückgabe: (gain, thr): Matrizen (m-1, d); gain = -inf, wo die Schwelle unzulässig ist (gleiche Werte, zu kleines Hesse-Gewicht in einem Kind)."""
    m, d = X.shape
    order = np.argsort(X, axis=0, kind="stable")
    xs = np.take_along_axis(X, order, axis=0)
    gs = grad[order]
    hs = hess[order]
    Gl = np.cumsum(gs, axis=0)[:-1]
    Hl = np.cumsum(hs, axis=0)[:-1]
    Gtot, Htot = float(gs[:, 0].sum()), float(hs[:, 0].sum())
    Gr, Hr = Gtot - Gl, Htot - Hl
    gain = 0.5 * (Gl ** 2 / (Hl + lam) + Gr ** 2 / (Hr + lam) - Gtot ** 2 / (Htot + lam)) - gamma
    ok = (xs[:-1] < xs[1:]) & (Hl >= min_child_weight) & (Hr >= min_child_weight)
    thr = (xs[:-1] + xs[1:]) / 2.0
    thr = np.where(thr >= xs[1:], xs[:-1], thr)                               # Rundung: die Schwelle bleibt links vom oberen Wert
    return np.where(ok, gain, -np.inf), thr


def best_split(X, grad, hess, lam, gamma, min_child_weight):
    """(Merkmal, Schwelle, Gewinn) des besten Schnitts oder None (kein zulässiger Schnitt oder bester Gewinn <= 0 - Vorwärts-Beschneidung durch gamma). Bei Gleichstand gewinnt das kleinste Merkmal, dann die kleinste Schwelle."""
    m = len(grad)
    if m < 2:
        return None
    gain, thr = gain_matrix(X, grad, hess, lam, gamma, min_child_weight)
    per_feature = gain.argmax(axis=0)
    best_per = gain[per_feature, np.arange(gain.shape[1])]
    f = int(best_per.argmax())
    if not np.isfinite(best_per[f]) or best_per[f] <= 0.0:
        return None
    return f, float(thr[per_feature[f], f]), float(best_per[f])


# --- Wachsen -------------------------------------------------------------------------------------------------------------------------------------------

def grow(X, grad, hess, max_depth=None, lam=1.0, gamma=0.0, min_child_weight=1.0):
    """Wächst den Baum Ebene für Ebene. Ein Knoten wird nicht geteilt, wenn die Tiefe erreicht ist oder kein Schnitt mit positivem Gewinn existiert (gamma-Beschneidung, vorwärts statt nachträglich wie in cart-demo)."""
    X = np.asarray(X, dtype=float)
    grad = np.asarray(grad, dtype=float)
    hess = np.asarray(hess, dtype=float)
    max_depth = 10 ** 6 if max_depth is None else int(max_depth)
    feature, threshold, left, right, value, gain, g_sum, h_sum, size, depth = [], [], [], [], [], [], [], [], [], []

    def new_node(idx, dep):
        G, H = float(grad[idx].sum()), float(hess[idx].sum())
        feature.append(-1), threshold.append(np.nan), left.append(-1), right.append(-1)
        value.append(-G / (H + lam)), gain.append(0.0), g_sum.append(G), h_sum.append(H), size.append(len(idx)), depth.append(dep)
        return len(feature) - 1

    members = {0: np.arange(len(grad))}
    new_node(members[0], 0)
    queue = [0]
    while queue:
        t = queue.pop(0)
        idx = members.pop(t)
        if depth[t] >= max_depth or len(idx) < 2:
            continue
        s = best_split(X[idx], grad[idx], hess[idx], lam, gamma, min_child_weight)
        if s is None:
            continue
        f, thr, g = s
        go_left = X[idx, f] <= thr
        lt, rt = new_node(idx[go_left], depth[t] + 1), new_node(idx[~go_left], depth[t] + 1)
        feature[t], threshold[t], left[t], right[t], gain[t] = f, thr, lt, rt, g
        members[lt], members[rt] = idx[go_left], idx[~go_left]
        queue += [lt, rt]
    return Tree(np.array(feature), np.array(threshold), np.array(left), np.array(right), np.array(value), np.array(gain), np.array(g_sum), np.array(h_sum),
                np.array(size), np.array(depth), lam, gamma, len(grad), X.shape[1])


# --- Anwenden --------------------------------------------------------------------------------------------------------------------------------------

def apply(tree, X):
    """Blatt (Knotennummer) jedes Beispiels, vektorisiert Ebene für Ebene."""
    X = np.asarray(X, dtype=float)
    node = np.zeros(len(X), dtype=int)
    while True:
        idx = np.nonzero(tree.feature[node] >= 0)[0]
        if len(idx) == 0:
            return node
        cur = node[idx]
        go_left = X[idx, tree.feature[cur]] <= tree.threshold[cur]
        node[idx] = np.where(go_left, tree.left[cur], tree.right[cur])


def predict_value(tree, X):
    """Der Newton-Schritt (Blattwert) für jedes Beispiel - wird mit der Lernrate skaliert zur laufenden Vorhersage addiert."""
    return tree.value[apply(tree, X)]


def importances(tree):
    """Wichtigkeit je Merkmal: Summe der Gewinne aller Schnitte dieses Merkmals, auf Summe 1 normiert (0, wenn der Baum nur die Wurzel hat)."""
    imp = np.zeros(tree.n_features)
    for t in tree.internal_nodes():
        imp[tree.feature[t]] += tree.gain[t]
    s = imp.sum()
    return imp / s if s > 0 else imp
