"""XGBoost - regularisiertes Ziel + Newton-Schritt - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - XGBoost - und lässt stattdessen das Beispiel wachsen.
Siebtes Stück der Baumbasierten Linie der "Konzepte"-Reihe, drittes Stück des Boosting-Asts (nach AdaBoost, Gradient Boosting): Gradient Boosting nutzt nur den Gradienten (erste Ableitung) und
korrigiert Blattwerte nachträglich - XGBoost nutzt zusätzlich die Hesse-Matrix (zweite Ableitung) direkt in der Split-Suche und regularisiert das Ziel selbst (lambda, gamma) statt nur zu schrumpfen.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

import xgb_algorithm as xgm
import xgb_constants as C
import xgb_evaluation as ev
from xgb_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from xgb_visualization import (
    build_convergence_chart,
    build_importance,
    build_leaves_chart,
    build_map,
    build_reg_chart,
    build_round_curve,
    build_round_tree,
    feature_label,
)

st.set_page_config(page_title="XGBoost – Sebastian Hanisch", layout="wide")

GB_DIR = Path(__file__).resolve().parent.parent / "gradient-boosting-demo"
_gb = None
if GB_DIR.exists():
    sys.path.insert(0, str(GB_DIR))
    import gb_algorithm as _gb

VERDICT_TEXT = {
    "stump": "ℹ️ **Nur ein Schritt eingestellt** - das ist die Vorhersage eines einzelnen (mit Lernrate skalierten) Baums, noch kein Boosting.",
    "overfit": "⚠️ **Überanpassung:** der Testfehler liegt deutlich über dem Trainingsfehler - mehr γ (Mindest-Gain je Split) probieren.",
    "underfit": "⚠️ **Unteranpassung:** kaum besser als Raten - mehr Runden, größere Lernrate oder tiefere Bäume könnten helfen.",
    "ok": "✅ **Sieht vernünftig aus:** Training und Test liegen nicht weit auseinander.",
}


def _err(task, x):
    return f"{x:.1%}" if task == "class" else f"{x:.1f} min"


@st.cache_resource(show_spinner=False, max_entries=24)
def _analysis(*params):
    return ev.analyse(*params)


@st.cache_data(show_spinner=False, max_entries=8)
def _round_rows(*params):
    a = _analysis(*params)
    return ev.round_rows(a)


@st.cache_data(show_spinner=False, max_entries=4)
def _convergence(depth, lam, xgb_lr, gb_lr, n, n_noise, label_noise):
    return ev.convergence_rows(depth, lam, xgb_lr, gb_lr, n, n_noise, label_noise, gb_fit=_gb.fit if _gb else None, gb_predict=_gb.predict if _gb else None)


@st.cache_data(show_spinner=False, max_entries=6)
def _lambda_rows(depth, gamma, n_rounds, lr, n, n_noise, label_noise):
    return ev.lambda_rows(depth, gamma, n_rounds, lr, n, n_noise, label_noise)


@st.cache_data(show_spinner=False, max_entries=6)
def _gamma_rows(depth, lam, n_rounds, lr, n, n_noise, label_noise):
    return ev.gamma_rows(depth, lam, n_rounds, lr, n, n_noise, label_noise)


st.title("🚀🌳 XGBoost – regularisiertes Ziel und Newton-Schritt")
st.markdown(
    """
**Gradient Boosting** (voriges Stück) wächst jede Runde einen gewöhnlichen Regressionsbaum auf den Pseudo-Residuen (nur der Gradient, die erste Ableitung) und trägt danach den verlustoptimalen
Blattwert nach. **XGBoost** (Chen & Guestrin 2016) geht einen Schritt weiter: die Split-Suche selbst optimiert direkt das **regularisierte** Ziel `Σ l(y, F+Baum) + γ·Blätter + 0.5·λ·Σ Blattgewichte²`
über eine Newton-Näherung **zweiter** Ordnung (Gradient **und** Hesse-Matrix). Der Blattwert ist der Newton-Schritt `-G/(H+λ)`, der Gain eines Splits `0.5·[GL²/(HL+λ) + GR²/(HR+λ) - G²/(H+λ)] - γ` -
ein Split lohnt sich nur, wenn dieser Gain positiv ist (**Vorwärts-Beschneidung** durch γ, statt wie in cart-demo nachträglich über Kosten-Komplexität).
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - siebtes Stück der Baumbasierten Linie der \"Konzepte\"-Reihe und drittes Stück des "
    "**Boosting-Asts** (nach AdaBoost, Gradient Boosting) - **ein** Verfahren an einem wachsenden Beispiel. Das Verfahren geht auf Chen und Guestrin (2016) zurück; alle Lieferungen, Merkmale und Zahlen "
    "dieser Demo sind erzeugt und gemessen - keine echten Daten. Der Baumkern (`xgb_tree.py`) ist neu geschrieben (die Gain-Formel unterscheidet sich fundamental vom Varianz-Kriterium aus cart-demo/"
    "gradient-boosting-demo); die echte `xgboost`-Bibliothek kommt nur in den Tests als Gegenprobe vor."
)
st.caption(
    "**Bezug zu OR:** ein kleineres, robusteres Ensemble (weniger Blätter durch γ, gedämpfte Gewichte durch λ) liefert stabilere Lieferzeit-Vorhersagen für die Tourenplanung - Überanpassung an einzelne "
    "Trainingslieferungen wirkt sich sonst direkt auf die Zuverlässigkeit der geplanten Zeitfenster aus."
)

with st.expander("So funktioniert XGBoost", expanded=True):
    st.markdown(
        r"""
1. **Start:** die beste konstante Vorhersage $F_0$ (Mittelwert für Regression, Log-Odds der Basisrate für Klassifikation) - nicht durch λ regularisiert, wie im echten XGBoost.
2. **Jede Runde $m$:** Gradient $g_i$ **und** Hesse-Diagonale $h_i$ der Verlustfunktion an $F_{m-1}(x_i)$ ausrechnen (quadratischer Verlust bei Regression, Log-Loss bei Klassifikation).
3. **Split-Suche:** für jeden Knoten $G=\sum g_i$, $H=\sum h_i$; ein Split in zwei Kinder mit Summen $G_L,H_L,G_R,H_R$ lohnt sich, wenn $\tfrac12\big[\tfrac{G_L^2}{H_L+\lambda}+\tfrac{G_R^2}{H_R+\lambda}-\tfrac{G^2}{H+\lambda}\big]-\gamma>0$ ist - sonst bleibt der Knoten ein Blatt.
4. **Blattwert:** direkt der Newton-Schritt $w^*=-\tfrac{G}{H+\lambda}$ - keine nachträgliche Korrektur wie in gradient-boosting-demo nötig.
5. **Update:** $F_m(x)=F_{m-1}(x)+\eta\cdot\text{Baum}_m(x)$, optional auf einer Teilstichprobe der Zeilen.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    task = st.selectbox("Aufgabe", C.TASKS, key="task_select", format_func=lambda k: C.TASK_LABELS[k],
                        help="Klassifikation: kommt die Lieferung zu spät? Regression: wie lange dauert sie? Verlust fest (Log-Loss bzw. quadratisch), nur die Merkmale/Bäume sind gleich.")
    depth = st.slider("Tiefe der Bäume", *bounds("depth_slider"), key="depth_slider")
    n_rounds = st.slider("Zahl der Runden", *bounds("n_rounds_slider"), key="n_rounds_slider")
    lr = st.slider("Lernrate", *bounds("lr_slider"), key="lr_slider", step=0.01, format="%.2f")
    lam = st.slider("λ (L2 auf Blattgewichte)", *bounds("lam_slider"), key="lam_slider", step=0.1, format="%.1f",
                    help="Dämpft die Blattgewichte -G/(H+λ) - größer heißt vorsichtigere Schritte, ändert aber kaum, WIE GROSS die Bäume werden (siehe Experiment).")
    gamma = st.slider("γ (Mindest-Gain je Split)", *bounds("gamma_slider"), key="gamma_slider", step=0.1, format="%.1f",
                      help="Ein Split wird nur gemacht, wenn sein Gain (nach Abzug von γ) positiv ist - steuert direkt, wie groß die Bäume werden (siehe Experiment).")
    mcw = st.slider("Mindest-Hessegewicht je Blatt", *bounds("mcw_slider"), key="mcw_slider", step=0.5, format="%.1f",
                    help="Wie min_samples_leaf, aber gewichtet mit der Hesse-Matrix statt gezählten Zeilen (bei Regression = Zeilenzahl, bei Klassifikation kleiner für unsichere Zeilen).")
    subsample = st.slider("Teilstichprobe je Runde [%]", *bounds("subsample_slider"), key="subsample_slider", format="%.0f%%")
    st.markdown("**Daten**")
    n = st.slider("Lieferungen", *bounds("n_slider"), key="n_slider", step=100)
    n_noise = st.slider("Rauschmerkmale", *bounds("n_noise_slider"), key="n_noise_slider")
    if task == "class":
        label_noise = st.slider("Falsche Etiketten im Training [%]", *bounds("label_noise_slider"), key="label_noise_slider")
        st.session_state["_label_noise_kept"] = label_noise
    else:
        label_noise = int(st.session_state.get("_label_noise_kept", C.DEFAULT_LABEL_NOISE))
        st.caption("Falsche Etiketten gibt es nur bei der Klassifikation.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Daten generieren", width="stretch", on_click=randomize_seed)

base_params = (task, int(depth), float(lam), float(gamma), float(mcw), int(n_rounds), float(lr), int(subsample))
data_params = (int(n), int(n_noise), int(label_noise), int(seed))
with st.spinner("Rechne ..."):
    a = _analysis(*base_params, *data_params)
ds = a.ds
names = ds.names
n_feat = len(names)
n_test = len(ds.test)
n_trees = len(a.ensemble.trees)

with st.sidebar:
    st.markdown("**Ansicht**")
    for key, default in (("map_x_select", C.DEFAULT_MAP[0]), ("map_y_select", C.DEFAULT_MAP[1])):
        if st.session_state[key] >= n_feat:
            st.session_state[key] = default
    fx = st.selectbox("Karte: waagerecht", range(n_feat), key="map_x_select", format_func=lambda f: feature_label(names, f))
    fy = st.selectbox("Karte: senkrecht", range(n_feat), key="map_y_select", format_func=lambda f: feature_label(names, f))
    if st.session_state.get("sample_slider", 0) > n_test - 1:
        st.session_state["sample_slider"] = 0
    sample_idx = st.slider("Testlieferung", 0, n_test - 1, 0, key="sample_slider")
sync_query_params({"task_select": task, "depth_slider": int(depth), "n_rounds_slider": int(n_rounds), "lr_slider": float(lr), "lam_slider": float(lam), "gamma_slider": float(gamma),
                   "mcw_slider": float(mcw), "subsample_slider": int(subsample), "n_slider": int(n), "n_noise_slider": int(n_noise), "label_noise_slider": int(label_noise),
                   "seed_input": int(seed), "map_x_select": int(fx), "map_y_select": int(fy)})

view_key = (base_params, data_params)
if st.session_state.get("xgb_owner") != view_key:
    st.session_state["xgb_owner"] = view_key
    st.session_state["xgb_step"] = n_trees

# --- XGBoost in Aktion -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🚀 XGBoost in Aktion")
st.caption("Runde für Runde: links der Baum dieser Runde (Blattwerte = Newton-Schritte, Knoten zeigen den Gain ihres Splits), rechts die Vorhersage des Ensembles bis dahin.")
if n_trees > 1:
    step_col, play_col = st.columns([5, 2])
    with step_col:
        step = st.slider("Runden", 1, n_trees, key="xgb_step")
    with play_col:
        auto_play = st.button("▶️ Abspielen", width="stretch")
else:
    step, auto_play = 1, False
    st.info("ℹ️ Nur eine Runde eingestellt - mehr Runden in der Seitenleiste zeigen den Effekt.")
view_slot = st.empty()
Xte_full, yte_full = ds.X[ds.test], ds.y_true[ds.test] if task == "class" else ds.y_reg[ds.test]
sample_x = Xte_full[sample_idx]
sample_y = yte_full[sample_idx]


def _render(current):
    tree = a.ensemble.trees[current - 1]
    with view_slot.container():
        c1, c2 = st.columns([2, 3])
        with c1:
            st.plotly_chart(build_round_tree(tree, names), width="stretch", key=f"tree_chart_{current}")
            st.caption(f"Runde {current}: {tree.n_leaves} Blätter (λ = {lam:.1f}, γ = {gamma:.1f}).")
        with c2:
            st.plotly_chart(build_map(a.ensemble, ds, task, fx, fy, upto=current, sample=sample_x), width="stretch", key=f"map_chart_{current}")
        pred_here = xgm.predict_value(a.ensemble, sample_x.reshape(1, -1), upto=current)[0]
        pred_text = f"{pred_here:.0%} zu spät" if task == "class" else f"{pred_here:.1f} min"
        truth_text = ("zu spät" if sample_y == 1 else "pünktlich") if task == "class" else f"{sample_y:.1f} min"
        st.markdown(f"**Testlieferung {sample_idx}:** Vorhersage nach {current} Runden = **{pred_text}**; tatsächlich: **{truth_text}**.")


if auto_play:
    frames = sorted(set(np.unique(np.round(np.linspace(1, n_trees, min(12, n_trees))).astype(int))))
    for kk in frames:
        _render(kk)
        time.sleep(min(0.9, 6.0 / len(frames)))
    step = n_trees
else:
    _render(step)

st.markdown("---")

# --- Was das Ensemble gelernt hat -------------------------------------------------------------------------------------------------------------------

st.markdown("## 📐 Was das Ensemble gelernt hat – und wie gut es auf neuen Lieferungen ist")
rrows = _round_rows(*base_params, *data_params)
best = ev.best_round(rrows)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Runden", n_trees)
m2.metric("Trainingsfehler" if task == "class" else "Trainings-RMSE", _err(task, ev.primary(a.train, task)))
m3.metric("Testfehler" if task == "class" else "Test-RMSE", _err(task, ev.primary(a.test, task)), delta=f"ohne Modell: {_err(task, a.baseline)}", delta_color="off")
m4.metric("Blätter insgesamt", a.n_leaves, delta=f"bester Testfehler bei Runde {best['k']}", delta_color="off")
st.markdown(VERDICT_TEXT[a.verdict])

st.markdown("**Testfehler gegen die Rundenzahl**")
st.plotly_chart(build_round_curve(rrows, task, a.baseline, n_trees, best["k"]), width="stretch", key="round_chart")
st.caption(f"Der Trainingsfehler sinkt fast durchgehend; der Testfehler erreicht sein Minimum bei Runde {best['k']} ({_err(task, best['test'])}).")

st.markdown("**Wichtigkeit je Merkmal**")
st.plotly_chart(build_importance(names, a.imp), width="stretch", key="importance_chart")
st.caption("Gemittelt über alle Bäume des Ensembles: Summe der Split-Gains je Merkmal, auf 1 normiert.")

st.markdown("---")

# --- Experimente -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Konvergenz gegen Gradient Boosting")
if st.button("Testfehler gegen die Rundenzahl messen, XGBoost gegen gradient-boosting-demo (dauert einen Moment)", key="conv_start"):
    st.session_state["conv_on"] = True
if st.session_state.get("conv_on"):
    with st.spinner("Trainiere beide Verfahren über mehrere Rundenzahlen auf fünf Datensätzen ..."):
        crows = _convergence(2, 1.0, 0.3, 0.1, int(n), int(n_noise), int(label_noise))
    st.plotly_chart(build_convergence_chart(crows), width="stretch", key="convergence_chart")
    if _gb is not None:
        x15 = next((r["xgb"] for r in crows if r["n_rounds"] == 15), None)
        g30 = next((r["gb"] for r in crows if r["n_rounds"] == 30), None)
        st.caption(f"Jeweils mit der eigenen typischen Lernrate (XGBoost 0.3, Gradient Boosting 0.1), Tiefe 2, Mittel über fünf Datensätze: XGBoost unterschreitet einen Testfehler von 15 % schon "
                   f"bei Runde 15 ({x15:.1%}), Gradient Boosting erst bei Runde 30 ({g30:.1%}) - dank Regularisierung (λ) verträgt XGBoost hier eine größere Lernrate, ohne stärker zu überanpassen.")
    else:
        st.caption("gradient-boosting-demo wurde nicht neben diesem Repo gefunden - nur die XGBoost-Kurve wird gezeigt.")

st.markdown("---")

st.subheader("🔬 Wirkung von λ auf Überanpassung und Blätterzahl")
if st.button("Trainings- und Testfehler sowie Blätterzahl gegen λ messen (dauert einen Moment)", key="lam_start"):
    st.session_state["lam_on"] = True
if st.session_state.get("lam_on"):
    with st.spinner("Trainiere über sechs λ-Werte auf fünf Datensätzen ..."):
        lrows = _lambda_rows(4, 0.0, 150, 0.3, int(n), int(n_noise), int(label_noise))
    c1, c2 = st.columns(2)
    c1.plotly_chart(build_reg_chart(lrows, "lam"), width="stretch", key="lambda_chart")
    c2.plotly_chart(build_leaves_chart(lrows, "lam"), width="stretch", key="lambda_leaves_chart")
    l0, lmax = lrows[0], lrows[-1]
    st.caption(f"Tiefe 4, 150 Runden, γ = 0, Mittel über fünf Datensätze: die Blätterzahl **wächst** sogar leicht mit λ ({l0['leaves']:.0f} → {lmax['leaves']:.0f} bei λ = {lmax['lam']:.0f}) - λ dämpft "
               f"nur die BLATTGEWICHTE, nicht ob überhaupt geschnitten wird. Der Testfehler bewegt sich entsprechend kaum ({l0['test']:.1%} bis {lmax['test']:.1%}), erst bei sehr großem λ sichtbar besser.")

st.markdown("---")

st.subheader("🔬 Wirkung von γ auf Überanpassung und Blätterzahl")
if st.button("Trainings- und Testfehler sowie Blätterzahl gegen γ messen (dauert einen Moment)", key="gam_start"):
    st.session_state["gam_on"] = True
if st.session_state.get("gam_on"):
    with st.spinner("Trainiere über sechs γ-Werte auf fünf Datensätzen ..."):
        grows = _gamma_rows(4, 1.0, 150, 0.3, int(n), int(n_noise), int(label_noise))
    c1, c2 = st.columns(2)
    c1.plotly_chart(build_reg_chart(grows, "gamma"), width="stretch", key="gamma_chart")
    c2.plotly_chart(build_leaves_chart(grows, "gamma"), width="stretch", key="gamma_leaves_chart")
    g0 = grows[0]
    best_g = min(grows, key=lambda r: r["test"])
    st.caption(f"Tiefe 4, 150 Runden, λ = 1, Mittel über fünf Datensätze: anders als λ steuert γ direkt die Baumgröße - die Blätterzahl fällt steil ({g0['leaves']:.0f} bei γ = 0 auf "
               f"{grows[-1]['leaves']:.0f} bei γ = {grows[-1]['gamma']:.0f}), die Lücke zwischen Trainings- und Testfehler schrumpft von {g0['test']-g0['train']:+.1%} auf fast 0. Der beste Testfehler "
               f"liegt bei γ = {best_g['gamma']:.1f} ({best_g['test']:.1%}) - zu viel γ unterpasst wieder (γ = {grows[-1]['gamma']:.0f}: {grows[-1]['test']:.1%}).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **λ allein reicht als Regularisierung** | λ dämpft nur die Blattgewichte, nicht die Baumgröße - in diesem Datensatz wächst die Blätterzahl sogar leicht mit λ (gemessen oben). | γ (Mindest-Gain je Split) zusätzlich einsetzen |
| **γ zu hoch gewählt** | Zu viel Vorwärts-Beschneidung unterpasst wieder - der Testfehler steigt nach einem Minimum erneut (gemessen oben). | γ anhand eines zurückgehaltenen Testfehlers wählen, nicht fest vorgeben |
| **Exakte Split-Suche bei großen Datensätzen** | Diese Demo sucht jede Schwelle jedes Merkmals exakt - bei sehr großen n wird das teuer; das echte XGBoost bietet dafür Histogramm-Varianten (`tree_method="hist"`), hier nicht gebaut. | LightGBM (nächstes Stück): blattweises Wachsen mit Histogramm-Splits von Grund auf |
| **Hesse-Gewicht als Blattgröße** | Bei Klassifikation ist h = p(1-p) ≤ 0.25 - unsichere Zeilen (p nahe 0.5) zählen für `min_child_weight` weniger als sichere; ein Blatt kann dadurch mehr Zeilen enthalten, als die Zahl allein vermuten lässt. | Zeilenzahl direkt prüfen, wenn das überrascht |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**XGBoost** (Chen & Guestrin 2016). Ziel je Runde $m$ (Taylor-Näherung zweiter Ordnung um $F_{m-1}$):
$$\mathcal L^{(m)} \approx \sum_i \big[l(y_i,F_{m-1}(x_i)) + g_i\,\text{Baum}_m(x_i) + \tfrac12 h_i\,\text{Baum}_m(x_i)^2\big] + \gamma T + \tfrac12\lambda\sum_j w_j^2$$
mit $g_i=\partial l/\partial F$, $h_i=\partial^2 l/\partial F^2$, $T$ = Blattzahl, $w_j$ = Blattgewichte.

**Für ein festes Baumgerüst** ist das optimale Blattgewicht $w_j^*=-\dfrac{G_j}{H_j+\lambda}$ mit $G_j=\sum_{i\in R_j} g_i$, $H_j=\sum_{i\in R_j} h_i$; eingesetzt ergibt sich der optimale Zielwert
$-\tfrac12\sum_j \dfrac{G_j^2}{H_j+\lambda}+\gamma T$ - je kleiner, desto besser.

**Gain eines Splits** (Blatt in zwei Kinder L, R geteilt): $\text{Gain}=\tfrac12\Big[\dfrac{G_L^2}{H_L+\lambda}+\dfrac{G_R^2}{H_R+\lambda}-\dfrac{G^2}{H+\lambda}\Big]-\gamma$ - die Differenz der
Zielwerte vor und nach dem Split; nur bei Gain > 0 lohnt sich der Split.

**Verlustfunktionen dieser Demo:** quadratisch (Regression): $g_i=F_i-y_i$, $h_i=1$; Log-Loss (Klassifikation): $g_i=\sigma(F_i)-y_i$, $h_i=\sigma(F_i)(1-\sigma(F_i))$.

**Mit λ = 0, γ = 0** reduziert sich die Gain-Formel exakt auf die (halbierte) Varianzabnahme, die cart-demo/gradient-boosting-demo für quadratischen Verlust berechnet, und der Blattwert auf den
gewöhnlichen Mittelwert der Residuen - beide Bäume wachsen dann strukturell identisch (geprüft in `tests/test_algorithm.py`, bis unabhängige Fließkomma-Gleichstände nach vielen Runden auseinanderlaufen).

Implementiert in `xgb_tree.py` (Baumkern mit regularisierter Split-Suche), `xgb_algorithm.py` (Fit, Vorhersage), `xgb_evaluation.py` (Analyse, Rundenkurve, Konvergenz-, λ- und γ-Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
