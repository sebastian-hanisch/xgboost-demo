"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Aufgabenwechsel, Abspielen, Permalink, alle drei Experimente auf Abruf, Schlüssel."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import xgb_constants as C
from xgb_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _play(at):
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()


def test_default_renders_without_exception_and_gives_one_verdict():
    at = _run()
    assert any("XGBoost in Aktion" in m.value for m in at.markdown) and not at.error


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert not at.error and not at.exception


def test_a_single_step_shows_no_play_button_and_says_so():
    at = _run(lambda a: _apply(a, C.PRESETS["🪓 Ein Schritt (kein Boosting)"]))
    assert not any(b.label == "▶️ Abspielen" for b in at.button)
    assert any("Nur eine Runde eingestellt" in i.value for i in at.info)


def test_switching_task_hides_label_noise_for_regression():
    at = _run()
    labels = {w.label for w in at.sidebar.slider}
    assert "Falsche Etiketten im Training [%]" in labels
    at.session_state["task_select"] = "reg"
    at.run()
    assert not at.exception
    labels = {w.label for w in at.sidebar.slider}
    assert "Falsche Etiketten im Training [%]" not in labels


def test_a_kept_slider_survives_a_round_trip_through_the_other_task():
    at = _run()
    at.session_state["label_noise_slider"] = 12
    at.run()
    at.session_state["task_select"] = "reg"
    at.run()
    at.session_state["task_select"] = "class"
    at.run()
    assert not at.exception and at.slider(key="label_noise_slider").value == 12


def test_extreme_settings_render():
    def small(at):
        at.session_state["n_slider"] = C.N_MIN
        at.session_state["depth_slider"] = C.DEPTH_MIN
        at.session_state["n_rounds_slider"] = C.N_ROUNDS_MIN
        at.session_state["n_noise_slider"] = 0
        at.session_state["lam_slider"] = C.LAM_MIN
        at.session_state["gamma_slider"] = C.GAMMA_MIN

    def big(at):
        at.session_state["task_select"] = "reg"
        at.session_state["n_slider"] = C.N_MAX
        at.session_state["depth_slider"] = C.DEPTH_MAX
        at.session_state["n_rounds_slider"] = 80
        at.session_state["n_noise_slider"] = C.NOISE_MAX
        at.session_state["lam_slider"] = C.LAM_MAX
        at.session_state["gamma_slider"] = C.GAMMA_MAX
    for setup in (small, big):
        at = _run(setup)
        assert not at.exception


def test_map_features_beyond_the_columns_fall_back_after_fewer_noise_features():
    at = _run()
    at.session_state["map_x_select"] = 10
    at.run()
    at.session_state["n_noise_slider"] = 0
    at.run()
    assert not at.exception and at.selectbox(key="map_x_select").value == C.DEFAULT_MAP[0]


def test_the_test_delivery_slider_survives_a_smaller_data_set():
    at = _run()
    at.session_state["sample_slider"] = 300
    at.run()
    at.session_state["n_slider"] = C.N_MIN
    at.run()
    assert not at.exception and at.slider(key="sample_slider").value == 0


def test_step_slider_returns_to_the_last_round_when_settings_change():
    at = _run()
    at.slider(key="xgb_step").set_value(5)
    at.run()
    assert at.slider(key="xgb_step").value == 5
    at.session_state["n_rounds_slider"] = 50
    at.run()
    assert not at.exception and at.slider(key="xgb_step").value == 50


def test_every_round_of_a_small_ensemble_renders():
    at = _run(lambda a: a.session_state.__setitem__("n_rounds_slider", 6))
    for k in range(1, int(at.slider(key="xgb_step").max) + 1):
        at.slider(key="xgb_step").set_value(k)
        at.run()
        assert not at.exception, k


def test_play_renders_several_frames_without_duplicate_keys(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    at = _run(lambda a: a.session_state.__setitem__("n_rounds_slider", 8))
    _play(at)
    assert not at.exception, [e.value for e in at.exception]


def test_permalink_parameters_are_clamped():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["nr"] = "99999"
    at.query_params["depth"] = "-4"
    at.query_params["fx"] = "99"
    at.run()
    assert not at.exception
    assert at.slider(key="n_rounds_slider").value == C.N_ROUNDS_MAX and at.slider(key="depth_slider").value == C.DEPTH_MIN


def test_the_address_bar_mirrors_the_settings():
    at = _run(lambda a: _apply(a, C.PRESETS["✂️ Mit Gamma-Beschneidung"]))
    assert str(at.query_params["gam"]) in ("0.5", "['0.5']") and str(at.query_params["nr"]) in ("150", "['150']")


def test_convergence_experiment_runs_on_demand():
    """gradient-boosting-demo ist nur lokal neben diesem Repo vorhanden (auf CI wird je Repo einzeln ausgecheckt) - dann zeigt die App nur die XGBoost-Kurve mit einem Hinweis
    statt des vollen Vergleichstexts. Beide Fälle sind ein gültiges, unfallfreies Ergebnis."""
    at = _run()
    assert not any("Mittel über fünf Datensätze" in c.value for c in at.caption)
    at.button(key="conv_start").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(c.value for c in at.caption)
    assert "Mittel über fünf Datensätze" in text or "nicht neben diesem Repo gefunden" in text


def test_lambda_experiment_runs_on_demand():
    at = _run()
    at.button(key="lam_start").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(c.value for c in at.caption)
    assert "nur die BLATTGEWICHTE" in text


def test_gamma_experiment_runs_on_demand():
    at = _run()
    at.button(key="gam_start").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(c.value for c in at.caption)
    assert "steuert γ direkt die Baumgröße" in text


def _calls(src, name):
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    assert sorted(set(keys)) == sorted(["tree_chart", "map_chart", "round_chart", "importance_chart", "convergence_chart", "lambda_chart", "lambda_leaves_chart",
                                        "gamma_chart", "gamma_leaves_chart"]), keys
    looped = [c for c in calls if 'key=f"' in c]
    assert len(looped) == 2 and all('_{current}"' in c for c in looped)
    viz = (ROOT / "xgb_visualization.py").read_text(encoding="utf-8")
    assert "fixedrange=True" in viz and viz.count("lock_axes(fig") >= 6


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))
