"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio, siehe gb_presets.py in gradient-boosting-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import xgb_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


N_FEATURES_MAX = C.N_BASE + C.NOISE_MAX

SETTING_SPECS = {
    "task_select": SettingSpec("task", _choice(C.TASKS), C.DEFAULT_TASK),
    "depth_slider": SettingSpec("depth", int, C.DEFAULT_DEPTH, C.DEPTH_MIN, C.DEPTH_MAX),
    "n_rounds_slider": SettingSpec("nr", int, C.DEFAULT_N_ROUNDS, C.N_ROUNDS_MIN, C.N_ROUNDS_MAX),
    "lr_slider": SettingSpec("lr", float, C.DEFAULT_LR, C.LR_MIN, C.LR_MAX),
    "lam_slider": SettingSpec("lam", float, C.DEFAULT_LAM, C.LAM_MIN, C.LAM_MAX),
    "gamma_slider": SettingSpec("gam", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX),
    "mcw_slider": SettingSpec("mcw", float, C.DEFAULT_MIN_CHILD_WEIGHT, C.MIN_CHILD_WEIGHT_MIN, C.MIN_CHILD_WEIGHT_MAX),
    "subsample_slider": SettingSpec("sub", int, C.DEFAULT_SUBSAMPLE, C.SUBSAMPLE_MIN, C.SUBSAMPLE_MAX),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "n_noise_slider": SettingSpec("nn", int, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "label_noise_slider": SettingSpec("ln", int, C.DEFAULT_LABEL_NOISE, C.LABEL_NOISE_MIN, C.LABEL_NOISE_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
    "map_x_select": SettingSpec("fx", int, C.DEFAULT_MAP[0], 0, N_FEATURES_MAX - 1),
    "map_y_select": SettingSpec("fy", int, C.DEFAULT_MAP[1], 0, N_FEATURES_MAX - 1),
}
PRESET_KEYS = {"task": "task_select", "depth": "depth_slider", "n_rounds": "n_rounds_slider", "lr": "lr_slider", "lam": "lam_slider", "gamma": "gamma_slider",
               "min_child_weight": "mcw_slider", "subsample": "subsample_slider", "n": "n_slider", "n_noise": "n_noise_slider", "label_noise": "label_noise_slider",
               "seed": "seed_input", "fx": "map_x_select", "fy": "map_y_select"}
KEPT = {"label_noise_slider": "_label_noise_kept"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = st.session_state.get(KEPT[state_key], spec.default) if state_key in KEPT else spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
                if state_key in KEPT:
                    st.session_state[KEPT[state_key]] = value
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    p = C.PRESETS[name]
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = p[key]
    for state_key, kept in KEPT.items():
        st.session_state[kept] = st.session_state[state_key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
