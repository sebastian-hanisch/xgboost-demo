"""Konstanten und Grenzen der Regler. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

# Merkmale der Lieferungen: (Name, Einheit) - wortgleich aus cart-demo/gradient-boosting-demo
FEATURES = [("Distanz", "km"), ("Ladegewicht", "kg"), ("Stopps", ""), ("Verkehr", "0-1"), ("Wetter", "0-1"), ("Wochentag", "0 = Mo"), ("Zeitfenster-Enge", "0-1"), ("Fahrerjahre", "Jahre")]
N_BASE = len(FEATURES)

TASKS = ("class", "reg")
TASK_LABELS = {"class": "Klassifikation: kommt die Lieferung zu spät?", "reg": "Regression: wie lange dauert die Lieferung?"}
DEFAULT_TASK = "class"

N_MIN, N_MAX, DEFAULT_N = 400, 3000, 1200
NOISE_MIN, NOISE_MAX, DEFAULT_NOISE = 0, 8, 3
LABEL_NOISE_MIN, LABEL_NOISE_MAX, DEFAULT_LABEL_NOISE = 0, 20, 0
TEST_SHARE = 0.3
DEFAULT_SEED = 7

SWEEP_SEEDS = tuple(range(100000, 100005))

OVERFIT_GAP_CLASS = 0.08
OVERFIT_RATIO_REG = 1.6
UNDERFIT_SHARE = 0.75

# XGBoost-eigene Regler
N_ROUNDS_MIN, N_ROUNDS_MAX, DEFAULT_N_ROUNDS = 1, 300, 60
DEPTH_MIN, DEPTH_MAX, DEFAULT_DEPTH = 1, 6, 3
LR_MIN, LR_MAX, DEFAULT_LR = 0.02, 1.0, 0.3
LAM_MIN, LAM_MAX, DEFAULT_LAM = 0.0, 20.0, 1.0                        # L2-Regularisierung der Blattgewichte
GAMMA_MIN, GAMMA_MAX, DEFAULT_GAMMA = 0.0, 20.0, 0.0                  # Mindest-Gain je Split (Vorwärts-Beschneidung)
MIN_CHILD_WEIGHT_MIN, MIN_CHILD_WEIGHT_MAX, DEFAULT_MIN_CHILD_WEIGHT = 0.0, 50.0, 1.0
SUBSAMPLE_MIN, SUBSAMPLE_MAX, DEFAULT_SUBSAMPLE = 20, 100, 100        # Prozent der Trainingszeilen je Runde

DEFAULT_MAP = (0, 3)          # Kartenausschnitt: Distanz x Verkehr

COLORS = {"train": "#1f77b4", "test": "#d62728", "gb": "#9467bd", "xgb": "#1f77b4"}

PRESETS = {
    "🌳 Standard": dict(task="class", depth=3, n_rounds=DEFAULT_N_ROUNDS, lr=DEFAULT_LR, lam=1.0, gamma=0.0, min_child_weight=1.0, subsample=100,
                        n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "🪓 Ein Schritt (kein Boosting)": dict(task="class", depth=1, n_rounds=1, lr=1.0, lam=1.0, gamma=0.0, min_child_weight=1.0, subsample=100,
                                          n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "🌲 Ohne Bremse (gamma = 0)": dict(task="class", depth=6, n_rounds=150, lr=0.3, lam=1.0, gamma=0.0, min_child_weight=1.0, subsample=100,
                                       n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "✂️ Mit Gamma-Beschneidung": dict(task="class", depth=6, n_rounds=150, lr=0.3, lam=1.0, gamma=0.5, min_child_weight=1.0, subsample=100,
                                     n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "📈 Regression Standard": dict(task="reg", depth=3, n_rounds=DEFAULT_N_ROUNDS, lr=DEFAULT_LR, lam=1.0, gamma=0.0, min_child_weight=1.0, subsample=100,
                                   n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED, fx=0, fy=3),
}
PRESET_HELP = {
    "🌳 Standard": "Tiefe 3, 60 Runden, Lernrate 0.3, λ = 1, γ = 0: Trainingsfehler 1.9 %, Testfehler 15.6 % (Raten: 46.4 %), 453 Blätter insgesamt über alle Runden.",
    "🪓 Ein Schritt (kein Boosting)": "Ein einzelner Tiefe-1-Baum (Lernrate 1, keine weiteren Runden): Testfehler 24.4 % - kaum besser als Raten, 2 Blätter insgesamt.",
    "🌲 Ohne Bremse (gamma = 0)": "Tiefe 6, 150 Runden, γ = 0 (kein Mindest-Gain je Split): Trainingsfehler 0 %, Testfehler 16.7 %, aber 1963 Blätter insgesamt - das Ensemble ist riesig und passt sich dem Training vollständig an.",
    "✂️ Mit Gamma-Beschneidung": "Dieselben Einstellungen, nur γ = 0.5 statt 0: nur noch 492 Blätter (−75 %), Trainingsfehler steigt leicht auf 2.0 %, Testfehler sinkt sogar auf 14.7 % (statt 16.7 % ohne Bremse) - kleinere, aber bessere Bäume.",
    "📈 Regression Standard": "Tiefe 3, 60 Runden, Lernrate 0.3, λ = 1, γ = 0, Ziel Lieferdauer: Test-RMSE 9.0 Minuten, 464 Blätter insgesamt.",
}
