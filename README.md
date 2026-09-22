# XGBoost – regularisiertes Ziel und Newton-Schritt – Streamlit-Demo

Siebtes Stück der **Baumbasierten Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning" und das **dritte Stück des Boosting-Asts**
(nach AdaBoost, Gradient Boosting): anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **XGBoost**
(Chen & Guestrin 2016) – an einem wachsenden Beispiel.
Vehikel: dieselben **Lieferungen** wie in cart-demo/.../gradient-boosting-demo, beide Aufgaben (Klassifikation und Regression).
Alle Daten sind erzeugt, alle Zahlen gemessen und in `tests/test_claims.py` festgehalten – keine echten Daten, `xgboost` nur in den Tests als Gegenprobe.

**Bezug zu OR:** ein kleineres, robusteres Ensemble (weniger Blätter durch γ, gedämpfte Gewichte durch λ) liefert stabilere Lieferzeit-Vorhersagen für die Tourenplanung – Überanpassung an
einzelne Trainingslieferungen wirkt sich sonst direkt auf die Zuverlässigkeit der geplanten Zeitfenster aus.

**Einordnung in die Reihe:** Gradient Boosting (voriges Stück) wächst jede Runde einen gewöhnlichen Regressionsbaum auf den Pseudo-Residuen (nur der Gradient, die erste Ableitung) und trägt
danach den verlustoptimalen Blattwert nach. XGBoost geht einen Schritt weiter: die Schnittsuche selbst optimiert direkt das **regularisierte** Ziel `Σ l(y, F+Baum) + γ·Blätter + 0.5·λ·Σ Blattgewichte²`
über eine Newton-Näherung **zweiter** Ordnung (Gradient **und** Hesse-Matrix). γ ist eine **eingebaute Vorwärts-Beschneidung**: ein Schnitt wird nur gemacht, wenn sein Gewinn positiv ist –
anders als das nachträgliche Kosten-Komplexitäts-Beschneiden in cart-demo.

```
CART → Bagging → Random Forest → Extra Trees   (Bagging-Ast, fertig)
CART → AdaBoost → Gradient Boosting → XGBoost (dieses Stück) → { LightGBM, CatBoost }   (Boosting-Ast)
```

| Frage | Ergebnis (1200 Lieferungen, 3 Rauschmerkmale, 70 % Training / 30 % Test, Seed 7, sofern nicht anders angegeben) |
|---|---|
| **Gewinnformel gegen Brute-Force** | ✅ Die vektorisierte Schnittsuche (kumulative Summen von G, H je Merkmal) liefert exakt denselben besten Schnitt wie eine Brute-Force-Suche über alle Schwellen. |
| **Kreuzprobe mit der echten `xgboost`-Bibliothek** (wenige Runden, `tree_method="exact"`, gleiche Parameter) | ✅ Vorhersagen stimmen fast exakt überein (Abweichung ~1e-7, reines Rundungsrauschen) – sowohl Regression als auch Klassifikation, über mehrere Tiefen. |
| **λ = 0, γ = 0 = gradient-boosting-demo** (quadratischer Verlust) | ✅ Exakt identisch für bis zu ~20 Runden; danach können unabhängige Fließkomma-Gleichstände zwischen den zwei verschieden geschriebenen (aber mathematisch identischen) Gewinnformeln einen Gleichstand unterschiedlich auflösen – dokumentiert, nicht versteckt. |
| **Konvergenz gegen Gradient Boosting** (jeweils typische Lernrate, Tiefe 2, Mittel über fünf Datensätze) | ✅ XGBoost unterschreitet einen Testfehler von 15 % bereits bei Runde 15 (14.8 %), Gradient Boosting erst bei Runde 30 (14.9 %) – dank λ verträgt XGBoost eine größere Lernrate (0.3 gegen 0.1), ohne stärker zu überanpassen. |
| **Wirkung von λ** (Tiefe 4, 150 Runden, γ = 0, Mittel über fünf Datensätze) | ⚠️ λ dämpft nur die **Blattgewichte** – die Blätterzahl **wächst** sogar leicht mit λ (1317 → 2026 bei λ = 0 → 20), der Testfehler bewegt sich bis λ = 10 kaum (14.6–14.9 %), erst bei λ = 20 sichtbar besser (13.9 %). |
| **Wirkung von γ** (Tiefe 4, 150 Runden, λ = 1, Mittel über fünf Datensätze) | ✅ γ steuert die Baumgröße direkt: Blätterzahl fällt steil (1536 → 157 von γ = 0 auf 20), die Trainings-Test-Lücke schrumpft von 14.7 auf 0.5 Prozentpunkte. Bestes γ liegt in der Mitte (0.5–1.0, Testfehler 13.8–13.9 %) – zu viel γ unterpasst wieder (γ = 20: 19.4 %). |

## Was die Demo zeigt

- **XGBoost in Aktion:** Runde für Runde mit Schritt-Regler und Abspielen: links der Baum dieser Runde (Blattwerte = Newton-Schritte, Knoten zeigen ihren Schnittgewinn), rechts die Vorhersage
  des Ensembles bis dahin (Entscheidungsgrenze bei Klassifikation, Regressionsfläche bei Regression).
- **Was das Ensemble gelernt hat:** Trainings- und Testfehler gegen die Rundenzahl mit dem besten Testpunkt markiert, Gesamtzahl der Blätter, Wichtigkeit je Merkmal (Summe der Schnittgewinne).
- **Regler:** Aufgabe, Tiefe der Bäume, Rundenzahl, Lernrate, λ, γ, Mindest-Hessegewicht je Blatt, Teilstichprobe je Runde, Rauschmerkmale, falsche Etiketten, Lieferungen, Seed.
- **Drei Experimente auf Knopfdruck:** Konvergenz gegen gradient-boosting-demo, Wirkung von λ, Wirkung von γ (jeweils Trainings-/Testfehler und Blätterzahl).

## Modell und Verfahren

- **Baumkern** (`xgb_tree.py`, **neu geschrieben** – die Gewinnformel unterscheidet sich fundamental vom Varianz-Kriterium aus cart-demo/gradient-boosting-demo): Schnittsuche über
  `0.5·[GL²/(HL+λ) + GR²/(HR+λ) − G²/(H+λ)] − γ`, vektorisiert wie in cart-demo (sortieren, kumulative Summen von Gradient und Hesse-Diagonale je Merkmal). Der Blattwert ist direkt der
  Newton-Schritt `-G/(H+λ)` – keine nachträgliche Korrektur wie `set_leaf_values` in gradient-boosting-demo nötig, weil der Gewinn schon der regularisierte Newton-Schritt ist.
- **Verlustfunktionen:** fest je Aufgabe (quadratisch bei Regression, Log-Loss bei Klassifikation) – der Fokus dieses Stücks liegt auf der Regularisierung, nicht auf wählbaren Verlusten
  (das war gradient-boosting-demos Punkt).
- **Fit:** additive Boosting-Schleife wie in gradient-boosting-demo (`F_m = F_{m-1} + η·Baum_m`), aber jede Runde wächst direkt auf Gradient **und** Hesse-Diagonale statt nur auf dem
  Pseudo-Residuum.

## Was nicht funktioniert hat / gefundene Überraschung

- **λ allein ist ein schwacher Überanpassungs-Hebel in diesem Datensatz:** die ursprüngliche Erwartung war, dass sowohl λ als auch γ die Überanpassung ähnlich gut senken. Gemessen zeigt sich:
  λ dämpft nur die Blattgewichte (macht jede Korrektur kleiner), verhindert aber nicht, dass der Baum genauso tief und groß wächst wie ohne λ – die Blätterzahl wächst sogar leicht mit λ
  (mehr, kleinere Korrekturen brauchen mehr Runden/Blätter, um denselben Trainingsfehler zu erreichen). Erst γ (Mindestgewinn je Schnitt) verhindert das Wachstum selbst. Die App und das
  README berichten das ehrlich, statt beide Regler symmetrisch als "Regularisierung" zu bewerben.
- **Der exakte Abgleich mit gradient-boosting-demo bricht nach vielen Runden ab** – nicht wegen eines Fehlers, sondern weil zwei unabhängig geschriebene (aber mathematisch identische)
  Gewinnformeln bei sehr kleinen Residuen unterschiedlich runden und dadurch einen Gleichstand verschieden auflösen können. Nachgewiesen durch direkten Vergleich der Gewinnwerte an der
  betroffenen Stelle (Unterschied in der 12. Nachkommastelle). Deshalb ist der exakte Test auf moderate Rundenzahlen begrenzt, für mehr Runden gibt es nur noch eine Plausibilitätsprüfung.
- **Kein histogrammbasiertes `tree_method="hist"`:** die Demo sucht jede Schwelle jedes Merkmals exakt (wie cart-demo) – das echte XGBoost bietet dafür eine schnellere Histogramm-Variante,
  hier bewusst nicht gebaut, weil LightGBM (nächstes Stück) genau das zum Hauptthema macht.

## Verifikation

`tests/test_algorithm.py` (14 Tests): Gewinnformel exakt gegen Brute-Force; Vorhersagen fast exakt (~1e-7) gegen die echte `xgboost`-Bibliothek für wenige Runden (Regression und Klassifikation,
mehrere Tiefen); λ = 0/γ = 0 exakt gegen gradient-boosting-demo (moderate Rundenzahl) und dokumentiert nah beieinander (viele Runden); Grenzfälle (Mindest-Hessegewicht, γ prunt stärker, größeres
λ dämpft Gewichte, eine Runde, Reproduzierbarkeit der Teilstichprobe). `tests/test_claims.py` (12 Tests) hält **jede Zahl** aus App und README fest. `tests/test_app.py` (22 Tests) prüft die
Oberfläche per AppTest (jedes Preset, Aufgabenwechsel, Abspielen mit rundenspezifischen Diagramm-Schlüsseln, Permalink, alle drei Experimente).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `xgb_tree.py` | Baumkern mit regularisierter Schnittsuche (neu geschrieben) |
| `xgb_algorithm.py` | Gradient/Hesse je Verlust, Fit, Vorhersage |
| `xgb_scenario.py` | Lieferdaten (wortgleich aus gradient-boosting-demo, ohne `add_outliers`) |
| `xgb_evaluation.py` | Analyse, Rundenkurve, Konvergenz-, λ- und γ-Experimente |
| `xgb_visualization.py` | Baum-, Karten-, Kurven- und Wichtigkeitsdiagramme |
| `xgb_presets.py`, `xgb_constants.py` | Regler, Permalink, Schnellstart-Beispiele, Grenzen |
| `tests/` | Algorithmus-, Claims- und App-Tests |

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m streamlit run app.py
```

## Tests ausführen

```bash
venv\Scripts\python -m pip install -r requirements-dev.txt
venv\Scripts\python -m pytest tests -q
```

---

Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
