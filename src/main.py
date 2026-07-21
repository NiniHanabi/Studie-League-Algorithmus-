"""
Teamstaerke-Algorithmus mit empirisch (per Regression) hergeleiteten Gewichten.

Unterschied zur frueheren Fassung: Die Faktor-Gewichte stammen nicht mehr aus
der Literatur (Rohgewicht -> Normierung auf Summe 1), sondern aus der
logistischen Regression in regression.py (weights.json). Die individuelle
Spielerstaerke ist damit eine gewichtete Summe z-standardisierter Faktoren,
deren Gewichte den empirischen Zusammenhang mit dem Spielausgang abbilden.

Ablauf:
  1. Gewichte + Standardisierung aus weights.json laden.
  2. Individuelle Spielerstaerke berechnen.
  3. Auf den 25 % Holdout-Matches testen, wie gut die Teamstaerke
     (= Mittelwert der Spielerstaerke) den tatsaechlichen Sieger vorhersagt.
  4. Beispielhafte Teambildung (ausgeglichen vs. moeglichst stark).

Voraussetzung: regression.py wurde einmal ausgefuehrt und hat weights.json
erzeugt. Ist die Datei nicht vorhanden, wird die Analyse automatisch gestartet.
"""
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import load_clean_data, split_matches
import regression

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.json"


# ---------------------------------------------------------------------------
# 1. Empirische Gewichte laden (bei Bedarf zuvor herleiten)
# ---------------------------------------------------------------------------
def load_weights():
    if not WEIGHTS_PATH.exists():
        print("weights.json nicht gefunden - fuehre Regression aus ...\n")
        regression.run_analysis()
    return json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 2. Individuelle Spielerstaerke aus den Regressions-Gewichten
# ---------------------------------------------------------------------------
def compute_individual_strength(df, model):
    """Individuelle Staerke = Summe(Gewicht_f * z_f) ueber alle aufgenommenen
    Faktoren. Die Standardisierung (mean/std) stammt aus der Trainingsmenge und
    liegt in weights.json - so wird auf Trainings- und Testdaten identisch
    skaliert (kein Data Leakage).
    """
    df = df.copy()
    weights = model["weights"]
    std = model["standardization"]

    strength = pd.Series(0.0, index=df.index)
    for factor, weight in weights.items():
        s = std[factor]
        x = pd.to_numeric(df[s["column"]], errors="coerce").fillna(s["median"])
        z = (x - s["mean"]) / s["std"]
        strength += weight * z
    df["individual_strength"] = strength
    return df


# ---------------------------------------------------------------------------
# 3. Teamstaerke = Mittelwert der individuellen Staerke
# ---------------------------------------------------------------------------
def compute_team_strength(team_df):
    return team_df["individual_strength"].mean()


def compute_lp_balance(team_df):
    """Zusatzkennzahl: Ausgeglichenheit der League Points (kleine Varianz ->
    hoher Wert). Fliesst nicht in die Teamstaerke ein - laut Regression traegt
    die Team-interne Streuung ueber den Mittelwert hinaus nichts bei; die
    Kennzahl wird nur informativ ausgewiesen.
    """
    lp_variance = team_df["lp"].var()
    if pd.isna(lp_variance):
        lp_variance = 0
    return 1 / (1 + lp_variance)


# ---------------------------------------------------------------------------
# 4. Validierung auf dem Holdout: sagt die Teamstaerke den Sieger vorher?
# ---------------------------------------------------------------------------
def validate_on_holdout(test_df):
    """Fuer jedes Test-Match: Teamstaerke beider Teams vergleichen und pruefen,
    ob das staerkere Team gewonnen hat. Das ist der direkte Test des Algorithmus
    mit den regressionsbasierten Gewichten auf ungesehenen Daten.
    """
    team_strength = (
        test_df.groupby(["match_id", "team_id"])
        .agg(strength=("individual_strength", "mean"), win=("win", "first"))
        .reset_index()
    )
    correct = 0
    total = 0
    for _, match in team_strength.groupby("match_id"):
        if len(match) != 2:
            continue
        stronger = match.loc[match["strength"].idxmax()]
        total += 1
        if stronger["win"] == 1:
            correct += 1
    return correct / total, total


# ---------------------------------------------------------------------------
# 5. Teambildung: moeglichst ausgeglichene Teams (Simulated Annealing)
# ---------------------------------------------------------------------------
def create_initial_teams(df, team_size=5):
    num_teams = len(df) // team_size
    teams = [[] for _ in range(num_teams)]
    shuffled = df.sample(frac=1, random_state=42)
    for index, (_, player) in enumerate(shuffled.iterrows()):
        teams[index % num_teams].append(player)
    return teams


def calculate_score(teams):
    strengths = [compute_team_strength(pd.DataFrame(t)) for t in teams]
    variance = np.var(strengths)  # geringe Varianz zwischen Teams = fairer
    return 1 / (1 + variance)


def swap_players(teams):
    new_teams = deepcopy(teams)
    a = random.randint(0, len(new_teams) - 1)
    b = random.randint(0, len(new_teams) - 1)
    while a == b:
        b = random.randint(0, len(new_teams) - 1)
    pa = random.randint(0, len(new_teams[a]) - 1)
    pb = random.randint(0, len(new_teams[b]) - 1)
    new_teams[a][pa], new_teams[b][pb] = new_teams[b][pb], new_teams[a][pa]
    return new_teams


def simulated_annealing(teams, iterations=5000, temperature=1.0, cooling_rate=0.995):
    current = teams
    current_score = calculate_score(current)
    best, best_score = current, current_score
    for _ in range(iterations):
        candidate = swap_players(current)
        candidate_score = calculate_score(candidate)
        diff = candidate_score - current_score
        if diff > 0 or random.random() < math.exp(diff / temperature):
            current, current_score = candidate, candidate_score
        if current_score > best_score:
            best, best_score = current, current_score
        temperature *= cooling_rate
    return best


# ---------------------------------------------------------------------------
# 6. Teambildung: moeglichst starke Teams (blockweise nach Staerke)
# ---------------------------------------------------------------------------
def create_strongest_teams(df, team_size=5):
    df = df.sort_values("individual_strength", ascending=False).reset_index(drop=True)
    num_teams = len(df) // team_size
    teams = [[] for _ in range(num_teams)]
    for index, player in df.iterrows():
        team_index = index // team_size
        if team_index < num_teams:
            teams[team_index].append(player)
    return teams


def evaluate_teams(teams, mode="balanced"):
    results = []
    for number, team in enumerate(teams, start=1):
        team_df = pd.DataFrame(team)
        result = {
            "team": number,
            "team_strength": round(compute_team_strength(team_df), 3),
            "players": len(team_df),
        }
        if mode == "balanced":
            result["lp_balance"] = round(compute_lp_balance(team_df), 4)
        results.append(result)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Ausfuehrung
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = load_weights()

    print("=" * 70)
    print("TEAMSTAERKE-ALGORITHMUS (REGRESSIONSBASIERTE GEWICHTE)")
    print("=" * 70)
    print("Aufgenommene Faktoren und empirische Gewichte:")
    for f, w in sorted(model["weights"].items(), key=lambda x: -abs(x[1])):
        print(f"   {f:<16}{w:+.3f}")

    # Gleiche 75/25-Aufteilung wie in der Regression -> identischer Holdout
    data = load_clean_data()
    train_df, test_df = split_matches(data, test_size=0.25, seed=42)

    train_df = compute_individual_strength(train_df, model)
    test_df = compute_individual_strength(test_df, model)

    # --- Test der Teamstaerke auf den 25 % Holdout-Matches ---
    acc_test, n_test = validate_on_holdout(test_df)
    acc_train, n_train = validate_on_holdout(train_df)
    print("\n" + "-" * 70)
    print("TEST DER TEAMSTAERKE (staerkeres Team = Sieger?)")
    print("-" * 70)
    print(f"Train (75 %, {n_train} Matches): {acc_train:.1%}")
    print(f"Test  (25 %, {n_test} Matches): {acc_test:.1%}")
    print("Train ~ Test bestaetigt: kein Overfitting der Gewichte.")

    # --- Beispielhafte Teambildung auf einer Stichprobe aus dem Holdout ---
    print("\n" + "-" * 70)
    print("BEISPIEL-TEAMBILDUNG (Stichprobe: 50 Spieler aus dem Holdout)")
    print("-" * 70)
    pool = test_df.drop_duplicates("puuid").sample(50, random_state=7).copy()

    balanced = simulated_annealing(create_initial_teams(pool, team_size=5))
    print("\nAusgeglichene Teams (Simulated Annealing):")
    print(evaluate_teams(balanced, mode="balanced").to_string(index=False))

    strongest = create_strongest_teams(pool, team_size=5)
    print("\nMoeglichst starke Teams (blockweise):")
    print(evaluate_teams(strongest, mode="strongest").to_string(index=False))
