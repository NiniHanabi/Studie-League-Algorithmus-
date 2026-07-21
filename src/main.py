import numpy as np
import pandas as pd
import random
import math
from copy import deepcopy


# 1. Gewichtung der Faktoren festlegen
INDIVIDUAL_RAW_WEIGHTS = {
    "rank": 3,
    "league_points": 3,
    "winrate": 1,
    "matches": 3,
    "summoner_level": 1,
    "champion_mastery": 1,
    "flex_rank": 1
}

# 2. Normalisierung der Gewichte (In Prozentanteile umrechnen: z.B. 3 --> 0.3)
def normalize_weights(raw_dict):
    #Summe aller Gewichte
    total = sum(raw_dict.values())
    #Jedes Gewicht durch die Summe teilen
    return {
        key: value / total
        for key, value in raw_dict.items()
    }
INDIVIDUAL_WEIGHTS = normalize_weights(
    INDIVIDUAL_RAW_WEIGHTS
)

# 3. Min-Max Normalisierung (Da Faktoren unterschiedliche Wertebereiche haben)
#(Schlechtester Wert -> 0, Bester Wert -> 1)
def normalize(series):
    if series.max() == series.min():
        return pd.Series(
            [0] * len(series),
            index=series.index
        )
    return (
        series - series.min()
    ) / (
        series.max() - series.min()
    )

# 4. Berechnung der individuellen Spielerstärke
def compute_individual_strength(df):
    df = df.copy()

    # Normalisierung der Faktoren (Anwendung der Normalisierungs-Funktion)
    df["rank_n"] = normalize(df["rank"])
    df["lp_n"] = normalize(df["league_points"])
    df["winrate_n"] = normalize(df["winrate"])
    df["matches_n"] = normalize(df["matches"])
    df["summoner_level_n"] = normalize(df["summoner_level"])
    df["champion_mastery_n"] = normalize(df["champion_mastery"])
    df["flex_rank_n"] = normalize(df["flex_rank"])

    # Berechnung Spielerstärke
    df["individual_strength"] = (
        df["rank_n"] *
        INDIVIDUAL_WEIGHTS["rank"]
        +
        df["lp_n"] *
        INDIVIDUAL_WEIGHTS["league_points"]
        +
        df["winrate_n"] *
        INDIVIDUAL_WEIGHTS["winrate"]
        +
        df["matches_n"] *
        INDIVIDUAL_WEIGHTS["matches"]
        +
        df["summoner_level_n"] *
        INDIVIDUAL_WEIGHTS["summoner_level"]
        +
        df["champion_mastery_n"] *
        INDIVIDUAL_WEIGHTS["champion_mastery"]
        +
        df["flex_rank_n"] *
        INDIVIDUAL_WEIGHTS["flex_rank"]
    )
    return df

# 5. Berechnung der Teamstärke
def compute_team_strength(team_df):
    return team_df["individual_strength"].mean()

# 6. Berechnung der LP Balance zur Messung der Ausgeglichenheit innerhalb eines Teams (Kleine Varianz --> Hoher Wert)
#(Wird nur als zusätzliche Kennzahl angezeigt --> Beeinflusst nicht die Team Erstellung)
def compute_lp_balance(team_df):
    lp_variance = team_df["league_points"].var()
    if pd.isna(lp_variance):
        lp_variance = 0
    return 1 / (1 + lp_variance)

# 7. Anwendung Simulated Annealing um möglichst ausgeglichene Teams zu erstellen
    #Zufällige Startteams erstellen
def create_initial_teams(df, team_size=5):
    num_teams = len(df) // team_size
    teams = [
        []
        for _ in range(num_teams)
    ]

    shuffled = df.sample(
        frac=1,
        random_state=42
    )

    for index, (_, player) in enumerate(shuffled.iterrows()):
        team_index = index % num_teams
        teams[team_index].append(player)

    return teams

    #Bewertung der Fairness
def calculate_score(teams):
    team_strengths = []

    for team in teams:
        team_df = pd.DataFrame(team)
        team_strengths.append(
            compute_team_strength(team_df)
        )

        #geringe Varianz = bessere Teams
    variance = np.var(team_strengths)
    return 1 / (1 + variance)

    #Spieler zwischen Teams tauschen
def swap_players(teams):
    new_teams = deepcopy(teams)
    team_a = random.randint(
        0,
        len(new_teams)-1
    )

    team_b = random.randint(
        0,
        len(new_teams)-1
    )

    while team_a == team_b:
        team_b = random.randint(
            0,
            len(new_teams)-1
        )

    player_a = random.randint(
        0,
        len(new_teams[team_a])-1
    )
    player_b = random.randint(
        0,
        len(new_teams[team_b])-1
    )

    new_teams[team_a][player_a], new_teams[team_b][player_b] = (
        new_teams[team_b][player_b],
        new_teams[team_a][player_a]
    )
    return new_teams

    #Simulated Annealing Optimierung
def simulated_annealing(
        teams,
        iterations=10000,
        temperature=1.0,
        cooling_rate=0.995
):
    current_solution = teams
    current_score = calculate_score(
        current_solution
    )
    best_solution = current_solution
    best_score = current_score

    for i in range(iterations):
        new_solution = swap_players(
            current_solution
        )
        new_score = calculate_score(
            new_solution
        )
        difference = new_score - current_score

        # bessere Lösung akzeptieren
        if difference > 0:
            current_solution = new_solution
            current_score = new_score

        # schlechtere Lösung manchmal akzeptieren
        elif random.random() < math.exp(
            difference / temperature
        ):
            current_solution = new_solution
            current_score = new_score

        if current_score > best_score:
            best_solution = current_solution
            best_score = current_score

        temperature *= cooling_rate

    return best_solution

# 8. Erstellung der Teams
def create_strongest_teams(df, team_size=5):

    #Spieler werden nach der Stärke sortiert
    df = df.sort_values(
        "individual_strength",
        ascending=False
    ).reset_index(drop=True)

    #Anzahl der Teams berechnen
    num_teams = len(df) // team_size
    teams = [
        []
        for _ in range(num_teams)
    ]

    #Möglichst starke Teams mit blockweiser Sortierung ermitteln
    for index, player in df.iterrows():
        team_index = index // team_size
        if team_index < num_teams:
            teams[team_index].append(
                player
        )
    return teams

# 9. Bewertung der Teams
def evaluate_teams(teams, mode="balanced"):
    results = []

    for number, team in enumerate(teams, start=1):
        team_df = pd.DataFrame(team)
        result = {
    # Zeigt die Team Anzahl, Teamstärke, Spieler und Spieler Namen  
    "team":
        number,
    "team_strength":
        compute_team_strength(team_df),
    "players":
        len(team_df),
    "player_names":
        list(team_df["player_name"])
}
        # Zeigt die LP Balance (Nur bei ausgeglichenen Teams)
        if mode == "balanced":
            result["lp_balance"] = (
                compute_lp_balance(team_df)
            )
        results.append(result)
    return pd.DataFrame(results)

# 10. Algorithmus ausführen
    #Spielerstärke berechnen
df = compute_individual_strength(df)

    #Ergebnis 1: Möglichst ausgeglichene Teams
initial_teams = create_initial_teams(
    df,
    team_size=5
)
balanced_teams = simulated_annealing(
    initial_teams
)
balanced_results = evaluate_teams(
    balanced_teams,
    mode="balanced"
)
    #Ergebnis 2: möglichst starke Teams
strongest_teams = create_strongest_teams(
    df,
    team_size=5
)
strongest_results = evaluate_teams(
    strongest_teams,
    mode="strongest"
)

# 11. Ergebnisse aufzeigen
print("Ausgeglichene Teams")
print(balanced_results)

print("\nStärkste Teams")
print(strongest_results)