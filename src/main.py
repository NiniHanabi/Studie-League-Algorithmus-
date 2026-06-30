import numpy as np
import pandas as pd

# -----------------------------
# 1. GEWICHTE AUS LITERATUR
# -----------------------------

INDIVIDUAL_RAW_WEIGHTS = {
    "rank": 3,
    "league_points": 3,
    "winrate": 1,
    "matches": 3,
    "role_consistency": 1
}

TEAM_RAW_WEIGHTS = {
    "lp_balance": 2,
    "teammate_diversity": 8
}


def normalize_weights(raw_dict):
    total = sum(raw_dict.values())
    return {k: v / total for k, v in raw_dict.items()}


INDIVIDUAL_WEIGHTS = normalize_weights(INDIVIDUAL_RAW_WEIGHTS)
TEAM_WEIGHTS = normalize_weights(TEAM_RAW_WEIGHTS)


# -----------------------------
# 2. NORMALISIERUNG
# -----------------------------

def normalize(series):
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())


# -----------------------------
# 3. INDIVIDUAL SCORE
# -----------------------------

def compute_individual_strength(df):
    df = df.copy()

    df["rank_n"] = normalize(df["rank"])
    df["lp_n"] = normalize(df["league_points"])
    df["winrate_n"] = normalize(df["winrate"])
    df["matches_n"] = normalize(df["matches"])
    df["role_n"] = normalize(df["role_consistency"])

    df["individual_strength"] = (
        df["rank_n"] * INDIVIDUAL_WEIGHTS["rank"] +
        df["lp_n"] * INDIVIDUAL_WEIGHTS["league_points"] +
        df["winrate_n"] * INDIVIDUAL_WEIGHTS["winrate"] +
        df["matches_n"] * INDIVIDUAL_WEIGHTS["matches"] +
        df["role_n"] * INDIVIDUAL_WEIGHTS["role_consistency"]
    )

    return df


# -----------------------------
# 4. TEAM SCORE
# -----------------------------

def compute_team_strength(team_df):

    avg_individual = team_df["individual_strength"].mean()

    lp_variance = team_df["league_points"].var()
    lp_balance = 1 / (1 + lp_variance)

    diversity = team_df["unique_teammates"].mean()
    diversity = normalize(pd.Series([diversity]))[0]

    return (
        avg_individual * 0.75 +
        lp_balance * TEAM_WEIGHTS["lp_balance"] +
        diversity * TEAM_WEIGHTS["teammate_diversity"]
    )


# -----------------------------
# 5. BALANCED TEAMS (FAIR DISTRIBUTION)
# -----------------------------

def build_teams(df, team_size=5):

    df = compute_individual_strength(df)

    # nach Stärke sortieren
    df = df.sort_values("individual_strength", ascending=False).reset_index(drop=True)

    num_teams = len(df) // team_size
    teams = [[] for _ in range(num_teams)]

    direction = 1
    idx = 0

    for _, row in df.iterrows():
        teams[idx].append(row)

        idx += direction

        if idx == num_teams:
            idx = num_teams - 1
            direction = -1
        elif idx < 0:
            idx = 0
            direction = 1

    results = []

    for team in teams:
        team_df = pd.DataFrame(team)

        if len(team_df) < team_size:
            continue

        results.append({
            "team_strength": compute_team_strength(team_df),
            "avg_individual_strength": team_df["individual_strength"].mean()
        })

    return pd.DataFrame(results)


# -----------------------------
# 6. STRONGEST TEAMS (GREEDY STACKING)
# -----------------------------

def build_strongest_teams(df, team_size=5):

    df = compute_individual_strength(df)

    df = df.sort_values("individual_strength", ascending=False).reset_index(drop=True)

    num_teams = len(df) // team_size
    teams = [[] for _ in range(num_teams)]

    for i in range(len(df)):
        team_idx = i // team_size
        if team_idx >= num_teams:
            break
        teams[team_idx].append(df.iloc[i])

    results = []

    for team in teams:
        team_df = pd.DataFrame(team)

        if len(team_df) < team_size:
            continue

        results.append({
            "team_strength": compute_team_strength(team_df),
            "avg_individual_strength": team_df["individual_strength"].mean()
        })

    return pd.DataFrame(results)


# -----------------------------
# 7. OPTIONAL: BEIDES ZUSAMMEN
# -----------------------------

def build_both_team_types(df, team_size=5):

    return {
        "balanced_teams": build_teams(df, team_size),
        "strongest_teams": build_strongest_teams(df, team_size)
    }