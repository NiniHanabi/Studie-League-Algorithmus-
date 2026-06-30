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

    team_strength = (
        avg_individual * 0.75 +  # stabiler Fix-Anteil (optional wissenschaftlich begründbar)
        lp_balance * TEAM_WEIGHTS["lp_balance"] +
        diversity * TEAM_WEIGHTS["teammate_diversity"]
    )

    return team_strength


# -----------------------------
# 5. TEAM BUILDING
# -----------------------------

def build_teams(df, team_size=5):
    df = df.sample(frac=1).reset_index(drop=True)

    results = []

    for i in range(0, len(df), team_size):
        team = df.iloc[i:i+team_size]

        if len(team) < team_size:
            continue

        results.append({
            "team_strength": compute_team_strength(team),
            "avg_individual_strength": team["individual_strength"].mean()
        })

    return pd.DataFrame(results)