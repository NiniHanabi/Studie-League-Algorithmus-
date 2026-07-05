"""
F1: Wie beeinflusst LP-Varianz die Team-Stärke?

Modelliert jedes Match als Paar (Team A vs Team B).
Analysiert, ob Team mit niedrigerer Varianz eine höhere Gewinnwahrscheinlichkeit hat,
kontrolliert für durchschnittliche Team-LP.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/matches.csv")

# Filter: nur Rows mit LP-Daten
df_clean = df[df["lp"].notna()].copy()

# Pro Team pro Match: aggregiere Statistiken
team_stats = df_clean.groupby(["match_id", "team_id"]).agg({
    "lp": ["mean", "std", "min", "max", "count"],
    "win": "first"  # all players on a team have same win value
}).reset_index()

team_stats.columns = ["match_id", "team_id", "lp_mean", "lp_std", "lp_min", "lp_max", "n_players", "team_won"]

# Erstelle Match-Paare (Team A=100, Team B=200)
matches = []
for match_id in team_stats["match_id"].unique():
    teams = team_stats[team_stats["match_id"] == match_id].sort_values("team_id")
    if len(teams) == 2:
        t1, t2 = teams.iloc[0], teams.iloc[1]
        matches.append({
            "match_id": match_id,
            "team_a_mean": t1["lp_mean"],
            "team_b_mean": t2["lp_mean"],
            "team_a_std": t1["lp_std"],
            "team_b_std": t2["lp_std"],
            "team_a_won": t1["team_won"],
        })

df_matches = pd.DataFrame(matches)

# Berechne Differenzen für Regression
df_matches["mean_diff"] = df_matches["team_a_mean"] - df_matches["team_b_mean"]
df_matches["std_diff"] = df_matches["team_a_std"] - df_matches["team_b_std"]

# Entferne NaN-Reihen
df_matches = df_matches.dropna()

# Logistische Regression: P(Team A wins) ~ mean_diff + std_diff
X = df_matches[["mean_diff", "std_diff"]].values
y = df_matches["team_a_won"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

print("=" * 60)
print("F1: LP-Varianz und Team-Staerke")
print("=" * 60)
print(f"\nMatches analysiert: {len(df_matches)}")
print(f"\nLogistische Regression Koeffizienten:")
print(f"  Intercept: {model.intercept_[0]:.4f}")
print(f"  Mean LP Differenz: {model.coef_[0][0]:.4f} (pro 1 Std.Dev)")
print(f"  LP Varianz Differenz: {model.coef_[0][1]:.4f} (pro 1 Std.Dev)")

print(f"\nInterpretation:")
print(f"  - +1 SD hoehere durchschn. LP -> {np.exp(model.coef_[0][0]) / (1 + np.exp(model.coef_[0][0])) * 100 - 50:.1f}pp hohere Gewinnchance")
print(f"  - +1 SD niedrigere LP-Varianz (kohaerenter Team) -> {model.coef_[0][1]:.3f} log-odds (negativ=Vorteil)")

# Zusätzliche Statistiken
print(f"\nLP-Varianz-Statistiken:")
print(df_matches[["team_a_std", "team_b_std"]].describe())
