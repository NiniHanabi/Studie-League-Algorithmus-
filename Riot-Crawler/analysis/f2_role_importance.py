"""
F2: Wie wichtig ist die LP auf einer bestimmten Rolle?

Modelliert Matches als Paare. Für jede Rolle: LP-Differenz zwischen Teams.
Logistische Regression mit Rollen als separate Features.
Koeffizienten zeigen relative Wichtigkeit pro Rolle.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/matches.csv")
df_clean = df[df["lp"].notna()].copy()

# Pro Match, pro Team, pro Rolle: durchschn. LP
role_lp = df_clean.pivot_table(
    index=["match_id", "team_id"],
    columns="role",
    values="lp",
    aggfunc="mean"
).reset_index()

# Gewinnergebnis pro Team
team_outcome = df_clean.groupby(["match_id", "team_id"])["win"].first().reset_index()
role_lp = role_lp.merge(team_outcome, on=["match_id", "team_id"])

# Erstelle Match-Paare
matches = []
for match_id in role_lp["match_id"].unique():
    teams = role_lp[role_lp["match_id"] == match_id].sort_values("team_id")
    if len(teams) == 2:
        t1, t2 = teams.iloc[0], teams.iloc[1]
        roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        row = {"match_id": match_id, "team_a_won": t1["win"]}
        for role in roles:
            row[f"{role}_diff"] = t1.get(role, np.nan) - t2.get(role, np.nan)
        matches.append(row)

df_matches = pd.DataFrame(matches)
df_matches = df_matches.dropna()

roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
X = df_matches[[f"{r}_diff" for r in roles]].values
y = df_matches["team_a_won"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

print("=" * 60)
print("F2: Rollen-Wichtigkeit fuer Team-Staerke")
print("=" * 60)
print(f"\nMatches analysiert: {len(df_matches)}")
print(f"\nLogistische Regression - LP-Differenz Koeffizienten pro Rolle:")
print("(hoeher = wichtiger fuer Gewinn)\n")

role_importance = sorted(
    zip(roles, model.coef_[0]),
    key=lambda x: abs(x[1]),
    reverse=True
)

for role, coef in role_importance:
    print(f"  {role:10} -> {coef:+.4f} (log-odds pro Std.Dev Differenz)")

print(f"\nRanking nach Wichtigkeit:")
for i, (role, coef) in enumerate(role_importance, 1):
    print(f"  {i}. {role:10} ({coef:+.4f})")

print(f"\nModel Accuracy: {model.score(X_scaled, y):.2%}")
