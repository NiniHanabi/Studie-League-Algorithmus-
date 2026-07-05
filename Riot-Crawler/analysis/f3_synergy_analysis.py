"""
F3: Synergien zwischen Rollen

Interaktionsterme zwischen Rollen-LP-Differenzen.
z.B. (JG_diff * MID_diff) → wenn beide hoch, extra Vorteil?

Nutzt Gradient Boosting für nicht-lineare Synergien.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/matches.csv")
df_clean = df[df["lp"].notna()].copy()

# Pro Match, pro Team, pro Rolle: durchschn. LP
role_elo = df_clean.pivot_table(
    index=["match_id", "team_id"],
    columns="role",
    values="lp",
    aggfunc="mean"
).reset_index()

team_outcome = df_clean.groupby(["match_id", "team_id"])["win"].first().reset_index()
role_elo = role_elo.merge(team_outcome, on=["match_id", "team_id"])

# Match-Paare
matches = []
for match_id in role_elo["match_id"].unique():
    teams = role_elo[role_elo["match_id"] == match_id].sort_values("team_id")
    if len(teams) == 2:
        t1, t2 = teams.iloc[0], teams.iloc[1]
        roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        row = {"match_id": match_id, "team_a_won": t1["win"]}
        for role in roles:
            row[f"{role}_diff"] = t1.get(role, np.nan) - t2.get(role, np.nan)
        matches.append(row)

df_matches = pd.DataFrame(matches).dropna()

roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
role_diffs = [f"{r}_diff" for r in roles]

# Basis-Features
X = df_matches[role_diffs].values
y = df_matches["team_a_won"].values

# Mit Interaktionen (wichtige Paare)
interactions = [
    ("JUNGLE", "MIDDLE"),  # Gank-Synergien
    ("BOTTOM", "UTILITY"),  # Lane-Partner
    ("TOP", "JUNGLE"),      # Gank-Unterstützung
]

X_interaction = X.copy()
for r1, r2 in interactions:
    col1 = roles.index(r1)
    col2 = roles.index(r2)
    interaction = X[:, col1] * X[:, col2]
    X_interaction = np.column_stack([X_interaction, interaction])

# Skalieren & Trainieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_interaction)

# Gradient Boosting für Feature Importance
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_scaled, y)

print("=" * 60)
print("F3: Rollen-Synergien")
print("=" * 60)
print(f"\nMatches analysiert: {len(df_matches)}")
print(f"\nGradient Boosting Feature Importance:")
print("(Synergien = Interaktionsterme)\n")

# Feature-Namen für Output
feature_names = role_diffs + [f"{r1}*{r2}" for r1, r2 in interactions]
importance_sorted = sorted(
    zip(feature_names, gb_model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)

for name, imp in importance_sorted:
    if imp > 0.01:
        print(f"  {name:20} -> {imp:.4f}")

print(f"\nModel Accuracy (Gradient Boosting): {gb_model.score(X_scaled, y):.2%}")

# Zusätzlich: Logistische Regression mit Interaktionen (interpretierbar)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_scaled, y)

print(f"\nLogistische Regression mit Interaktionen:")
print(f"  Accuracy: {log_model.score(X_scaled, y):.2%}")

# Top Synergien
print(f"\nStaerkste Synergien (Log-Regression Koeffizienten):")
sync_coefs = list(zip(
    [f"{r1}*{r2}" for r1, r2 in interactions],
    log_model.coef_[0][-len(interactions):]
))
for name, coef in sorted(sync_coefs, key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name:20} -> {coef:+.4f}")
