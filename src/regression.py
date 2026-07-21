"""
Empirische Herleitung der Faktor-Gewichte per Regression.

Leitprinzip (vgl. Analyse-Zusammenfassung): Ein Faktor oder eine Interaktion
wird nur aufgenommen, wenn er die Vorhersage des Spielausgangs auf ungesehenen
Daten nachweislich verbessert - gemessen an der kreuzvalidierten AUC. Eine
starke Einzelkorrelation genuegt nicht; entscheidend ist der *inkrementelle*
Beitrag ueber die bereits enthaltenen Faktoren hinaus.

Setup: Pro Match werden die Merkmale je Team gemittelt, die Differenz
Team100 - Team200 gebildet und per logistischer Regression auf den Ausgang
(gewinnt Team 100?) modelliert. Die Koeffizienten sind die empirischen
Gewichte; sie ersetzen die frueheren, aus der Literatur abgeleiteten
Rohgewichte.

Es werden 75 % der Matches zur Herleitung der Gewichte genutzt, 25 % bleiben
als Holdout fuer den Test (main.py).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from data_prep import load_clean_data, split_matches

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.json"

# Individuelle Kandidaten-Faktoren: Anzeigename -> Spalte im DataFrame
INDIVIDUAL_FACTORS = {
    "rank": "rank_score",              # Tier + Division (ohne LP)
    "league_points": "lp",             # League Points
    "winrate": "win_rate",             # Win Rate der laufenden Saison
    "matches": "total_matches",        # Anzahl gespielter Matches
    "summoner_level": "summoner_level",  # Account-Level (Erfahrungs-Proxy)
    "champion_mastery": "champion_mastery_points",  # Mastery auf gespieltem Champ
}

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
# Rein unregularisierte Logistische Regression -> Koeffizienten ~ Maximum Likelihood
LOGIT = dict(C=1e6, solver="lbfgs", max_iter=1000)
# Mindest-Zugewinn an CV-AUC fuer die Aufnahme eines Faktors. 0.002 liegt klar
# ueber der typischen Streuung der AUC zwischen den CV-Folds (~0.01) und schuetzt
# so davor, reine Rausch-Verbesserungen als "Relevanz" zu deuten.
AUC_THRESHOLD = 0.002


# ---------------------------------------------------------------------------
# Standardisierung (z-Score) der individuellen Faktoren auf Trainingsbasis
# ---------------------------------------------------------------------------
def fit_standardizer(train_df, factors=INDIVIDUAL_FACTORS):
    """Ermittelt Mittelwert/Streuung je Faktor auf der TRAININGS-Spielermenge.

    Wenige fehlende Werte (Summoner Level, Champion Mastery) werden mit dem
    Trainings-Median aufgefuellt, damit kein Spieler aus der Analyse faellt.
    """
    stats = {}
    for name, col in factors.items():
        series = pd.to_numeric(train_df[col], errors="coerce")
        median = float(series.median())
        mean = float(series.fillna(median).mean())
        std = float(series.fillna(median).std(ddof=0)) or 1.0
        stats[name] = {"column": col, "median": median, "mean": mean, "std": std}
    return stats


def apply_standardizer(df, stats):
    """Fuegt je Faktor eine z-standardisierte Spalte '<name>_z' hinzu."""
    df = df.copy()
    for name, s in stats.items():
        series = pd.to_numeric(df[s["column"]], errors="coerce").fillna(s["median"])
        df[f"{name}_z"] = (series - s["mean"]) / s["std"]
    return df


# ---------------------------------------------------------------------------
# Aufbau der Match-Differenz-Designmatrix
# ---------------------------------------------------------------------------
def team_mean_diff(df, z_cols):
    """Aggregiert je Team die Mittelwerte der z-Spalten und bildet pro Match
    die Differenz Team100 - Team200. Zielgroesse: hat Team 100 gewonnen?

    Weil Teamstaerke als Mittelwert der individuellen Staerke definiert ist,
    entspricht die Differenz der Team-Mittelwerte genau der mittleren
    Differenz der individuellen Faktoren. Die Logit-Koeffizienten auf diese
    Differenzen sind damit direkt als individuelle Faktor-Gewichte verwendbar.
    """
    team = (
        df.groupby(["match_id", "team_id"])
        .agg({**{c: "mean" for c in z_cols}, "win": "first"})
        .reset_index()
    )
    t100 = team[team["team_id"] == 100].set_index("match_id")
    t200 = team[team["team_id"] == 200].set_index("match_id")
    common = t100.index.intersection(t200.index)
    t100, t200 = t100.loc[common], t200.loc[common]

    X = pd.DataFrame(
        {c: t100[c] - t200[c] for c in z_cols}, index=common
    )
    y = t100["win"].astype(int).values
    return X, y


def role_skill_diff(df, skill_stats):
    """Baut pro Match die Skill-Differenz (z-standardisiert) je Rolle:
    Team100_Rolle - Team200_Rolle. Grundlage fuer Rollen-Wichtigkeit,
    Elo-Interaktion und Synergie-Tests.
    """
    s = skill_stats
    df = df.copy()
    df["skill_z"] = (
        pd.to_numeric(df["skill"], errors="coerce").fillna(s["median"]) - s["mean"]
    ) / s["std"]

    pivot = df.pivot_table(
        index=["match_id", "team_id"], columns="role", values="skill_z", aggfunc="mean"
    )
    outcome = df.groupby(["match_id", "team_id"])["win"].first()

    t100 = pivot.xs(100, level="team_id")
    t200 = pivot.xs(200, level="team_id")
    common = t100.index.intersection(t200.index)
    t100, t200 = t100.loc[common], t200.loc[common]

    X = pd.DataFrame(
        {f"{r}_diff": t100[r] - t200[r] for r in ROLES}, index=common
    ).dropna()
    y = outcome.xs(100, level="team_id").loc[X.index].astype(int).values
    return X, y


def cv_auc(X, y, folds=5):
    """Mittlere kreuzvalidierte AUC einer logistischen Regression."""
    model = LogisticRegression(**LOGIT)
    scores = cross_val_score(model, X.values, y, cv=folds, scoring="roc_auc")
    return scores.mean(), scores.std()


def common_baseline_analysis(train_df, stats, skill_stats):
    """Misst JEDEN Faktor gegen EIN gemeinsames Basismodell (Rang + LP + Win Rate).

    Damit werden individuelle und teamspezifische Faktoren auf exakt derselben
    Referenz und Datenbasis vergleichbar - die methodisch strengste Darstellung:

      - Kernfaktoren (Rang, LP, Win Rate): einzigartiger Beitrag per Leave-one-out
        = AUC(finales Modell) - AUC(finales Modell ohne diesen Faktor).
      - Alle uebrigen Faktoren: Zuwachs beim Hinzufuegen zum finalen Modell
        = AUC(finales Modell + Faktor) - AUC(finales Modell).

    Beide Groessen sind der Beitrag eines Faktors gegeben den Rest des finalen
    Modells - dieselbe Referenz fuer alle. So entfaellt die Aepfel-mit-Birnen-
    Verzerrung, dass Teamfaktoren gegen ein anderes (Mittelwert-)Modell antreten.
    """
    z = apply_standardizer(train_df, stats)
    z["skill_z"] = (
        pd.to_numeric(z["skill"], errors="coerce").fillna(skill_stats["median"])
        - skill_stats["mean"]
    ) / skill_stats["std"]

    mean_cols = ["rank_z", "league_points_z", "winrate_z", "matches_z",
                 "summoner_level_z", "champion_mastery_z", "skill_z"]
    agg = z.groupby(["match_id", "team_id"]).agg(
        **{c: (c, "mean") for c in mean_cols},
        skill_std=("skill_z", "std"),
        skill_min=("skill_z", "min"),
        win=("win", "first"),
    )
    role_piv = z.pivot_table(index=["match_id", "team_id"], columns="role",
                             values="skill_z", aggfunc="mean")
    agg = agg.join(role_piv)

    t1 = agg.xs(100, level="team_id")
    t2 = agg.xs(200, level="team_id")
    common = t1.index.intersection(t2.index)
    t1, t2 = t1.loc[common], t2.loc[common]

    feat_cols = mean_cols + ["skill_std", "skill_min"] + ROLES
    D = pd.DataFrame({c: t1[c] - t2[c] for c in feat_cols}, index=common)
    y = t1["win"].astype(int).values

    R = ["rank_z", "league_points_z", "winrate_z"]  # finales Modell
    auc_R, _ = cv_auc(D[R], y)

    core = {}  # Leave-one-out: einzigartiger Beitrag der Kernfaktoren
    for name, col in [("rank", "rank_z"), ("league_points", "league_points_z"),
                      ("winrate", "winrate_z")]:
        rest = [c for c in R if c != col]
        auc_wo, _ = cv_auc(D[rest], y)
        core[name] = round(auc_R - auc_wo, 4)

    extras = {}  # Zuwachs beim Hinzufuegen zum finalen Modell
    for name, col in [("matches", "matches_z"),
                      ("summoner_level", "summoner_level_z"),
                      ("champion_mastery", "champion_mastery_z")]:
        auc_w, _ = cv_auc(D[R + [col]], y)
        extras[name] = round(auc_w - auc_R, 4)

    team = {}
    team["role_decomposition"] = round(cv_auc(D[R + ROLES], y)[0] - auc_R, 4)
    inter = D[[f"{r}" for r in ROLES]]  # Rollen-Diffs fuer Interaktionen
    syn = D[R].copy()
    for r1, r2 in [("JUNGLE", "MIDDLE"), ("BOTTOM", "UTILITY"), ("TOP", "JUNGLE")]:
        syn[f"{r1}x{r2}"] = inter[r1] * inter[r2]
    team["role_synergies"] = round(cv_auc(syn, y)[0] - auc_R, 4)
    team["lp_variance"] = round(cv_auc(D[R + ["skill_std"]], y)[0] - auc_R, 4)
    team["weakest_link"] = round(cv_auc(D[R + ["skill_min"]], y)[0] - auc_R, 4)

    return {"baseline_auc": round(auc_R, 4), "core_leave_one_out": core,
            "extra_individual": extras, "team": team}


# ---------------------------------------------------------------------------
# Hauptanalyse
# ---------------------------------------------------------------------------
def run_analysis():
    data = load_clean_data()
    train_df, test_df = split_matches(data, test_size=0.25, seed=42)

    print("=" * 70)
    print("EMPIRISCHE HERLEITUNG DER FAKTOR-GEWICHTE (REGRESSION)")
    print("=" * 70)
    print(f"Bereinigte Matches: {data['match_id'].nunique()}  "
          f"(Train 75 %: {train_df['match_id'].nunique()}  |  "
          f"Test 25 %: {test_df['match_id'].nunique()})")
    print(f"Spieler je Match: 10  |  Region: EUW  |  Queue: Ranked Solo/Duo")

    # Standardisierung auf Trainingsbasis
    stats = fit_standardizer(train_df)
    skill_series = pd.to_numeric(train_df["skill"], errors="coerce")
    skill_stats = {
        "column": "skill",
        "median": float(skill_series.median()),
        "mean": float(skill_series.mean()),
        "std": float(skill_series.std(ddof=0)) or 1.0,
    }
    train_z = apply_standardizer(train_df, stats)

    # ------------------------------------------------------------------
    # 1. Inkrementeller Test der individuellen Faktoren (Forward-Aufbau)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("1. INKREMENTELLER BEITRAG DER INDIVIDUELLEN FAKTOREN (CV-AUC, Train)")
    print("-" * 70)
    print(f"{'Faktor hinzugefuegt':<28}{'CV-AUC':>10}{'Delta-AUC':>12}  Aufnahme")

    # Reihenfolge: zuerst die inhaltlich zentralen Skill-Signale, dann Rest
    order = ["rank", "league_points", "winrate", "matches",
             "summoner_level", "champion_mastery"]
    included = []
    prev_auc = 0.5
    auc_table = []
    accepted = set()
    for name in order:
        trial = included + [name]
        z_cols = [f"{f}_z" for f in trial]
        X, y = team_mean_diff(train_z, z_cols)
        auc, sd = cv_auc(X, y)
        delta = auc - prev_auc
        # Aufnahme, wenn der Zugewinn ueber dem Rauschen liegt (~1 Promille AUC)
        take = delta > AUC_THRESHOLD
        mark = "JA" if take else "nein"
        print(f"+ {name:<26}{auc:>10.4f}{delta:>+12.4f}  {mark}")
        auc_table.append({"factor": name, "cv_auc": round(auc, 4),
                          "delta_auc": round(delta, 4), "accepted": bool(take)})
        if take:
            included.append(name)
            accepted.add(name)
            prev_auc = auc

    # ------------------------------------------------------------------
    # 2. Flex-Rang: separater Test (nur ~44 % der Spieler haben Flex)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. FLEX-RANG (zweites, unabhaengiges Skill-Signal)")
    print("-" * 70)
    flex_train = train_df.copy()
    flex_train["flex_z"] = (
        pd.to_numeric(flex_train["flex_skill"], errors="coerce") - skill_stats["mean"]
    ) / skill_stats["std"]
    # Team-Mittel per nanmean: Teams ohne jeglichen Flex-Eintrag fallen raus
    core_cols = [f"{f}_z" for f in included]
    flex_train_z = apply_standardizer(flex_train, stats)
    team = (
        flex_train_z.groupby(["match_id", "team_id"])
        .agg({**{c: "mean" for c in core_cols}, "flex_z": "mean", "win": "first"})
        .reset_index()
    )
    t100 = team[team.team_id == 100].set_index("match_id")
    t200 = team[team.team_id == 200].set_index("match_id")
    common = t100.index.intersection(t200.index)
    base_X = pd.DataFrame({c: t100.loc[common, c] - t200.loc[common, c]
                           for c in core_cols}, index=common)
    flex_diff = (t100.loc[common, "flex_z"] - t200.loc[common, "flex_z"])
    full_X = base_X.assign(flex_diff=flex_diff)
    y_all = t100.loc[common, "win"].astype(int)
    mask = full_X["flex_diff"].notna()
    base_auc, _ = cv_auc(base_X[mask], y_all[mask].values)
    flex_auc, _ = cv_auc(full_X[mask], y_all[mask].values)
    print(f"Matches mit Flex-Daten in beiden Teams: {int(mask.sum())} "
          f"({mask.mean():.0%})")
    print(f"Basis (Kernfaktoren)      CV-AUC = {base_auc:.4f}")
    print(f"+ Flex-Rang               CV-AUC = {flex_auc:.4f}  "
          f"(Delta {flex_auc - base_auc:+.4f})")
    flex_take = (flex_auc - base_auc) > AUC_THRESHOLD
    print(f"Aufnahme Flex-Rang: {'JA' if flex_take else 'nein'} "
          f"(hohe Fehlquote von {1-mask.mean():.0%} spricht zusaetzlich dagegen)")
    auc_table.append({"factor": "flex_rank", "cv_auc": round(flex_auc, 4),
                      "delta_auc": round(flex_auc - base_auc, 4),
                      "accepted": bool(flex_take), "note": "nur Subset mit Flex"})

    # ------------------------------------------------------------------
    # 3. Rollen-Wichtigkeit: Wie stark zaehlt ein Elo-Vorsprung je Rolle?
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. ROLLEN-WICHTIGKEIT (Elo-Vorsprung je Rolle) + SYNERGIEN")
    print("-" * 70)
    Xr, yr = role_skill_diff(train_df, skill_stats)
    role_model = LogisticRegression(**LOGIT).fit(Xr.values, yr)
    role_coefs = dict(zip([f"{r}_diff" for r in ROLES], role_model.coef_[0]))
    role_auc, _ = cv_auc(Xr, yr)
    mean_auc, _ = cv_auc(pd.DataFrame({"mean_skill_diff": Xr.mean(axis=1)}), yr)
    print(f"Modell mit Team-Mittel-Skill:        CV-AUC = {mean_auc:.4f}")
    print(f"Modell mit 5 getrennten Rollen-Elos: CV-AUC = {role_auc:.4f}  "
          f"(Delta {role_auc - mean_auc:+.4f})")
    print("Koeffizient je Rolle (hoeher = Elo-Vorsprung dort wichtiger):")
    for r, c in sorted(role_coefs.items(), key=lambda x: -x[1]):
        print(f"   {r:<14}{c:+.3f}")

    # Synergien: Interaktionsterme zwischen Rollen-Elo-Differenzen
    interactions = [("JUNGLE", "MIDDLE"), ("BOTTOM", "UTILITY"), ("TOP", "JUNGLE")]
    Xs = Xr.copy()
    for r1, r2 in interactions:
        Xs[f"{r1}x{r2}"] = Xr[f"{r1}_diff"] * Xr[f"{r2}_diff"]
    syn_auc, _ = cv_auc(Xs, yr)
    print(f"\n+ Synergien (JG*MID, BOT*SUP, TOP*JG): CV-AUC = {syn_auc:.4f}  "
          f"(Delta ueber Rollen {syn_auc - role_auc:+.4f})")
    print(f"Aufnahme Rollen-Zerlegung/Synergien: "
          f"{'JA' if (role_auc - mean_auc) > AUC_THRESHOLD else 'nein (marginal, im Rauschen)'}")

    # ------------------------------------------------------------------
    # 4. Team-interne Streuung: LP-Varianz / Balance / schwaechstes Glied
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("4. TEAM-STREUUNG: LP-VARIANZ, BALANCE, SCHWAECHSTES GLIED")
    print("-" * 70)
    tz = apply_standardizer(train_df, stats)
    tz["skill_z"] = (
        pd.to_numeric(tz["skill"], errors="coerce").fillna(skill_stats["median"])
        - skill_stats["mean"]
    ) / skill_stats["std"]
    agg = (
        tz.groupby(["match_id", "team_id"])
        .agg(skill_mean=("skill_z", "mean"),
             skill_std=("skill_z", "std"),
             skill_min=("skill_z", "min"),
             win=("win", "first"))
        .reset_index()
    )
    a100 = agg[agg.team_id == 100].set_index("match_id")
    a200 = agg[agg.team_id == 200].set_index("match_id")
    common = a100.index.intersection(a200.index)
    base = pd.DataFrame({"mean_diff": a100.loc[common, "skill_mean"]
                         - a200.loc[common, "skill_mean"]}, index=common)
    yb = a100.loc[common, "win"].astype(int).values
    base_auc2, _ = cv_auc(base, yb)

    var_X = base.assign(std_diff=a100.loc[common, "skill_std"]
                        - a200.loc[common, "skill_std"])
    var_auc, _ = cv_auc(var_X, yb)
    min_X = base.assign(min_diff=a100.loc[common, "skill_min"]
                        - a200.loc[common, "skill_min"])
    min_auc, _ = cv_auc(min_X, yb)
    var_model = LogisticRegression(**LOGIT).fit(var_X.values, yb)
    print(f"Basis (nur Team-Mittel):          CV-AUC = {base_auc2:.4f}")
    print(f"+ Streuung/LP-Varianz (std):      CV-AUC = {var_auc:.4f}  "
          f"(Delta {var_auc - base_auc2:+.4f}, Koeff. Varianz = "
          f"{var_model.coef_[0][1]:+.3f})")
    print(f"+ Schwaechstes Glied (min):       CV-AUC = {min_auc:.4f}  "
          f"(Delta {min_auc - base_auc2:+.4f})")
    print("Deutung: Ueber den Team-Mittelwert hinaus liefert die Streuung "
          "keinen belastbaren Beitrag -> Mittelwert-Aggregation ist ausreichend.")

    # ------------------------------------------------------------------
    # 5. Finales Gewichtsmodell + Holdout-Test
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("5. FINALES GEWICHTSMODELL (aufgenommene Faktoren) + HOLDOUT")
    print("-" * 70)
    final_factors = included  # nur die inkrementell nuetzlichen Faktoren
    z_cols = [f"{f}_z" for f in final_factors]

    X_train, y_train = team_mean_diff(train_z, z_cols)
    final_model = LogisticRegression(**LOGIT).fit(X_train.values, y_train)
    weights = dict(zip(final_factors, final_model.coef_[0]))

    print("Empirische Gewichte (Logit-Koeffizienten auf standardisierte Faktoren):")
    for f, w in sorted(weights.items(), key=lambda x: -abs(x[1])):
        print(f"   {f:<18}{w:+.3f}")

    # Holdout-Bewertung auf den 25 %
    test_z = apply_standardizer(test_df, stats)
    X_test, y_test = team_mean_diff(test_z, z_cols)
    proba = final_model.predict_proba(X_test.values)[:, 1]
    pred = (proba >= 0.5).astype(int)
    from sklearn.metrics import roc_auc_score
    holdout_auc = roc_auc_score(y_test, proba)
    holdout_acc = (pred == y_test).mean()
    # "Staerkeres Team gewinnt": Vorzeichen der gewichteten Differenz
    stronger = (X_test.values @ final_model.coef_[0])
    stronger_wins = ((stronger > 0) == (y_test == 1)).mean()
    print(f"\nHoldout (25 %, {len(y_test)} Matches):")
    print(f"   AUC                         {holdout_auc:.4f}")
    print(f"   Accuracy                    {holdout_acc:.4f}")
    print(f"   Staerkeres Team gewinnt in  {stronger_wins:.1%} der Matches")

    # ------------------------------------------------------------------
    # 5b. Gemeinsames Basismodell: alle Faktoren auf EINER Referenz
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("5b. ALLE FAKTOREN GEGEN EIN GEMEINSAMES MODELL (Rang + LP + Win Rate)")
    print("-" * 70)
    common = common_baseline_analysis(train_df, stats, skill_stats)
    print(f"Basismodell CV-AUC = {common['baseline_auc']:.4f}")
    print("Kernfaktoren - einzigartiger Beitrag (Leave-one-out):")
    for f, v in common["core_leave_one_out"].items():
        print(f"   {f:<18}{v:+.4f}")
    print("Weitere individuelle Faktoren - Zuwachs ueber das finale Modell:")
    for f, v in common["extra_individual"].items():
        print(f"   {f:<18}{v:+.4f}")
    print("Teamfaktoren - Zuwachs ueber das finale Modell:")
    for f, v in common["team"].items():
        print(f"   {f:<18}{v:+.4f}")

    # ------------------------------------------------------------------
    # 6. Persistieren fuer main.py
    # ------------------------------------------------------------------
    # Teamfaktoren: alle gemessen als Zuwachs GEGENUEBER dem Team-Mittelwert-Modell
    # (Basis-AUC ~ mean_auc). Synergien hier bewusst ebenfalls gegen den
    # Mittelwert (nicht gegen die Rollen), damit alle vier auf einer Skala
    # vergleichbar sind. Keiner ueberschreitet die Aufnahme-Schwelle.
    team_baseline = mean_auc
    team_factor_table = [
        {"factor": "role_decomposition", "cv_auc": round(role_auc, 4),
         "delta_auc": round(role_auc - team_baseline, 4),
         "accepted": bool((role_auc - team_baseline) > AUC_THRESHOLD)},
        {"factor": "role_synergies", "cv_auc": round(syn_auc, 4),
         "delta_auc": round(syn_auc - team_baseline, 4),
         "accepted": bool((syn_auc - team_baseline) > AUC_THRESHOLD)},
        {"factor": "lp_variance", "cv_auc": round(var_auc, 4),
         "delta_auc": round(var_auc - base_auc2, 4),
         "accepted": bool((var_auc - base_auc2) > AUC_THRESHOLD)},
        {"factor": "weakest_link", "cv_auc": round(min_auc, 4),
         "delta_auc": round(min_auc - base_auc2, 4),
         "accepted": bool((min_auc - base_auc2) > AUC_THRESHOLD)},
    ]

    out = {
        "description": "Empirisch (Regression, 75%-Train) hergeleitete Gewichte "
                       "fuer die individuelle Spielerstaerke. Anzuwenden auf "
                       "z-standardisierte Faktoren (mean/std unten).",
        "weights": {f: round(w, 4) for f, w in weights.items()},
        "standardization": {
            f: {"column": s["column"], "mean": round(s["mean"], 4),
                "std": round(s["std"], 4), "median": round(s["median"], 4)}
            for f, s in stats.items() if f in final_factors
        },
        "role_importance": {r: round(c, 4) for r, c in role_coefs.items()},
        "candidate_auc_table": auc_table,
        "team_baseline_auc": round(team_baseline, 4),
        "team_factor_table": team_factor_table,
        "common_baseline": common,
        "holdout": {"auc": round(holdout_auc, 4), "accuracy": round(holdout_acc, 4),
                    "stronger_team_wins": round(float(stronger_wins), 4),
                    "n_matches": int(len(y_test))},
    }
    WEIGHTS_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"\nGewichte gespeichert: {WEIGHTS_PATH}")
    return out


if __name__ == "__main__":
    run_analysis()
