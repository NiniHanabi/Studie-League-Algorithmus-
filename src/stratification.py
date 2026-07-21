"""
Elo-Stratifizierung: sind die Faktor-Gewichte ueber die Leistungsniveaus stabil?

Die gepoolte Stichprobe ist nicht repraesentativ (Apex ~9-fach ueberrepraesentiert).
Deshalb wird die Regression getrennt fuer drei Elo-Baender wiederholt:

    low  = Iron - Gold        (mittlerer Tier-Index < 3,5)
    mid  = Platin - Diamond   (3,5 <= Index < 6,5)
    high = Master - Challenger (Index >= 6,5)

Die Bandzuordnung erfolgt ueber den mittleren Tier-Index der zehn Spieler eines
Matches. Innerhalb jedes Bandes werden die Faktoren MIT DER BANDEIGENEN Streuung
z-standardisiert - so bedeutet jedes Gewicht "Effekt pro bandinterner
Standardabweichung" und ist zwischen den Baendern interpretierbar vergleichbar.

Ergebnis pro Band: standardisierte Gewichte (Rang, LP, Win Rate), CV-AUC und der
Anteil "staerkeres Team gewinnt". Wird als stratification.json gespeichert und
in Abbildung 4 dargestellt.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data_prep import load_clean_data, split_matches, TIER_ORDER
from regression import fit_standardizer, apply_standardizer, team_mean_diff, cv_auc, LOGIT

OUT_PATH = Path(__file__).resolve().parent / "stratification.json"

FACTORS_STRAT = {"rank": "rank_score", "league_points": "lp", "winrate": "win_rate"}
BANDS = ["low", "mid", "high"]
BAND_LABEL = {"low": "Iron-Gold", "mid": "Platin-Diamond", "high": "Master+"}


def assign_bands(df):
    """Ordnet jedem Match ein Elo-Band ueber den mittleren Tier-Index zu."""
    idx = df.assign(tier_idx=df["tier"].map(TIER_ORDER)) \
            .groupby("match_id")["tier_idx"].mean()

    def band(a):
        if a < 3.5:
            return "low"
        if a < 6.5:
            return "mid"
        return "high"

    return idx.map(band)


def analyse_band(band_train, band_test):
    """Leitet innerhalb eines Bandes die Gewichte her und bewertet sie."""
    stats = fit_standardizer(band_train, FACTORS_STRAT)
    z_cols = ["rank_z", "league_points_z", "winrate_z"]

    tr = apply_standardizer(band_train, stats)
    te = apply_standardizer(band_test, stats)
    X_tr, y_tr = team_mean_diff(tr, z_cols)
    X_te, y_te = team_mean_diff(te, z_cols)

    model = LogisticRegression(**LOGIT).fit(X_tr.values, y_tr)
    weights = dict(zip(["rank", "league_points", "winrate"], model.coef_[0]))

    auc_cv, _ = cv_auc(X_tr, y_tr)
    proba = model.predict_proba(X_te.values)[:, 1]
    holdout_auc = roc_auc_score(y_te, proba)
    stronger = X_te.values @ model.coef_[0]
    stronger_wins = float(((stronger > 0) == (y_te == 1)).mean())

    return {
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "factor_std": {k: round(stats[k]["std"], 2) for k in FACTORS_STRAT},
        "cv_auc": round(auc_cv, 4),
        "holdout_auc": round(holdout_auc, 4),
        "stronger_team_wins": round(stronger_wins, 4),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }


def run():
    data = load_clean_data()
    data["band"] = data["match_id"].map(assign_bands(data))
    train_df, test_df = split_matches(data, test_size=0.25, seed=42)

    print("=" * 74)
    print("ELO-STRATIFIZIERUNG: STABILITAET DER GEWICHTE UEBER DIE LEISTUNGSNIVEAUS")
    print("=" * 74)

    results = {}
    for b in BANDS:
        bt = train_df[train_df["band"] == b]
        be = test_df[test_df["band"] == b]
        results[b] = analyse_band(bt, be)

    # Vergleichstabelle
    hdr = f"{'':16}" + "".join(f"{BAND_LABEL[b]:>16}" for b in BANDS)
    print("\n" + hdr)
    print(f"{'Matches (Train)':16}" + "".join(f"{results[b]['n_train']:>16}" for b in BANDS))
    print("-" * 74)
    print("Standardisierte Gewichte (Effekt pro bandinterner Streuung):")
    for f, name in [("rank", "Rang"), ("league_points", "League Points"),
                    ("winrate", "Win Rate")]:
        print(f"  {name:14}" + "".join(f"{results[b]['weights'][f]:>+16.3f}" for b in BANDS))
    print("-" * 74)
    print("Modellguete je Band:")
    print(f"  {'CV-AUC':14}" + "".join(f"{results[b]['cv_auc']:>16.4f}" for b in BANDS))
    print(f"  {'Holdout-AUC':14}" + "".join(f"{results[b]['holdout_auc']:>16.4f}" for b in BANDS))
    print(f"  {'Staerk.gewinnt':14}" + "".join(f"{results[b]['stronger_team_wins']:>15.1%} " for b in BANDS))
    print("-" * 74)
    print("Absolute Streuung der Faktoren je Band (zeigt: Rang wird in Apex")
    print("nahezu konstant, LP uebernimmt dort die gesamte Skill-Ordnung):")
    for f, name in [("rank", "std Rang"), ("league_points", "std LP"),
                    ("winrate", "std WinRate")]:
        print(f"  {name:14}" + "".join(f"{results[b]['factor_std'][f]:>16.2f}" for b in BANDS))

    OUT_PATH.write_text(json.dumps({"bands": results, "band_label": BAND_LABEL},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nErgebnisse gespeichert: {OUT_PATH}")
    return results


if __name__ == "__main__":
    run()
