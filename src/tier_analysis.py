"""
Faktor-Check je Tier (Iron ... Challenger).

Die gepoolte und die 3-Band-Analyse koennen verdecken, dass ein global
irrelevanter Faktor in einem einzelnen Tier doch beitraegt (z. B. Rollen-
Wichtigkeit im High-Elo). Dieses Modul wiederholt den inkrementellen Test
deshalb GETRENNT fuer jedes der 10 Tiers und prueft ALLE Kandidaten:

  individuell : matches, summoner_level, champion_mastery, flex
  team        : role_decomposition (Rollen-Wichtigkeit), role_synergies,
                lp_variance, weakest_link

Jeder Faktor wird als Zuwachs an kreuzvalidierter AUC ueber dem tier-internen
Kernmodell (Rang + LP + Win Rate) gemessen. Ueberschreitet ein Faktor in einem
Tier die Schwelle, ist er dort - anders als global - beachtenswert.

Hinweis: Innerhalb eines Tiers ist die Skill-Streuung klein (Matchmaking-
Kompression greift noch staerker), und einzelne Tiers haben wenige Matches ->
kleine Delta-AUC sind mit Vorsicht zu lesen. n wird je Tier ausgewiesen.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data_prep import load_clean_data, TIER_ORDER
from regression import (fit_standardizer, apply_standardizer, cv_auc,
                        ROLES, LOGIT, AUC_THRESHOLD)

OUT_PATH = Path(__file__).resolve().parent / "tier_analysis.json"
INV_TIER = {v: k for k, v in TIER_ORDER.items()}

# Kandidaten (ueber dem Kernmodell Rang+LP+WinRate getestet)
CANDIDATES = ["matches", "summoner_level", "champion_mastery", "flex",
              "role_decomposition", "role_synergies", "lp_variance", "weakest_link"]
CAND_LABEL = {
    "matches": "Matches", "summoner_level": "Level", "champion_mastery": "Mastery",
    "flex": "Flex", "role_decomposition": "Rollen-Wichtigkeit",
    "role_synergies": "Rollen-Synergien", "lp_variance": "LP-Varianz",
    "weakest_link": "Schwaechstes Glied",
}
MIN_MATCHES = 60  # darunter ist 5-fach-CV nicht sinnvoll


def assign_tier(df):
    """Ordnet jedes Match dem Tier des gerundeten mittleren Tier-Index zu."""
    idx = (df.assign(ti=df["tier"].map(TIER_ORDER))
             .groupby("match_id")["ti"].mean())
    return idx.map(lambda a: INV_TIER[int(np.clip(round(a), 0, 9))])


def _team_diff(df, cols, extra_team=None):
    """Team-Mittel je Spalte -> Differenz Team100-Team200. extra_team enthaelt
    bereits auf Teamebene aggregierte Spalten (z. B. skill_std)."""
    agg = df.groupby(["match_id", "team_id"]).agg(
        {**{c: "mean" for c in cols}, **(extra_team or {}), "win": "first"})
    t1 = agg.xs(100, level="team_id")
    t2 = agg.xs(200, level="team_id")
    common = t1.index.intersection(t2.index)
    t1, t2 = t1.loc[common], t2.loc[common]
    diff_cols = cols + list((extra_team or {}).keys())
    X = pd.DataFrame({c: t1[c] - t2[c] for c in diff_cols}, index=common)
    y = t1["win"].astype(int).values
    return X, y


def _cv_delta(z, core, base_index, y_full, extra_cols=None, extra_team=None,
              role_diffs=False, synergy=False):
    """Baut das Modell 'Kern + Zusatz' auf demselben Match-Set und gibt Delta-AUC."""
    cols = core + (extra_cols or [])
    X, _ = _team_diff(z, cols, extra_team)
    if role_diffs or synergy:
        piv = z.pivot_table(index=["match_id", "team_id"], columns="role",
                            values="skill_z", aggfunc="mean")
        t1 = piv.xs(100, level="team_id"); t2 = piv.xs(200, level="team_id")
        common = X.index.intersection(t1.index).intersection(t2.index)
        rd = pd.DataFrame({f"{r}_d": t1.loc[common, r] - t2.loc[common, r] for r in ROLES})
        X = X.loc[common]
        if role_diffs:
            X = pd.concat([X, rd], axis=1)
        if synergy:
            for a, b in [("JUNGLE", "MIDDLE"), ("BOTTOM", "UTILITY"), ("TOP", "JUNGLE")]:
                X[f"{a}x{b}"] = rd[f"{a}_d"] * rd[f"{b}_d"]
    X = X.dropna()
    y = pd.Series(y_full, index=base_index).loc[X.index].values
    return cv_auc(X, y)[0]


def run():
    data = load_clean_data()
    data["tier_grp"] = data["match_id"].map(assign_tier(data))

    print("=" * 78)
    print("FAKTOR-CHECK JE TIER  (Delta-AUC ueber Kernmodell Rang+LP+WinRate)")
    print("=" * 78)

    tiers_sorted = sorted(data["tier_grp"].dropna().unique(), key=lambda t: TIER_ORDER[t])
    results = {}
    for tier in tiers_sorted:
        tdf = data[data["tier_grp"] == tier]
        stats = fit_standardizer(tdf)
        z = apply_standardizer(tdf, stats)
        sk = pd.to_numeric(z["skill"], errors="coerce")
        z["skill_z"] = (sk.fillna(sk.median()) - sk.mean()) / (sk.std(ddof=0) or 1.0)
        grp = z.groupby(["match_id", "team_id"])
        z = z.assign(_skill_std=grp["skill_z"].transform("std").fillna(0.0),
                     _skill_min=grp["skill_z"].transform("min"))

        core = ["rank_z", "league_points_z", "winrate_z"]
        X_core, y = _team_diff(z, core)
        n = len(y)
        if n < MIN_MATCHES:
            results[tier] = {"n": int(n), "core_auc": None, "core_std": None,
                             "deltas": {}}
            continue
        # core_std = CV-Streuung der Kern-AUC = tier-eigener Rauschpegel.
        # Bei kleinen Tiers ist er weit groesser als die feste 0,002-Schwelle;
        # nur Deltas oberhalb dieses Rauschpegels sind glaubwuerdig.
        auc_core, core_std = cv_auc(X_core, y)
        base_index = X_core.index

        deltas = {}
        deltas["matches"] = _cv_delta(z, core, base_index, y, ["matches_z"]) - auc_core
        deltas["summoner_level"] = _cv_delta(z, core, base_index, y, ["summoner_level_z"]) - auc_core
        deltas["champion_mastery"] = _cv_delta(z, core, base_index, y, ["champion_mastery_z"]) - auc_core
        deltas["role_decomposition"] = _cv_delta(z, core, base_index, y, role_diffs=True) - auc_core
        deltas["role_synergies"] = _cv_delta(z, core, base_index, y, synergy=True) - auc_core
        deltas["lp_variance"] = _cv_delta(z, core, base_index, y, extra_team={"_skill_std": "mean"}) - auc_core
        deltas["weakest_link"] = _cv_delta(z, core, base_index, y, extra_team={"_skill_min": "mean"}) - auc_core

        # Flex nur auf Subset mit Flex in beiden Teams
        zf = z.copy()
        zf["flex_z"] = ((pd.to_numeric(zf["flex_skill"], errors="coerce") - sk.mean())
                        / (sk.std(ddof=0) or 1.0))
        fagg = zf.groupby(["match_id", "team_id"]).agg(
            {**{c: "mean" for c in core}, "flex_z": "mean", "win": "first"})
        f1, f2 = fagg.xs(100, level="team_id"), fagg.xs(200, level="team_id")
        fc = f1.index.intersection(f2.index)
        base_f = pd.DataFrame({c: f1.loc[fc, c] - f2.loc[fc, c] for c in core}, index=fc)
        flexd = (f1.loc[fc, "flex_z"] - f2.loc[fc, "flex_z"])
        yf = f1.loc[fc, "win"].astype(int).values
        mask = flexd.notna().values
        if mask.sum() >= MIN_MATCHES:
            a_b = cv_auc(base_f[mask], yf[mask])[0]
            a_f = cv_auc(base_f.assign(flex=flexd)[mask], yf[mask])[0]
            deltas["flex"] = a_f - a_b
        else:
            deltas["flex"] = None

        results[tier] = {"n": int(n), "core_auc": round(auc_core, 4),
                         "core_std": round(core_std, 4),
                         "deltas": {k: (round(v, 4) if v is not None else None)
                                    for k, v in deltas.items()}}

    _print_table(results, tiers_sorted)
    OUT_PATH.write_text(json.dumps({"tiers": results, "threshold": AUC_THRESHOLD,
                                    "candidate_label": CAND_LABEL}, indent=2,
                                   ensure_ascii=False), encoding="utf-8")
    print(f"\nErgebnisse gespeichert: {OUT_PATH}")
    return results


def _credible(v, r):
    """Delta gilt nur als glaubwuerdig, wenn es sowohl die feste Schwelle als
    auch den tier-eigenen Rauschpegel (CV-Streuung der Kern-AUC) uebersteigt."""
    if v is None or r.get("core_std") is None:
        return False
    return v > AUC_THRESHOLD and v > r["core_std"]


def _print_table(results, tiers):
    print(f"\n{'Tier':<13}{'n':>6}{'Kern-AUC':>10}{'Rausch±':>9}   "
          f"Delta-AUC je Faktor (* = ueber Rauschpegel UND Schwelle)")
    print("-" * 78)
    for tier in tiers:
        r = results[tier]
        if r["core_auc"] is None:
            print(f"{tier:<13}{r['n']:>6}   (zu wenige Matches)")
            continue
        print(f"{tier:<13}{r['n']:>6}{r['core_auc']:>10.4f}{r['core_std']:>9.4f}")
        for c in CANDIDATES:
            v = r["deltas"].get(c)
            if v is None:
                print(f"     {CAND_LABEL[c]:<22}   n/a")
            else:
                mark = " *" if _credible(v, r) else ""
                print(f"     {CAND_LABEL[c]:<22}{v:>+9.4f}{mark}")
    print("-" * 78)
    hits = [(t, c) for t in tiers for c in CANDIDATES
            if _credible(results[t]["deltas"].get(c), results[t])]
    if hits:
        print("Ueber Rauschpegel UND Schwelle (in diesem Tier evtl. beachtenswert):")
        for t, c in hits:
            print(f"   {t:<13} {CAND_LABEL[c]}  (Delta {results[t]['deltas'][c]:+.4f}, "
                  f"Rausch {results[t]['core_std']:.4f})")
    else:
        print("Kein Faktor uebersteigt in IRGENDEINEM Tier den tier-eigenen")
        print("Rauschpegel. Die scheinbaren Treffer der festen 0,002-Schwelle")
        print("sind Stichproben-Rauschen (kleine n je Tier), kein echtes Signal.")


if __name__ == "__main__":
    run()
