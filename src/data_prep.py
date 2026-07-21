"""
Datenaufbereitung fuer die empirische Teamstaerke-Analyse.

Stellt das Laden, Bereinigen und Aufteilen des Roh-Datensatzes
(matches.csv aus dem Riot-Crawler) an einer zentralen Stelle bereit,
damit sowohl die Regression (regression.py) als auch der Team-Builder
(main.py) auf exakt derselben, sauberen Datenbasis arbeiten.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# Pfad zur CSV relativ zu dieser Datei (robust gegen aktuelles Arbeitsverzeichnis)
DEFAULT_CSV = (
    Path(__file__).resolve().parent.parent
    / "Riot-Crawler" / "data" / "processed" / "matches.csv"
)

# Ordinale Reihenfolge der Ligen (Iron am schwaechsten, Challenger am staerksten)
TIER_ORDER = {
    "IRON": 0, "BRONZE": 1, "SILVER": 2, "GOLD": 3, "PLATINUM": 4,
    "EMERALD": 5, "DIAMOND": 6, "MASTER": 7, "GRANDMASTER": 8, "CHALLENGER": 9,
}
# Divisionen laufen von IV (unten) nach I (oben)
DIVISION_ORDER = {"IV": 0, "III": 1, "II": 2, "I": 3}

# Apex-Ligen teilen sich eine gemeinsame LP-Leiter (keine Divisionen).
APEX_TIERS = {"MASTER", "GRANDMASTER", "CHALLENGER"}
# Basiswert, ab dem die Apex-Leiter beginnt (= Ende von Diamond I).
APEX_BASE = TIER_ORDER["MASTER"] * 400  # 2800


def rank_component(tier, division):
    """Reine Rang-Komponente (Tier + Division) OHNE League Points.

    - Iron bis Diamond: Tier * 400 + Division * 100.
    - Master/GM/Challenger: APEX_BASE (die drei Apex-Ligen kennen keine
      Divisionen; ihre Feinordnung uebernehmen allein die League Points).

    Wird getrennt vom LP gefuehrt, damit Rang und League Points im Modell
    zwei eigenstaendige, separat gewichtbare Faktoren sind (vgl. Basismodell).
    """
    if pd.isna(tier) or tier not in TIER_ORDER:
        return np.nan
    if tier in APEX_TIERS:
        return float(APEX_BASE)
    div = DIVISION_ORDER.get(division, 0)
    return float(TIER_ORDER[tier] * 400 + div * 100)


def skill_score(tier, division, lp):
    """Fasst Tier, Division und League Points zu einem durchgehenden,
    ordinalen Skill-Score zusammen (= Rang-Komponente + League Points).

    - Iron bis Diamond: Tier * 400 + Division * 100 + LP
      (LP liegt hier zwischen 0 und ~100, die Division staffelt innerhalb des Tiers).
    - Master/GM/Challenger: APEX_BASE + LP. Diese drei Ligen bilden real eine
      gemeinsame LP-Leiter; die League Points allein ordnen sie korrekt und
      vermeiden die Ueberlappungen, die ein zusaetzlicher Tier-Offset erzeugen wuerde.
    """
    rc = rank_component(tier, division)
    if pd.isna(rc):
        return np.nan
    lp = 0.0 if pd.isna(lp) else float(lp)
    return rc + lp


def load_clean_data(csv_path=DEFAULT_CSV):
    """Laedt den Datensatz und wendet die notwendigen Bereinigungsschritte an.

    Bereinigung (vgl. Analyse-Zusammenfassung, Abschnitt Datenqualitaet):
    - Nur EUW: Matches mit EUNE-Praefix (EUN1_) werden entfernt.
    - Nur Spieler mit gueltigem Solo/Duo-Ranglisten-Eintrag (tier vorhanden).
    - Skill-Score fuer Solo/Duo und (sofern vorhanden) fuer Flex.
    - Nur vollstaendige Matches mit genau 10 gerankten Spielern bleiben erhalten,
      damit die Team-Mittelwerte nicht durch fehlende Spieler verzerrt werden.
    """
    df = pd.read_csv(csv_path, sep=";", decimal=",")

    # Region aus dem Match-ID-Praefix (EUW1_... / EUN1_...)
    df["region"] = df["match_id"].str.split("_").str[0]
    df = df[df["region"] == "EUW1"].copy()

    # Spieler ohne Ranglisten-Eintrag koennen nicht bewertet werden
    df = df[df["tier"].notna()].copy()

    # Reine Rang-Komponente (Tier + Division, ohne LP)
    df["rank_score"] = df.apply(
        lambda r: rank_component(r["tier"], r["rank"]), axis=1
    )
    # Durchgehender Skill-Score (Solo/Duo) = Rang-Komponente + LP
    df["skill"] = df.apply(
        lambda r: skill_score(r["tier"], r["rank"], r["lp"]), axis=1
    )
    # Flex-Skill (nur fuer Spieler mit Flex-Eintrag, sonst NaN)
    df["flex_skill"] = df.apply(
        lambda r: skill_score(r["flex_tier"], r["flex_rank"], r["flex_lp"]), axis=1
    )

    # Numerische Felder sicher als float (Komma-Dezimal wurde bereits geparst)
    for col in ["win_rate", "total_matches", "summoner_level",
                "champion_mastery_points", "lp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Nur vollstaendige Matches: genau 10 gerankte Spieler, 2 Teams a 5
    sizes = df.groupby("match_id").size()
    valid_matches = sizes[sizes == 10].index
    df = df[df["match_id"].isin(valid_matches)].copy()

    # Zusaetzliche Kontrolle: 5 Spieler je Team
    team_sizes = df.groupby(["match_id", "team_id"]).size()
    bad_matches = team_sizes[team_sizes != 5].index.get_level_values("match_id").unique()
    df = df[~df["match_id"].isin(bad_matches)].copy()

    return df.reset_index(drop=True)


def split_matches(df, test_size=0.25, seed=42):
    """Teilt den Datensatz auf Match-Ebene in Trainings- und Testmenge.

    Der Split erfolgt bewusst pro Match (nicht pro Spieler): Beide Teams eines
    Matches landen gemeinsam in Train oder Test. So wird verhindert, dass Teile
    eines Matches beim Training gesehen werden und der Ausgang "durchsickert".
    75 % dienen der Herleitung der Gewichte, 25 % dem Test der Teamstaerke.
    """
    rng = np.random.default_rng(seed)
    match_ids = df["match_id"].unique()
    rng.shuffle(match_ids)
    n_test = int(len(match_ids) * test_size)
    test_ids = set(match_ids[:n_test])

    train_df = df[~df["match_id"].isin(test_ids)].copy()
    test_df = df[df["match_id"].isin(test_ids)].copy()
    return train_df, test_df


if __name__ == "__main__":
    data = load_clean_data()
    train, test = split_matches(data)
    print(f"Bereinigt: {len(data)} Zeilen, {data['match_id'].nunique()} Matches")
    print(f"Train: {train['match_id'].nunique()} Matches | "
          f"Test: {test['match_id'].nunique()} Matches")
    print(f"Flex-Abdeckung: {data['flex_skill'].notna().mean():.1%} der Spieler")
    print(data[["tier", "rank", "lp", "skill", "flex_skill",
                "win_rate", "champion_mastery_points"]].head())
