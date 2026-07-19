"""
Flacht die gecrawlten Raw-JSON-Matches in einen analysierbaren DataFrame.
Speichert als Parquet und CSV.
"""
import json
import pandas as pd
from pathlib import Path
from crawler.account_fetcher import load_account_cache, save_account_cache, fetch_account_info
from crawler.client import RiotClient

RAW_DIR = Path("data/matches_raw")
OUT_DIR = Path("data/processed")

VALID_ROLES = {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}
RANKED_SOLO_QUEUE_ID = 420  # Ranked Solo/Duo


def build_dataset() -> pd.DataFrame:
    account_cache = load_account_cache()
    rows = []

    print("Processing matches...")
    for match_file in RAW_DIR.glob("*.json"):
        data = json.loads(match_file.read_text(encoding="utf-8"))
        match_id = data["metadata"]["matchId"]

        # Only process Ranked Solo/Duo matches
        if data["info"]["queueId"] != RANKED_SOLO_QUEUE_ID:
            continue

        participants = data["info"]["participants"]

        # Überspringe Matches mit fehlenden/ungültigen Rollen
        roles = [p["teamPosition"] for p in participants]
        if not all(r in VALID_ROLES for r in roles):
            continue

        game_duration = data["info"].get("gameDuration", None)

        for p in participants:
            rank_data = p.get("rankData")

            if rank_data:
                tier = rank_data["tier"]
                rank = rank_data["rank"]
                lp = rank_data["leaguePoints"]
                wins = rank_data.get("wins")
                losses = rank_data.get("losses")
                total_matches = (wins + losses) if (wins is not None and losses is not None) else None
                win_rate = (wins / total_matches) if total_matches else None
            else:
                tier = rank = lp = wins = losses = total_matches = win_rate = None

            flex_data = p.get("flexRank")
            flex_tier = flex_data["tier"] if flex_data else None
            flex_rank = flex_data["rank"] if flex_data else None
            flex_lp = flex_data["leaguePoints"] if flex_data else None

            mastery_data = p.get("championMastery")
            champion_mastery_level = mastery_data["championLevel"] if mastery_data else None
            champion_mastery_points = mastery_data["championPoints"] if mastery_data else None

            # game_name + tagline direkt aus Match-Daten (kein extra API-Call nötig)
            game_name = p.get("riotIdGameName") or (account_cache.get(p["puuid"]) or {}).get("gameName")
            tag_line = p.get("riotIdTagline") or (account_cache.get(p["puuid"]) or {}).get("tagLine")

            rows.append({
                "match_id": match_id,
                "team_id": p["teamId"],
                "role": p["teamPosition"],
                "puuid": p["puuid"],
                "game_name": game_name,
                "tag_line": tag_line,
                "tier": tier,
                "rank": rank,
                "lp": lp,
                "wins": wins,
                "losses": losses,
                "total_matches": total_matches,
                "win_rate": win_rate,
                "summoner_level": p.get("summonerLevel"),
                "flex_tier": flex_tier,
                "flex_rank": flex_rank,
                "flex_lp": flex_lp,
                "champion_name": p.get("championName"),
                "champion_mastery_level": champion_mastery_level,
                "champion_mastery_points": champion_mastery_points,
                "match_duration_s": game_duration,
                "win": int(p["win"]),
            })

    df = pd.DataFrame(rows)

    # --- Rolle Konsistenz pro Spieler (Vertrautheit mit Rolle) ---
    # Anteil der Matches, in denen ein Spieler seine häufigste Rolle gespielt hat
    role_counts = df.groupby(["puuid", "role"]).size().reset_index(name="role_count")
    total_per_player = df.groupby("puuid").size().reset_index(name="player_total")
    dominant_role = (
        role_counts
        .sort_values("role_count", ascending=False)
        .groupby("puuid")
        .first()
        .reset_index()
        .rename(columns={"role": "main_role", "role_count": "main_role_count"})
    )
    dominant_role = dominant_role.merge(total_per_player, on="puuid")
    dominant_role["role_consistency"] = dominant_role["main_role_count"] / dominant_role["player_total"]
    df = df.merge(dominant_role[["puuid", "main_role", "role_consistency"]], on="puuid", how="left")

    # --- Vielfalt früherer Teampartner pro Spieler ---
    # Anzahl einzigartiger Mitspieler (gleiche team_id, gleiche match_id) im Datensatz
    teammate_counts = {}
    for match_id, match_df in df.groupby("match_id"):
        for team_id, team_df in match_df.groupby("team_id"):
            puuids = team_df["puuid"].tolist()
            for puuid in puuids:
                others = set(puuids) - {puuid}
                if puuid not in teammate_counts:
                    teammate_counts[puuid] = set()
                teammate_counts[puuid].update(others)

    df["unique_teammates"] = df["puuid"].map(lambda p: len(teammate_counts.get(p, set())))

    # win_rate auf 4 Nachkommastellen runden (16-stellige Float-Artefakte sind nicht sinnvoll lesbar)
    df["win_rate"] = df["win_rate"].round(4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "matches.parquet", index=False)
    # sep=";" und decimal="," entsprechen dem deutschen Excel-Zahlenformat,
    # damit Punkt-Dezimalzahlen beim Öffnen nicht als Tausender-Trennzeichen fehlinterpretiert werden
    df.to_csv(OUT_DIR / "matches.csv", index=False, sep=";", decimal=",")

    print(f"Dataset: {len(df)} Zeilen, {df['match_id'].nunique()} Matches")
    print(f"Neue Spalten: wins, losses, total_matches, win_rate, summoner_level, flex_tier, flex_rank, flex_lp, "
          f"champion_name, champion_mastery_level, champion_mastery_points, match_duration_s, main_role, "
          f"role_consistency, unique_teammates")

    # Fill in missing account info
    if not account_cache or len(account_cache) < df["puuid"].nunique():
        print("Fetching missing account info (this may take a minute)...")
        client = RiotClient()
        missing_puuids = [p for p in df["puuid"].unique() if p not in account_cache]
        for i, puuid in enumerate(missing_puuids, 1):
            if i % 50 == 0:
                print(f"  {i}/{len(missing_puuids)}")
            fetch_account_info(client, puuid, account_cache)

        save_account_cache(account_cache)
        print("Rebuilding CSV with account info...")
        for i, row in df.iterrows():
            account_info = account_cache.get(row["puuid"])
            if account_info:
                df.loc[i, "game_name"] = account_info["gameName"]
                df.loc[i, "tag_line"] = account_info["tagLine"]

        df.to_csv(OUT_DIR / "matches.csv", index=False, sep=";", decimal=",")
        print("Done!")

    return df


if __name__ == "__main__":
    build_dataset()
