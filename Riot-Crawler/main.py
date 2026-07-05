"""
Einstiegspunkt für den Riot-Crawler.

Ablauf:
  1. Seed-PUUIDs aus Challenger/GM/Master-Ladder holen
  2. Pro PUUID: Match-IDs fetchen
  3. Pro Match: Details fetchen
  4. Ladder-Daten als Fallback für Rank-Lookups
"""
import argparse
from crawler.client import RiotClient
from crawler.seed import get_seed_puuids
from crawler.match_fetcher import get_match_ids, get_match_detail
from crawler.rank_fetcher import (
    load_cache,
    save_cache,
    fetch_all_ladder_data,
    get_rank_from_ladder,
    fetch_rank_by_puuid,
)


def crawl(matches_per_player: int = 10, players_per_division: int = 100, save_interval: int = 10):
    client = RiotClient()
    rank_cache = load_cache()
    seen_matches: set[str] = set()

    print("=== Seed-Phase ===")
    puuids = get_seed_puuids(client, count_per_tier=players_per_division)
    print(f"Total Seed-PUUIDs: {len(puuids)}\n")

    print("=== Fetching Ladder Data (for rank lookups) ===")
    ladder_data = fetch_all_ladder_data(client)
    print()

    print("=== Crawl-Phase ===")
    for i, puuid in enumerate(puuids, 1):
        print(f"[{i}/{len(puuids)}] Fetching match IDs...")
        match_ids = get_match_ids(client, puuid, count=matches_per_player)
        new_ids = [m for m in match_ids if m not in seen_matches]
        print(f"  {len(new_ids)} neue Matches (von {len(match_ids)})")

        for match_id in new_ids:
            seen_matches.add(match_id)
            match = get_match_detail(client, match_id)
            if not match:
                continue

            for participant in match["info"]["participants"]:
                puuid_p = participant["puuid"]
                # Lookup from ladder cache — nur speichern wenn Daten vorhanden
                rank_data = get_rank_from_ladder(puuid_p, ladder_data)
                if rank_data:
                    rank_cache[puuid_p] = rank_data

        if i % save_interval == 0:
            save_cache(rank_cache)
            print(f"  Cache gespeichert ({len(rank_cache)} players, {len(seen_matches)} Matches)")

    save_cache(rank_cache)

    # === Rank-Enrichment: individuelle Lookups für Spieler ohne Ladder-Daten ===
    # Scannt ALLE gecachten Matches (nicht nur die aus diesem Run)
    from pathlib import Path
    import json
    all_puuids_in_matches = set()
    for path in Path("data/matches_raw").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for p in data["info"]["participants"]:
            all_puuids_in_matches.add(p["puuid"])

    missing = [p for p in all_puuids_in_matches if p not in rank_cache]
    print(f"\n=== Rank-Enrichment ===")
    print(f"Spieler ohne Rank-Daten: {len(missing)} von {len(all_puuids_in_matches)}")

    for i, puuid in enumerate(missing, 1):
        rank_data = fetch_rank_by_puuid(client, puuid)
        if rank_data:
            rank_cache[puuid] = rank_data
        if i % save_interval == 0:
            save_cache(rank_cache)
            print(f"  Enrichment: {i}/{len(missing)} ({sum(1 for v in rank_cache.values() if v)} total mit Rank)")

    save_cache(rank_cache)
    print(f"\n=== Done ===")
    print(f"Matches: {len(seen_matches)}")
    print(f"Players with rank data: {sum(1 for v in rank_cache.values() if v)}/{len(rank_cache)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=10, help="Matches pro Spieler")
    parser.add_argument("--players", type=int, default=100, help="Spieler pro Division")
    parser.add_argument("--save-interval", type=int, default=10, help="Speicher-Intervall")
    args = parser.parse_args()

    crawl(matches_per_player=args.matches, players_per_division=args.players, save_interval=args.save_interval)
