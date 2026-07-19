"""
Einstiegspunkt für den Riot-Crawler.

Ablauf:
  1. Pro Division (Iron-Challenger, I-IV) die angegebene Anzahl Spieler ziehen,
     inkl. exakter Elo (Tier/Rank/LP/Wins/Losses) aus derselben Response
  2. Pro Spieler: die angegebene Anzahl Matches fetchen
  3. Pro Match: für alle 10 Teilnehmer die Elo ins Match-JSON einbetten
     (bekannte Spieler aus dem Rank-Cache, unbekannte per Einzel-Request)
"""
import argparse
from crawler.client import RiotClient
from crawler.seed import get_seed_players
from crawler.match_fetcher import get_match_ids, fetch_match_detail, save_match
from crawler.rank_fetcher import save_cache, fetch_rank_by_puuid


def crawl(matches_per_player: int = 10, players_per_division: int = 100, save_interval: int = 10):
    # Wichtig: nur EIN Key pro Lauf — PUUIDs sind an das Produkt gebunden, das sie
    # ausgegeben hat, und lassen sich nicht mit einem anderen Key weiterverarbeiten.
    client = RiotClient()

    print("=== Seed-Phase ===")
    seed_players = get_seed_players(client, count_per_tier=players_per_division)
    print(f"Total Seed-Spieler: {len(seed_players)}\n")

    rank_cache = {p["puuid"]: {k: v for k, v in p.items() if k != "puuid"} for p in seed_players}

    print("=== Crawl-Phase ===")
    seen_matches: set[str] = set()
    for i, player in enumerate(seed_players, 1):
        puuid = player["puuid"]
        print(f"[{i}/{len(seed_players)}] Fetching match IDs...")
        match_ids = get_match_ids(client, puuid, count=matches_per_player)
        new_ids = [m for m in match_ids if m not in seen_matches]
        print(f"  {len(new_ids)} neue Matches (von {len(match_ids)})")

        for match_id in new_ids:
            seen_matches.add(match_id)
            match = fetch_match_detail(client, match_id)
            if not match:
                continue

            for participant in match["info"]["participants"]:
                p_puuid = participant["puuid"]
                if p_puuid not in rank_cache:
                    rank_cache[p_puuid] = fetch_rank_by_puuid(client, p_puuid)
                participant["rankData"] = rank_cache[p_puuid]

            save_match(match_id, match)

        if i % save_interval == 0:
            save_cache(rank_cache)
            print(f"  Fortschritt: {len(seen_matches)} Matches gespeichert")

    save_cache(rank_cache)
    print(f"\n=== Done ===")
    print(f"Matches: {len(seen_matches)}")
    print(f"Spieler mit Rank-Daten: {sum(1 for v in rank_cache.values() if v)}/{len(rank_cache)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=10, help="Matches pro Spieler")
    parser.add_argument("--players", type=int, default=100, help="Spieler pro Division")
    parser.add_argument("--save-interval", type=int, default=10, help="Speicher-Intervall")
    args = parser.parse_args()

    crawl(matches_per_player=args.matches, players_per_division=args.players, save_interval=args.save_interval)
