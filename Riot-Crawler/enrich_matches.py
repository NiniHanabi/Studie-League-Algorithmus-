"""
Reichert bereits gecrawlte Matches in data/matches_raw/ um zusätzliche Spieler-Infos an,
ohne neue Matches zu crawlen:
  - Summoner Level (Summoner-v4)
  - Champion Mastery für den in diesem Match gespielten Champion (Champion-Mastery-v4)
  - Flex-Queue-Rank (League-v4)

Nutzt Caches pro PUUID (bzw. PUUID+Champion), damit wiederkehrende Spieler
über mehrere Matches hinweg nicht erneut abgefragt werden.
"""
import argparse
import json
from pathlib import Path
from crawler.client import RiotClient
from crawler.enrichment_fetcher import (
    load_json_cache,
    save_json_cache,
    fetch_summoner_level,
    fetch_champion_mastery,
    fetch_flex_rank,
    SUMMONER_CACHE_FILE,
    MASTERY_CACHE_FILE,
    FLEX_CACHE_FILE,
)

RAW_DIR = Path("data/matches_raw")


def enrich(save_interval: int = 10):
    # Wichtig: nur EIN Key, da die PUUIDs in den bestehenden Matches an das Produkt
    # von RIOT_API_KEY gebunden sind und mit anderen Keys nicht entschlüsselt werden können.
    client = RiotClient()
    summoner_cache = load_json_cache(SUMMONER_CACHE_FILE)
    flex_cache = load_json_cache(FLEX_CACHE_FILE)
    mastery_cache = load_json_cache(MASTERY_CACHE_FILE)

    match_files = list(RAW_DIR.glob("*.json"))
    print(f"=== Anreicherung: {len(match_files)} Matches ===")

    for i, path in enumerate(match_files, 1):
        data = json.loads(path.read_text(encoding="utf-8"))

        for p in data["info"]["participants"]:
            puuid = p["puuid"]
            champion_id = p["championId"]
            mastery_key = f"{puuid}_{champion_id}"

            if puuid not in summoner_cache:
                summoner_cache[puuid] = fetch_summoner_level(client, puuid)
            if puuid not in flex_cache:
                flex_cache[puuid] = fetch_flex_rank(client, puuid)
            if mastery_key not in mastery_cache:
                mastery_cache[mastery_key] = fetch_champion_mastery(client, puuid, champion_id)

            p["summonerLevel"] = summoner_cache[puuid]
            p["flexRank"] = flex_cache[puuid]
            p["championMastery"] = mastery_cache[mastery_key]

        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        if i % save_interval == 0:
            save_json_cache(SUMMONER_CACHE_FILE, summoner_cache)
            save_json_cache(FLEX_CACHE_FILE, flex_cache)
            save_json_cache(MASTERY_CACHE_FILE, mastery_cache)
            print(f"  {i}/{len(match_files)} Matches angereichert")

    save_json_cache(SUMMONER_CACHE_FILE, summoner_cache)
    save_json_cache(FLEX_CACHE_FILE, flex_cache)
    save_json_cache(MASTERY_CACHE_FILE, mastery_cache)
    print(f"\n=== Done ===")
    print(f"Summoner Level: {len(summoner_cache)} Spieler")
    print(f"Flex-Rank: {len(flex_cache)} Spieler")
    print(f"Champion Mastery: {len(mastery_cache)} Spieler-Champion-Paare")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-interval", type=int, default=10, help="Speicher-Intervall (Matches)")
    args = parser.parse_args()

    enrich(save_interval=args.save_interval)
