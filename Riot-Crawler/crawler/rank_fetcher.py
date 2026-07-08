import json
from pathlib import Path
from crawler.client import RiotClient
from config import PLATFORM_URL

CACHE_FILE = Path("data/ranks_cache.json")


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_rank_by_puuid(client: RiotClient, puuid: str) -> dict | None:
    """
    Fetch rank für eine einzelne PUUID via individuellem API-Call.
    Genutzt für Match-Teilnehmer, die nicht schon im Rank-Cache stehen (z.B. keine Seed-Spieler).
    """
    url = f"{PLATFORM_URL}/lol/league/v4/entries/by-puuid/{puuid}"
    data = client.get(url)
    if not data:
        return None

    for entry in data:
        if entry.get("queueType") == "RANKED_SOLO_5x5":
            return {
                "tier": entry.get("tier"),
                "rank": entry.get("rank"),
                "leaguePoints": entry.get("leaguePoints"),
                "wins": entry.get("wins"),
                "losses": entry.get("losses"),
            }
    return None
