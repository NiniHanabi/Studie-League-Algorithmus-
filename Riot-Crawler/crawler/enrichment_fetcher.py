import json
from pathlib import Path
from crawler.client import RiotClient
from config import PLATFORM_URL

SUMMONER_CACHE_FILE = Path("data/summoner_cache.json")
MASTERY_CACHE_FILE = Path("data/mastery_cache.json")
FLEX_CACHE_FILE = Path("data/flex_cache.json")


def load_json_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_json_cache(path: Path, cache: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_summoner_level(client: RiotClient, puuid: str) -> int | None:
    url = f"{PLATFORM_URL}/lol/summoner/v4/summoners/by-puuid/{puuid}"
    data = client.get(url)
    return data.get("summonerLevel") if data else None


def fetch_champion_mastery(client: RiotClient, puuid: str, champion_id: int) -> dict | None:
    url = f"{PLATFORM_URL}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/by-champion/{champion_id}"
    data = client.get(url)
    if not data:
        return None
    return {
        "championLevel": data.get("championLevel"),
        "championPoints": data.get("championPoints"),
    }


def fetch_flex_rank(client: RiotClient, puuid: str) -> dict | None:
    """Flex-Queue-Rank (RANKED_FLEX_SR) über denselben Endpoint wie Solo-Rank-Enrichment."""
    url = f"{PLATFORM_URL}/lol/league/v4/entries/by-puuid/{puuid}"
    data = client.get(url)
    if not data:
        return None

    for entry in data:
        if entry.get("queueType") == "RANKED_FLEX_SR":
            return {
                "tier": entry.get("tier"),
                "rank": entry.get("rank"),
                "leaguePoints": entry.get("leaguePoints"),
                "wins": entry.get("wins"),
                "losses": entry.get("losses"),
            }
    return None
