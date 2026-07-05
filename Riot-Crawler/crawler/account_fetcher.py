import json
from pathlib import Path
from crawler.client import RiotClient
from config import REGION_URL

ACCOUNT_CACHE_FILE = Path("data/account_cache.json")


def load_account_cache() -> dict:
    if ACCOUNT_CACHE_FILE.exists():
        return json.loads(ACCOUNT_CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def save_account_cache(cache: dict):
    ACCOUNT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACCOUNT_CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_account_info(client: RiotClient, puuid: str, cache: dict) -> dict | None:
    """Fetch gameName and tagLine for a PUUID. Uses cache to avoid API calls."""
    if puuid in cache:
        return cache[puuid]

    url = f"{REGION_URL}/riot/account/v1/accounts/by-puuid/{puuid}"
    data = client.get(url)
    if data:
        account_info = {
            "gameName": data.get("gameName"),
            "tagLine": data.get("tagLine"),
        }
        cache[puuid] = account_info
        return account_info
    return None
