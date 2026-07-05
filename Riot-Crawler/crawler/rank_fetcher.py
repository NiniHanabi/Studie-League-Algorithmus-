import json
from pathlib import Path
from crawler.client import RiotClient
from config import PLATFORM_URL

CACHE_FILE = Path("data/ranks_cache.json")
LADDER_CACHE_FILE = Path("data/ladder_cache.json")


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def load_ladder_cache() -> dict:
    """Load cached ladder data (tier → list of players with rank)."""
    if LADDER_CACHE_FILE.exists():
        return json.loads(LADDER_CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def save_ladder_cache(cache: dict):
    LADDER_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LADDER_CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def fetch_all_ladder_data(client: RiotClient) -> dict:
    """
    Fetch ladder data for all tiers/divisions. Returns dict keyed by PUUID,
    with tier/rank/LP data. Gracefully handles timeouts.
    """
    ladder_by_puuid = {}
    queue = "RANKED_SOLO_5x5"

    # Master+ Tiers
    master_plus_tiers = ["challenger", "grandmaster", "master"]
    tier_names = {"challenger": "CHALLENGER", "grandmaster": "GRANDMASTER", "master": "MASTER"}

    for tier in master_plus_tiers:
        print(f"  Fetching {tier} ladder...")
        try:
            url = f"{PLATFORM_URL}/lol/league/v4/{tier}leagues/by-queue/{queue}"
            data = client.get(url)
            if not data:
                print(f"    (skip: no data)")
                continue

            tier_name = tier_names[tier]
            for entry in data.get("entries", []):
                puuid = entry.get("puuid")
                if puuid:
                    ladder_by_puuid[puuid] = {
                        "tier": tier_name,
                        "rank": entry.get("rank"),
                        "leaguePoints": entry.get("leaguePoints"),
                        "wins": entry.get("wins"),
                        "losses": entry.get("losses"),
                    }
            count = len([p for p in ladder_by_puuid.values() if p['tier'] == tier_name])
            print(f"    ({count} players)")
        except Exception as e:
            print(f"    (skip: {str(e)[:50]})")
            continue

    # Iron-Diamond Tiers (mit Ranks I-IV und Pagination)
    other_tiers = ["DIAMOND", "EMERALD", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON"]

    for tier in other_tiers:
        print(f"  Fetching {tier} ladder...")
        tier_count = 0
        try:
            for rank in ["I", "II", "III", "IV"]:
                page = 1
                while True:
                    url = f"{PLATFORM_URL}/lol/league/v4/entries/{queue}/{tier}/{rank}"
                    data = client.get(url, params={"page": page})

                    if not data or len(data) == 0:
                        break

                    for entry in data:
                        puuid = entry.get("puuid")
                        if puuid:
                            ladder_by_puuid[puuid] = {
                                "tier": tier,
                                "rank": rank,
                                "leaguePoints": entry.get("leaguePoints"),
                                "wins": entry.get("wins"),
                                "losses": entry.get("losses"),
                            }
                            tier_count += 1

                    page += 1

                    # Limit to 3 pages per rank to avoid API spam
                    if page > 3:
                        break

            print(f"    ({tier_count} players)")
        except Exception as e:
            print(f"    (skip: {str(e)[:50]})")
            continue

    print(f"  Cached {len(ladder_by_puuid)} players from ladder")
    return ladder_by_puuid


def get_rank_from_ladder(puuid: str, ladder_data: dict) -> dict | None:
    """Lookup rank from cached ladder data."""
    return ladder_data.get(puuid)


def fetch_rank_by_puuid(client: RiotClient, puuid: str) -> dict | None:
    """
    Fetch rank for a single PUUID via individual API call.
    Genutzt für Spieler, die nicht in der Ladder-Cache sind.
    """
    url = f"{PLATFORM_URL}/lol/league/v4/entries/by-puuid/{puuid}"
    data = client.get(url)
    if not data:
        return None

    # Suche Ranked Solo/Duo Eintrag
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
