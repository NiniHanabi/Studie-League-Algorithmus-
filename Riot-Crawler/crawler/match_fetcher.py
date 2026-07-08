import json
from pathlib import Path
from crawler.client import RiotClient
from config import REGION_URL, QUEUE

RAW_DIR = Path("data/matches_raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def clear_raw_dir():
    """Löscht alle vorher gefetchten Match-Dateien, damit nur der aktuelle Run übrig bleibt."""
    for path in RAW_DIR.glob("*.json"):
        path.unlink()


def get_match_ids(client: RiotClient, puuid: str, count: int = 20) -> list[str]:
    url = f"{REGION_URL}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    result = client.get(url, params={"queue": QUEUE, "count": count, "type": "ranked"})
    return result or []


def fetch_match_detail(client: RiotClient, match_id: str) -> dict | None:
    url = f"{REGION_URL}/lol/match/v5/matches/{match_id}"
    return client.get(url)


def save_match(match_id: str, data: dict):
    path = RAW_DIR / f"{match_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
