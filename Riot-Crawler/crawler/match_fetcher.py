import json
from pathlib import Path
from crawler.client import RiotClient
from config import REGION_URL, QUEUE

RAW_DIR = Path("data/matches_raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def get_match_ids(client: RiotClient, puuid: str, count: int = 20) -> list[str]:
    url = f"{REGION_URL}/lol/match/v5/matches/by-puuid/{puuid}/ids"
    result = client.get(url, params={"queue": QUEUE, "count": count, "type": "ranked"})
    return result or []


def get_match_detail(client: RiotClient, match_id: str) -> dict | None:
    path = RAW_DIR / f"{match_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    url = f"{REGION_URL}/lol/match/v5/matches/{match_id}"
    data = client.get(url)
    if data:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data
