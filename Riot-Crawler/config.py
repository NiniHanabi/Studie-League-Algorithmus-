import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["RIOT_API_KEY"]
# Optional: weitere Dev-Keys für RiotClientPool (RIOT_API_KEY_2, RIOT_API_KEY_3, ...)
API_KEYS = [API_KEY] + [
    v for i in range(2, 10) if (v := os.getenv(f"RIOT_API_KEY_{i}"))
]
PLATFORM = os.getenv("PLATFORM", "euw1")
REGION = os.getenv("REGION", "europe")
QUEUE = 420  # Ranked Solo/Duo

PLATFORM_URL = f"https://{PLATFORM}.api.riotgames.com"
REGION_URL = f"https://{REGION}.api.riotgames.com"

# Development key: 20 req/s, 100 req/2min
RATE_LIMIT_PER_SECOND = 20
RATE_LIMIT_PER_2MIN = 100
