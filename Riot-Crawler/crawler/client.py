import time
import requests
from collections import deque
from config import API_KEY, RATE_LIMIT_PER_SECOND, RATE_LIMIT_PER_2MIN


class RiotClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"X-Riot-Token": API_KEY})
        self._timestamps: deque[float] = deque()

    def _enforce_rate_limits(self):
        now = time.time()

        # Purge timestamps older than 2 minutes
        while self._timestamps and now - self._timestamps[0] > 120:
            self._timestamps.popleft()

        # 2-minute window full → wait until oldest drops out
        if len(self._timestamps) >= RATE_LIMIT_PER_2MIN:
            wait = 120 - (now - self._timestamps[0]) + 0.1
            print(f"  [rate limit] 2-min window full, sleeping {wait:.1f}s")
            time.sleep(wait)

        # Per-second window: count requests in last 1s
        cutoff = time.time() - 1.0
        recent = sum(1 for t in self._timestamps if t > cutoff)
        if recent >= RATE_LIMIT_PER_SECOND:
            time.sleep(0.1)

        self._timestamps.append(time.time())

    def get(self, url: str, params: dict | None = None) -> dict | list | None:
        self._enforce_rate_limits()
        resp = self.session.get(url, params=params)

        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 10))
            print(f"  [429] Rate limited by server. Waiting {retry_after}s...")
            time.sleep(retry_after)
            return self.get(url, params)

        if resp.status_code in (403, 404):
            return None

        resp.raise_for_status()
        return resp.json()
