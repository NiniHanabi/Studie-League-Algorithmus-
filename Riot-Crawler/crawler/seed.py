from crawler.client import RiotClient
from config import PLATFORM_URL


def get_seed_puuids(client: RiotClient, tiers: list[str] | None = None, count_per_tier: int = 100) -> list[str]:
    """
    Sammelt Seed-PUUIDs aus den angegebenen Tiers/Divisionen.
    Nutzt v4 League Endpoints für alle Divisionen.
    """
    if tiers is None:
        # Alle Divisionen von niedrig zu hoch
        tiers = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]

    puuids = []
    queue = "RANKED_SOLO_5x5"

    for tier in tiers:
        print(f"Fetching {tier} ladder...")

        # Master, Grandmaster, Challenger haben einen anderen Endpoint
        if tier in ["MASTER", "GRANDMASTER", "CHALLENGER"]:
            url = f"{PLATFORM_URL}/lol/league/v4/{tier.lower()}leagues/by-queue/{queue}"
            data = client.get(url)
            if not data:
                print(f"  > Keine Daten")
                continue

            entries = data.get("entries", [])[:count_per_tier]
            for entry in entries:
                puuid = entry.get("puuid")
                if puuid:
                    puuids.append(puuid)
            print(f"  > {len(entries)} PUUIDs aus {tier}")
        else:
            # Für Iron-Diamond: durch Ranks iterieren (I, II, III, IV)
            tier_count = 0
            for rank in ["I", "II", "III", "IV"]:
                url = f"{PLATFORM_URL}/lol/league/v4/entries/{queue}/{tier}/{rank}"

                # Pagination für niedrigere Divisionen (können viele Spieler sein)
                page = 1
                rank_count = 0
                per_page = count_per_tier // 4

                while rank_count < per_page:
                    data = client.get(url, params={"page": page})
                    if not data or len(data) == 0:
                        break

                    for entry in data[:per_page - rank_count]:
                        puuid = entry.get("puuid")
                        if puuid:
                            puuids.append(puuid)
                            rank_count += 1

                    page += 1

                if rank_count > 0:
                    print(f"  > {rank_count} PUUIDs aus {tier} {rank}")
                    tier_count += rank_count

    return puuids
