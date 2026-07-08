from crawler.client import RiotClient
from config import PLATFORM_URL


def get_seed_players(client: RiotClient, tiers: list[str] | None = None, count_per_tier: int = 100) -> list[dict]:
    """
    Sammelt Seed-Spieler aus den angegebenen Tiers/Divisionen (I-IV) inkl. Rank-Daten.
    Nutzt v4 League Endpoints für alle Divisionen. Die Rank-Infos (LP, Wins, Losses)
    kommen direkt aus der gleichen Response, die auch die PUUIDs liefert.
    """
    if tiers is None:
        # Alle Divisionen von niedrig zu hoch
        tiers = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]

    players = []
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
                    players.append({
                        "puuid": puuid,
                        "tier": tier,
                        "rank": entry.get("rank"),
                        "leaguePoints": entry.get("leaguePoints"),
                        "wins": entry.get("wins"),
                        "losses": entry.get("losses"),
                    })
            print(f"  > {len(entries)} Spieler aus {tier}")
        else:
            # Für Iron-Diamond: durch Ranks iterieren (I, II, III, IV),
            # bis die gewünschte Anzahl für diese Division erreicht ist
            tier_count = 0
            for rank in ["I", "II", "III", "IV"]:
                if tier_count >= count_per_tier:
                    break

                url = f"{PLATFORM_URL}/lol/league/v4/entries/{queue}/{tier}/{rank}"

                # Pagination für niedrigere Divisionen (können viele Spieler sein)
                page = 1
                rank_count = 0
                needed = count_per_tier - tier_count

                while rank_count < needed:
                    data = client.get(url, params={"page": page})
                    if not data or len(data) == 0:
                        break

                    for entry in data[:needed - rank_count]:
                        puuid = entry.get("puuid")
                        if puuid:
                            players.append({
                                "puuid": puuid,
                                "tier": tier,
                                "rank": rank,
                                "leaguePoints": entry.get("leaguePoints"),
                                "wins": entry.get("wins"),
                                "losses": entry.get("losses"),
                            })
                            rank_count += 1

                    page += 1

                if rank_count > 0:
                    print(f"  > {rank_count} Spieler aus {tier} {rank}")
                    tier_count += rank_count

    return players
