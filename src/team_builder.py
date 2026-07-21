"""
Team-Builder aus einer Spieler-CSV.

Liest eine CSV mit Spielern (RIOT ID, Main Position, Second Positions), holt fuer
jeden Spieler ueber die Riot API seinen Rang und berechnet die individuelle
Staerke mit den empirischen Gewichten (weights.json). Anschliessend werden zwei
Aufstellungen gebildet, bei denen JEDER Spieler eine seiner angegebenen Rollen
erhaelt:

  1. Staerkste Teams  - die leistungsstaerksten Spieler werden gebuendelt.
  2. Ausgeglichene Teams - die Teamstaerken werden moeglichst angeglichen.

CSV-Format (Semikolon oder Komma als Trenner, Kopfzeile erforderlich):
  RIOT ID;Main Position;Second Positions
  Faker#KR1;Mid;Top,Jungle
  ...
- RIOT ID: Name#Tagline
- Main Position: eine Rolle (Top/Jungle/Mid/Bot/Support bzw. Synonyme)
- Second Positions: 0..n weitere Rollen, mit Komma getrennt

Aufruf:  python team_builder.py <pfad/zur/spieler.csv>

Voraussetzung: gueltiger Riot API Key (RIOT_API_KEY, Riot-Crawler/.env oder riot.txt)
und weights.json (wird bei Bedarf ueber regression.py erzeugt).
"""
import math
import os
import random
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from scipy.optimize import linear_sum_assignment

from data_prep import rank_component
from main import load_weights, compute_individual_strength

# Region fest auf EUW (konsistent zur Studie). Bei Bedarf ueber Umgebungsvariablen.
REGION = os.getenv("REGION", "europe")     # Account-v1 Routing
PLATFORM = os.getenv("PLATFORM", "euw1")   # League-v4 Routing

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
# Anzeige-Labels in der Konvention der Eingabe: Top / Jungle / Mid / ADC / Support
ROLE_LABEL = {"TOP": "Top", "JUNGLE": "Jungle", "MIDDLE": "Mid",
              "BOTTOM": "ADC", "UTILITY": "Support"}
ROLE_ALIASES = {
    "top": "TOP", "toplane": "TOP", "oben": "TOP",
    "jungle": "JUNGLE", "jgl": "JUNGLE", "jg": "JUNGLE", "jungler": "JUNGLE",
    "mid": "MIDDLE", "middle": "MIDDLE", "midlane": "MIDDLE", "mitte": "MIDDLE",
    "bot": "BOTTOM", "bottom": "BOTTOM", "adc": "BOTTOM", "ad": "BOTTOM",
    "adcarry": "BOTTOM", "marksman": "BOTTOM", "botlane": "BOTTOM",
    "support": "UTILITY", "supp": "UTILITY", "sup": "UTILITY",
    "utility": "UTILITY", "sup port": "UTILITY", "heal": "UTILITY",
}


# ---------------------------------------------------------------------------
# Riot API
# ---------------------------------------------------------------------------
def load_api_key():
    key = os.getenv("RIOT_API_KEY")
    if key:
        return key.strip()
    root = Path(__file__).resolve().parent.parent
    env_path = root / "Riot-Crawler" / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("RIOT_API_KEY") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    riot_txt = root / "riot.txt"
    if riot_txt.exists() and riot_txt.read_text().strip():
        return riot_txt.read_text().strip()
    raise RuntimeError(
        "Kein Riot API Key gefunden. Setze RIOT_API_KEY, lege ihn in "
        "Riot-Crawler/.env ab oder in riot.txt.")


class RiotAPI:
    """Minimaler Client fuer die zwei benoetigten Endpunkte."""

    def __init__(self, api_key):
        self.s = requests.Session()
        self.s.headers.update({"X-Riot-Token": api_key})

    def _get(self, url):
        for _ in range(6):
            r = self.s.get(url, timeout=15)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 5))
                print(f"  [429] Rate-Limit, warte {wait}s ...")
                time.sleep(wait)
                continue
            if r.status_code in (400, 403, 404):
                return None
            r.raise_for_status()
            time.sleep(0.05)  # sanftes Drosseln
            return r.json()
        return None

    def puuid(self, name, tag):
        url = (f"https://{REGION}.api.riotgames.com/riot/account/v1/accounts/"
               f"by-riot-id/{quote(name)}/{quote(tag)}")
        data = self._get(url)
        return data.get("puuid") if data else None

    def solo_rank(self, puuid):
        url = (f"https://{PLATFORM}.api.riotgames.com/lol/league/v4/entries/"
               f"by-puuid/{puuid}")
        data = self._get(url)
        if not data:
            return None
        for e in data:
            if e.get("queueType") == "RANKED_SOLO_5x5":
                return e
        return None


# ---------------------------------------------------------------------------
# CSV einlesen und Rollen normalisieren
# ---------------------------------------------------------------------------
def normalize_role(text):
    if text is None:
        return None
    key = str(text).strip().lower()
    return ROLE_ALIASES.get(key)


def _find_col(cols, *candidates):
    low = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def read_players_csv(path):
    # Trenner automatisch erkennen (Semikolon oder Komma); Kodierung robust
    # nacheinander probieren (Windows-Excel speichert oft cp1252 statt UTF-8).
    df = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, sep=None, engine="python", dtype=str,
                             encoding=enc).fillna("")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if df is None:
        raise ValueError(f"CSV {path} konnte mit keiner gängigen Kodierung "
                         "(UTF-8, cp1252, Latin-1) gelesen werden.")
    c_id = _find_col(df.columns, "riot id", "riotid", "riot-id")
    c_main = _find_col(df.columns, "main position", "main", "mainposition")
    c_sec = _find_col(df.columns, "second positions", "second position",
                      "secondary positions", "seconds")
    if not c_id or not c_main:
        raise ValueError("CSV benoetigt mindestens die Spalten 'RIOT ID' und "
                         "'Main Position'.")

    players, problems = [], []
    for _, row in df.iterrows():
        riot_id = row[c_id].strip()
        if not riot_id:
            continue
        if "#" not in riot_id:
            problems.append(f"'{riot_id}' hat kein '#Tagline'.")
            continue
        name, tag = riot_id.rsplit("#", 1)
        main = normalize_role(row[c_main])
        if not main:
            problems.append(f"{riot_id}: Main-Rolle '{row[c_main]}' nicht erkannt.")
            continue
        seconds = []
        if c_sec and row[c_sec].strip():
            for part in row[c_sec].split(","):
                r = normalize_role(part)
                if r and r != main and r not in seconds:
                    seconds.append(r)
        allowed = [main] + seconds
        players.append({"riot_id": riot_id, "name": name.strip(), "tag": tag.strip(),
                        "main": main, "seconds": seconds, "allowed": set(allowed)})
    return players, problems


# ---------------------------------------------------------------------------
# Staerke je Spieler ueber die API + Modellgewichte
# ---------------------------------------------------------------------------
def enrich_strength(players, api, model):
    rows = []
    for p in players:
        puuid = api.puuid(p["name"], p["tag"])
        rank = api.solo_rank(puuid) if puuid else None
        if rank:
            tier, div, lp = rank["tier"], rank["rank"], rank["leaguePoints"]
            wins, losses = rank.get("wins", 0), rank.get("losses", 0)
            total = wins + losses
            rows.append({
                "riot_id": p["riot_id"],
                "rank_score": rank_component(tier, div),
                "lp": float(lp),
                "win_rate": (wins / total) if total else np.nan,
                "tier": tier, "division": div, "ranked": True,
            })
            print(f"  {p['riot_id']:<28} {tier} {div} {lp} LP")
        else:
            rows.append({"riot_id": p["riot_id"], "rank_score": np.nan, "lp": np.nan,
                         "win_rate": np.nan, "tier": "UNRANKED", "division": "",
                         "ranked": False})
            print(f"  {p['riot_id']:<28} kein Solo/Duo-Rang gefunden")

    df = pd.DataFrame(rows)
    # Modell-Staerke; fehlende Werte werden vom Modell mit dem Trainings-Median gefuellt
    df = compute_individual_strength(df, model)
    strength = dict(zip(df["riot_id"], df["individual_strength"]))
    meta = {r["riot_id"]: r for r in rows}
    for p in players:
        p["strength"] = float(strength[p["riot_id"]])
        p["tier"] = meta[p["riot_id"]]["tier"]
        p["division"] = meta[p["riot_id"]]["division"]
        p["ranked"] = meta[p["riot_id"]]["ranked"]
    return players


# ---------------------------------------------------------------------------
# Rollen-beschraenkte Zuordnung
# ---------------------------------------------------------------------------
def assign_roles(players, k):
    """Weist jedem Spieler eine seiner erlaubten Rollen zu, sodass jede Rolle
    genau k-mal vergeben wird. Minimiert die Zahl der Spieler abseits ihrer
    Main-Rolle (Main = Kosten 0, Zweitrolle = Kosten 1, unmoeglich = gesperrt).
    """
    n = len(players)
    LARGE = 1e6
    slots = [r for r in ROLES for _ in range(k)]  # n Slots (k je Rolle)
    cost = np.full((n, n), LARGE)
    for i, p in enumerate(players):
        for j, r in enumerate(slots):
            if r == p["main"]:
                cost[i, j] = 0
            elif r in p["seconds"]:
                cost[i, j] = 1
    rows, cols = linear_sum_assignment(cost)

    role_of = [None] * n
    infeasible, off_main = [], 0
    for i, j in zip(rows, cols):
        if cost[i, j] >= LARGE:
            infeasible.append(players[i])
        role_of[i] = slots[j]
        if cost[i, j] == 1:
            off_main += 1
    return role_of, off_main, infeasible


def _role_groups(players, role_of):
    return {r: [i for i in range(len(players)) if role_of[i] == r] for r in ROLES}


def team_mean(team, players):
    return sum(players[i]["strength"] for i in team.values()) / len(team)


def build_strongest(players, role_of, k):
    """Buendelt Staerke: staerkster Spieler je Rolle -> Team 1, usw."""
    groups = _role_groups(players, role_of)
    for r in ROLES:
        groups[r].sort(key=lambda i: players[i]["strength"], reverse=True)
    return [{r: groups[r][t] for r in ROLES} for t in range(k)]


def build_balanced(players, role_of, k, iters=30000, seed=42):
    """Gleicht die Teamstaerken an. Simulated Annealing ueber rollen-erhaltende
    Tausche (zwei Spieler derselben Rolle tauschen die Teams) - dadurch bleibt
    jedes Team automatisch vollstaendig und rollen-gueltig.
    """
    groups = _role_groups(players, role_of)
    # Start: je Rolle nach Staerke sortiert, Schlangen-Verteilung fuer guten Start
    assign = {}
    for r in ROLES:
        g = sorted(groups[r], key=lambda i: players[i]["strength"], reverse=True)
        snake = [None] * k
        direction, pos = 1, 0
        for i in g:
            snake[pos] = i
            pos += direction
            if pos == k:
                pos, direction = k - 1, -1
            elif pos < 0:
                pos, direction = 0, 1
        assign[r] = snake

    if k < 2:
        return [{r: assign[r][0] for r in ROLES}]

    def team_strengths(a):
        return [sum(players[a[r][t]]["strength"] for r in ROLES) for t in range(k)]

    def spread(a):
        ts = team_strengths(a)
        m = sum(ts) / k
        return sum((x - m) ** 2 for x in ts)

    rng = random.Random(seed)
    cur = {r: assign[r][:] for r in ROLES}
    cur_c = spread(cur)
    best, best_c = {r: cur[r][:] for r in ROLES}, cur_c
    T = 1.0
    for _ in range(iters):
        r = rng.choice(ROLES)
        t1, t2 = rng.sample(range(k), 2)
        cur[r][t1], cur[r][t2] = cur[r][t2], cur[r][t1]
        c = spread(cur)
        if c <= cur_c or rng.random() < math.exp((cur_c - c) / max(T, 1e-9)):
            cur_c = c
            if c < best_c:
                best_c, best = c, {rr: cur[rr][:] for rr in ROLES}
        else:
            cur[r][t1], cur[r][t2] = cur[r][t2], cur[r][t1]  # zuruecktauschen
        T *= 0.9997
    return [{r: best[r][t] for r in ROLES} for t in range(k)]


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------
def print_teams(title, teams, players):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for t, team in enumerate(teams, 1):
        print(f"\nTeam {t}  (Ø Stärke {team_mean(team, players):+.3f})")
        for r in ROLES:
            p = players[team[r]]
            off = "" if p["main"] == r else "  (Zweitrolle)"
            rank = f"{p['tier']} {p['division']}".strip()
            print(f"   {ROLE_LABEL[r]:<8} {p['riot_id']:<26} {rank:<16}"
                  f"Stärke {p['strength']:+.3f}{off}")
    means = [team_mean(t, players) for t in teams]
    if len(means) > 1:
        print(f"\n  Spanne der Team-Ø-Stärke: {max(means) - min(means):.3f} "
              f"(kleiner = ausgeglichener)")


def teams_to_dataframe(teams, players, mode):
    rec = []
    for t, team in enumerate(teams, 1):
        for r in ROLES:
            p = players[team[r]]
            rec.append({"modus": mode, "team": t, "rolle": ROLE_LABEL[r],
                        "riot_id": p["riot_id"], "rang": f"{p['tier']} {p['division']}".strip(),
                        "main_rolle": p["main"] == r, "staerke": round(p["strength"], 4)})
    return pd.DataFrame(rec)


# ---------------------------------------------------------------------------
# Ablauf
# ---------------------------------------------------------------------------
def run(csv_path):
    csv_path = Path(csv_path)
    players, problems = read_players_csv(csv_path)
    for msg in problems:
        print(f"[uebersprungen] {msg}")

    n = len(players)
    if n < 5 or n % 5 != 0:
        raise SystemExit(f"Es werden Vielfache von 5 Spielern benoetigt "
                         f"(gefunden: {n}). Bitte CSV anpassen.")
    k = n // 5
    print(f"\n{n} Spieler -> {k} Team(s) à 5. Hole Ränge über die Riot API "
          f"(Region {PLATFORM}) ...")

    model = load_weights()
    api = RiotAPI(load_api_key())
    players = enrich_strength(players, api, model)

    unranked = [p["riot_id"] for p in players if not p["ranked"]]
    if unranked:
        print(f"\n[Hinweis] Ohne Solo/Duo-Rang (Stärke = Modell-Median geschätzt): "
              f"{', '.join(unranked)}")

    role_of, off_main, infeasible = assign_roles(players, k)
    if infeasible:
        namen = ", ".join(p["riot_id"] for p in infeasible)
        raise SystemExit(
            "Keine gültige Rollenverteilung möglich - zu viele Spieler wollen "
            f"dieselben Rollen. Betroffen: {namen}. Bitte Zweitrollen ergänzen.")
    print(f"\nRollenverteilung gefunden: {n - off_main} auf Main-Rolle, "
          f"{off_main} auf einer Zweitrolle.")

    strongest = build_strongest(players, role_of, k)
    balanced = build_balanced(players, role_of, k)

    print_teams("STÄRKSTE TEAMS (Stärke gebündelt)", strongest, players)
    print_teams("AUSGEGLICHENE TEAMS (Stärke angeglichen)", balanced, players)

    out_s = csv_path.with_name(csv_path.stem + "_staerkste_teams.csv")
    out_b = csv_path.with_name(csv_path.stem + "_ausgeglichene_teams.csv")
    teams_to_dataframe(strongest, players, "staerkste").to_csv(
        out_s, index=False, sep=";", encoding="utf-8-sig")
    teams_to_dataframe(balanced, players, "ausgeglichen").to_csv(
        out_b, index=False, sep=";", encoding="utf-8-sig")
    print(f"\nGespeichert: {out_s.name}  und  {out_b.name}")


if __name__ == "__main__":
    # Windows-Konsole auf UTF-8, damit Umlaute korrekt erscheinen
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) != 2:
        print("Aufruf: python team_builder.py <pfad/zur/spieler.csv>")
        raise SystemExit(1)
    run(sys.argv[1])
