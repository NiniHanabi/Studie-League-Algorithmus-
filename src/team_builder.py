"""
Team-Builder aus einer Spieler-CSV.

Liest eine CSV mit Spielern (RIOT ID, Main Position, Second Positions), holt fuer
jeden Spieler ueber die Riot API seinen Rang und berechnet die individuelle
Staerke mit den empirischen Gewichten (weights.json). Anschliessend werden zwei
Aufstellungen gebildet, die alle Spieler-Praeferenzen erfuellen und jedem Spieler
nach Moeglichkeit eine seiner angegebenen Rollen geben:

  1. Staerkste Teams  - die leistungsstaerksten Spieler werden gebuendelt.
  2. Ausgeglichene Teams - die Teamstaerken werden moeglichst angeglichen.

Beide Modi werden zweimal geloest: heuristisch (Rollen fest, Sortierung bzw.
Simulated Annealing) und - falls gurobipy verfuegbar - exakt mit Gurobi. Gurobi
waehlt dabei Rolle UND Team gemeinsam (bei gleicher Zahl an Zweitrollen-Einsaetzen).

Ist die Spielerzahl kein Vielfaches von 5, werden die ueberzaehligen Spieler
Ersatz: so gewaehlt, dass die Prioritaeten unten bestmoeglich erfuellt sind; bei
Gleichstand sitzen die schwaechsten Spieler aus.

CSV-Format (Semikolon oder Komma als Trenner, Kopfzeile erforderlich):
  RIOT ID;Main Position;Second Positions
  Faker#KR1;Mid;Top,Jungle
  ...
- RIOT ID: Name#Tagline
- Main Position: eine Rolle (Top/Jungle/Mid/Bot/Support bzw. Synonyme)
- Second Positions: 0..n weitere Rollen, mit Komma getrennt
- Mit / With (optional): RIOT IDs, mit denen der Spieler in EIN Team will
  (Komma-getrennt)
- Nicht mit / Not with (optional): RIOT IDs, mit denen er NICHT in ein Team soll
  (Spaltennamen: Gross/klein, '_' und '-' egal, z. B. auch 'not_with')

Prioritaeten (lexikographisch):
  1. Spieler-Praeferenzen (Mit / Nicht mit, gelten in beide Richtungen) sind HART:
     sie werden zu 100 % erfuellt. 'Mit' heisst: beide spielen im selben Team
     (keiner ist Ersatz). Dafuer darf ein Spieler notfalls auch eine Rolle spielen,
     die er nicht angegeben hat. Sind die Praeferenzen logisch unerfuellbar
     (z. B. 'Mit'-Gruppe > 5 Spieler, Widerspruch zu 'Nicht mit'), bricht das
     Programm mit einer Erklaerung ab.
  2. Moeglichst wenige Spieler auf einer nicht angegebenen Rolle.
  3. Moeglichst wenige Spieler auf ihrer Zweitrolle.
  4. Ziel des Modus (staerkste bzw. ausgeglichene Teams).

Aufruf:  python team_builder.py <pfad/zur/spieler.csv> [gurobi-zeitlimit-s]

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

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except Exception:
    HAS_GUROBI = False

GRB_STATUS = {2: "optimal", 9: "Zeitlimit", 11: "unterbrochen",
              3: "unzulaessig", 13: "suboptimal"}

# Strafe je verletzter Spieler-Praeferenz in der SA des ausgeglichenen Modus, falls
# keine praeferenz-erfuellende Startaufstellung vorliegt (sonst: harte Regel).
PREF_WEIGHT = 1000.0
# Rollen-Kosten einer NICHT angegebenen Rolle (Main = 0, Zweitrolle = 1). Wird nur
# genutzt, wenn sonst eine Spieler-Praeferenz verletzt wuerde.
W_FOREIGN = 100

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


def _col_key(name):
    """Spaltenname normalisieren: Gross/klein, '_' / '-' und Mehrfach-Leerzeichen
    egal ('Not_With', 'not-with', 'Not  With' -> 'not with')."""
    return " ".join(str(name).lower().replace("_", " ").replace("-", " ").split())


def _find_col(cols, *candidates):
    low = {_col_key(c): c for c in cols}
    for cand in candidates:
        if _col_key(cand) in low:
            return low[_col_key(cand)]
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
    c_with = _find_col(df.columns, "mit", "zusammen mit", "duo", "with", "play with")
    c_not = _find_col(df.columns, "nicht mit", "nicht zusammen mit", "ohne", "not with",
                      "notwith", "without", "not play with")
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
        wants = {key: [s.strip() for s in row[col].split(",") if s.strip()] if col else []
                 for key, col in (("with", c_with), ("not_with", c_not))}
        players.append({"riot_id": riot_id, "name": name.strip(), "tag": tag.strip(),
                        "main": main, "seconds": seconds, "allowed": set(allowed),
                        **wants})
    return players, problems


def _id_key(riot_id):
    """Vergleichsschluessel fuer RIOT IDs: Gross/klein und Leerzeichen egal."""
    return "".join(riot_id.split()).lower()


def build_preferences(players):
    """Loest die Spalten 'Mit' / 'Nicht mit' in Spielerpaare (i, j), i < j, auf.
    Praeferenzen gelten symmetrisch; widerspruechliche Paare werden verworfen."""
    index = {_id_key(p["riot_id"]): i for i, p in enumerate(players)}
    pairs = {"with": set(), "not_with": set()}
    problems = []
    for i, p in enumerate(players):
        for key in pairs:
            for other in p.get(key, []):
                j = index.get(_id_key(other))
                if j is None:
                    problems.append(f"{p['riot_id']}: Präferenz '{other}' ist kein "
                                    "Spieler der Liste.")
                elif j != i:
                    pairs[key].add((min(i, j), max(i, j)))
    for i, j in pairs["with"] & pairs["not_with"]:
        problems.append(f"{players[i]['riot_id']} / {players[j]['riot_id']}: sowohl "
                        "'Mit' als auch 'Nicht mit' - Präferenz ignoriert.")
    conflict = pairs["with"] & pairs["not_with"]
    return sorted(pairs["with"] - conflict), sorted(pairs["not_with"] - conflict), problems


def pref_violations(teams, together, apart):
    """Liste der verletzten Praeferenzen [(i, j, 'mit'|'nicht mit'), ...].
    Wer in keinem Team steht, ist Ersatz: 'Mit' ist nur erfuellt, wenn beide im
    selben Team spielen; 'Nicht mit' ist mit einem Ersatzspieler immer erfuellt."""
    team_of = {i: t for t, team in enumerate(teams) for i in team.values()}
    bad = []
    for i, j in together:
        if i not in team_of or team_of.get(i) != team_of.get(j):
            bad.append((i, j, "mit"))
    for i, j in apart:
        if i in team_of and j in team_of and team_of[i] == team_of[j]:
            bad.append((i, j, "nicht mit"))
    return bad


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
            wr = (wins / total) if total else np.nan
            rows.append({
                "riot_id": p["riot_id"],
                "rank_score": rank_component(tier, div),
                "lp": float(lp),
                "win_rate": wr,
                "tier": tier, "division": div, "ranked": True,
            })
            wr_txt = f"{wr*100:.0f}% WR ({total} Sp.)" if total else "keine Spiele"
            print(f"  {p['riot_id']:<28} {tier} {div} {lp} LP · {wr_txt}")
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
        p["lp"] = meta[p["riot_id"]]["lp"]
        p["win_rate"] = meta[p["riot_id"]]["win_rate"]
        p["ranked"] = meta[p["riot_id"]]["ranked"]
    return players


# ---------------------------------------------------------------------------
# Rollen-beschraenkte Zuordnung
# ---------------------------------------------------------------------------
def assign_roles(players, k):
    """Weist jedem Spieler eine seiner erlaubten Rollen zu, sodass jede Rolle
    genau k-mal vergeben wird. Minimiert die Zahl der Spieler abseits ihrer
    Main-Rolle (Main = Kosten 0, Zweitrolle = Kosten 1, unmoeglich = gesperrt).
    Ueberzaehlige Spieler (n - 5k) werden Ersatz (role_of = None); deren Kosten
    sind so klein (Summe < 1), dass sie nur bei Gleichstand entscheiden: dann
    sitzt der schwaechere Spieler aus.
    """
    n = len(players)
    n_res = n - 5 * k
    LARGE = 1e6
    slots = [r for r in ROLES for _ in range(k)] + [None] * n_res
    cost = np.full((n, n), LARGE)
    for i, p in enumerate(players):
        for j, r in enumerate(slots):
            if r == p["main"]:
                cost[i, j] = 0
            elif r in p["seconds"]:
                cost[i, j] = 1
    by_strength = sorted(range(n), key=lambda i: players[i].get("strength", 0.0))
    for rank, i in enumerate(by_strength):
        cost[i, 5 * k:] = (rank + 1) / (n * (n_res + 1))
    rows, cols = linear_sum_assignment(cost)

    role_of = [None] * n
    infeasible, off_main = [], 0
    for i, j in zip(rows, cols):
        if cost[i, j] >= LARGE:
            infeasible.append(players[i])
        role_of[i] = slots[j]
        if slots[j] is not None and cost[i, j] == 1:
            off_main += 1
    return role_of, off_main, infeasible


def role_cost(p, r):
    """Kosten einer Rolle fuer Spieler p: Main 0, Zweitrolle 1, nicht angegeben
    W_FOREIGN (nur zulaessig, wenn sonst eine Praeferenz verletzt wuerde)."""
    if r == p["main"]:
        return 0
    if r in p["seconds"]:
        return 1
    return W_FOREIGN


def role_counts(teams, players):
    """(Spieler auf nicht angegebener Rolle, Spieler auf Zweitrolle)."""
    costs = [role_cost(players[i], r) for tm in teams for r, i in tm.items()]
    return costs.count(W_FOREIGN), costs.count(1)


def pack_preferences(players, k, together, apart):
    """Erste Aufstellung, die ALLE Praeferenzen erfuellt (harte Bedingung).

    'Mit'-Paare werden transitiv zu Gruppen zusammengefasst; jede Gruppe spielt
    komplett in einem Team. Gruppen werden per Backtracking so auf die Teams
    verteilt, dass kein 'Nicht mit'-Paar zusammen landet; Einzelspieler duerfen
    auf die Ersatzbank. Die Rollen je Team vergibt dann die Hungarian-Methode
    (Main vor Zweitrolle vor nicht angegebener Rolle).
    Bricht mit einer Erklaerung ab, wenn die Praeferenzen unerfuellbar sind.
    """
    n = len(players)
    n_res = n - 5 * k
    name = lambda i: players[i]["riot_id"]

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in together:
        parent[find(i)] = find(j)
    by_root = {}
    for i in range(n):
        by_root.setdefault(find(i), []).append(i)
    groups = list(by_root.values())

    errors = []
    for g in groups:
        if len(g) > 5:
            errors.append(f"'Mit'-Gruppe mit {len(g)} Spielern (max. 5): "
                          + ", ".join(map(name, g)))
    for i, j in apart:
        if find(i) == find(j):
            errors.append(f"{name(i)} 'nicht mit' {name(j)}, beide sind aber über "
                          "'Mit' verbunden: " + ", ".join(map(name, by_root[find(i)])))
    must = sum(len(g) for g in groups if len(g) > 1)
    if must > 5 * k:
        errors.append(f"{must} Spieler in 'Mit'-Gruppen, aber nur {5 * k} Plätze.")
    if errors:
        raise SystemExit("Präferenzen nicht erfüllbar:\n  - " + "\n  - ".join(errors))

    gid = {i: g for g, members in enumerate(groups) for i in members}
    conflict = [set() for _ in groups]
    for i, j in apart:
        conflict[gid[i]].add(gid[j])
        conflict[gid[j]].add(gid[i])
    # grosse Gruppen zuerst, Einzelspieler nach Staerke (Schwache eher auf die Bank)
    order = sorted(range(len(groups)), key=lambda g: (-len(groups[g]),
                   -max(players[i]["strength"] for i in groups[g])))
    size, in_team, benched = [0] * k, [set() for _ in range(k)], []
    budget = [200000]  # Abbruch, falls das Backtracking explodiert

    def dfs(pos):
        budget[0] -= 1
        if budget[0] < 0:
            return False
        if pos == len(order):
            return True
        g = order[pos]
        sz = len(groups[g])
        tried_empty = False
        for t in range(k):
            if size[t] + sz > 5 or conflict[g] & in_team[t]:
                continue
            if size[t] == 0:  # leere Teams sind gleichwertig
                if tried_empty:
                    continue
                tried_empty = True
            size[t] += sz
            in_team[t].add(g)
            if dfs(pos + 1):
                return True
            size[t] -= sz
            in_team[t].discard(g)
        if sz == 1 and len(benched) < n_res:
            benched.append(groups[g][0])
            if dfs(pos + 1):
                return True
            benched.pop()
        return False

    if not dfs(0):
        raise SystemExit("Präferenzen nicht erfüllbar: keine Aufteilung gefunden, bei "
                         "der alle 'Mit'-Gruppen zusammen und alle 'Nicht mit'-Paare "
                         "getrennt sind.")

    teams = []
    for t in range(k):
        ps = [i for g in in_team[t] for i in groups[g]]
        cost = np.array([[role_cost(players[i], r) for r in ROLES] for i in ps])
        rows, cols = linear_sum_assignment(cost)
        teams.append({ROLES[c]: ps[r] for r, c in zip(rows, cols)})
    return teams, benched, [g for g in groups if len(g) > 1]


def assign_roles_pref(players, k, together, apart, iters=60000, seed=42, restarts=4):
    """Rollen + Ersatz so waehlen, dass ALLE Spieler-Praeferenzen erfuellt sind.

    Start ist pack_preferences (erfuellt alle Praeferenzen). Darauf Simulated
    Annealing ueber komplette Aufstellungen mit lexikographischen Kosten:
        1. verletzte Praeferenzen (Strafe groesser als alle Rollen-Kosten, und
           da der Start keine Verletzung hat, hat auch die beste Loesung keine)
        2. Spieler auf nicht angegebener Rolle   3. Spieler auf Zweitrolle
        4. Gleichstand: schwaechere Spieler sitzen aus
    Zuege: zwei Spieler derselben Rolle tauschen das Team, zwei Spieler tauschen
    ihre Rollen, ein Ersatz kommt fuer einen Spieler ins Team, oder eine ganze
    'Mit'-Gruppe wechselt (rollen-erhaltend) in ein anderes Team.
    Gibt (role_of, teams) zurueck; teams dient als Startloesung.
    """
    n = len(players)
    teams0, bench0, groups = pack_preferences(players, k, together, apart)

    n_res = len(bench0)
    by_strength = sorted(range(n), key=lambda i: players[i]["strength"])
    bench_cost = {i: (rank + 1) / (n * (n_res + 1)) for rank, i in enumerate(by_strength)}
    W_PREF = W_FOREIGN * (n + 1)  # eine Praeferenz wiegt mehr als alle Rollen-Kosten

    def cost(ts, bn):
        rc = sum(role_cost(players[i], r) for tm in ts for r, i in tm.items())
        return (W_PREF * len(pref_violations(ts, together, apart)) + rc
                + sum(bench_cost[i] for i in bn))

    slots = [(t, r) for t in range(k) for r in ROLES]
    rng = random.Random(seed)
    best = ([dict(tm) for tm in teams0], bench0[:], cost(teams0, bench0))
    for _ in range(restarts):
        teams, bench = [dict(tm) for tm in teams0], bench0[:]
        best = _anneal_lineup(k, teams, bench, groups, cost, slots, rng, iters,
                              T0=W_FOREIGN / 2, best=best)
        if best[2] < 1:  # keine Zweitrolle, keine fremde Rolle - besser geht nicht
            break

    teams = best[0]
    assert not pref_violations(teams, together, apart)
    role_of = [None] * n
    for tm in teams:
        for r, i in tm.items():
            role_of[i] = r
    return role_of, teams


def _anneal_lineup(k, teams, bench, groups, cost, slots, rng, iters, T0, best):
    """Ein SA-Lauf fuer assign_roles_pref; aktualisiert und liefert best."""
    cur_c = cost(teams, bench)
    T, decay = T0, (0.01 / T0) ** (1 / iters)
    for _ in range(iters):
        move = rng.random()
        if move < 0.4 and k > 1:                       # gleiche Rolle, Team tauschen
            r = rng.choice(ROLES)
            t1, t2 = rng.sample(range(k), 2)
            undo = [((t1, r), teams[t1][r]), ((t2, r), teams[t2][r])]
            teams[t1][r], teams[t2][r] = teams[t2][r], teams[t1][r]
        elif move < 0.55 and groups and k > 1:         # ganze Gruppe wechselt Team
            g = set(rng.choice(groups))
            t1 = next((t for t in range(k) if g & set(teams[t].values())), None)
            if t1 is None:
                continue
            t2 = rng.choice([t for t in range(k) if t != t1])
            roles = [r for r, i in teams[t1].items() if i in g]
            undo = [((t, r), teams[t][r]) for r in roles for t in (t1, t2)]
            for r in roles:
                teams[t1][r], teams[t2][r] = teams[t2][r], teams[t1][r]
        elif move < 0.85 or not bench:                 # zwei Spieler tauschen Rollen
            (t1, r1), (t2, r2) = rng.sample(slots, 2)
            if r1 == r2:
                continue
            undo = [((t1, r1), teams[t1][r1]), ((t2, r2), teams[t2][r2])]
            teams[t1][r1], teams[t2][r2] = teams[t2][r2], teams[t1][r1]
        else:                                          # Ersatz kommt ins Team
            t, r = rng.choice(slots)
            b = rng.randrange(len(bench))
            undo = [((t, r), teams[t][r]), ("bench", b, bench[b])]
            teams[t][r], bench[b] = bench[b], teams[t][r]
        c = cost(teams, bench)
        if c <= cur_c or rng.random() < math.exp((cur_c - c) / max(T, 1e-9)):
            cur_c = c
            if c < best[2]:
                best = ([dict(tm) for tm in teams], bench[:], c)
        else:
            for u in reversed(undo):
                if u[0] == "bench":
                    bench[u[1]] = u[2]
                else:
                    (t, r), p = u
                    teams[t][r] = p
        T *= decay
    return best


def _role_groups(players, role_of):
    return {r: [i for i in range(len(players)) if role_of[i] == r] for r in ROLES}


def team_mean(team, players):
    return sum(players[i]["strength"] for i in team.values()) / len(team)


def build_strongest(players, role_of, k, together=(), apart=(), start=None):
    """Buendelt Staerke: staerkster Spieler je Rolle -> Team 1, usw.
    Mit Spieler-Praeferenzen folgt eine lokale Suche ueber rollen-erhaltende
    Tausche mit lexikographischem Ziel: erst verletzte Praeferenzen, dann die
    Teamstaerken (Team 1 zuerst). Sie startet von der Sortierung und - falls
    gegeben - zusaetzlich von start (praeferenz-optimale Aufstellung); das
    bessere Ergebnis gewinnt.
    """
    groups = _role_groups(players, role_of)
    for r in ROLES:
        groups[r].sort(key=lambda i: players[i]["strength"], reverse=True)
    teams = [{r: groups[r][t] for r in ROLES} for t in range(k)]
    if not (together or apart):
        return teams

    def key(ts):
        sums = sorted((sum(players[i]["strength"] for i in tm.values()) for tm in ts),
                      reverse=True)
        return (len(pref_violations(ts, together, apart)), [-s for s in sums])

    def descend(ts):
        cur = key(ts)
        while True:
            best, best_key = None, cur
            for r in ROLES:
                for t1 in range(k):
                    for t2 in range(t1 + 1, k):
                        ts[t1][r], ts[t2][r] = ts[t2][r], ts[t1][r]
                        kk = key(ts)
                        if kk < best_key:
                            best, best_key = (r, t1, t2), kk
                        ts[t1][r], ts[t2][r] = ts[t2][r], ts[t1][r]
            if best is None:
                return ts, cur
            r, t1, t2 = best
            ts[t1][r], ts[t2][r] = ts[t2][r], ts[t1][r]
            cur = best_key

    cands = [descend(teams)]
    if start:
        cands.append(descend([dict(tm) for tm in start]))
    teams = min(cands, key=lambda c: c[1])[0]
    return sorted(teams, key=lambda tm: -sum(players[i]["strength"] for i in tm.values()))


def build_balanced(players, role_of, k, iters=30000, seed=42, together=(), apart=(),
                   start=None):
    """Gleicht die Teamstaerken an. Simulated Annealing ueber rollen-erhaltende
    Tausche (zwei Spieler derselben Rolle tauschen die Teams) - dadurch bleibt
    jedes Team automatisch vollstaendig und rollen-gueltig. start: optionale
    Startaufstellung mit denselben Rollen (z. B. aus assign_roles_pref); dann
    sind Spieler-Praeferenzen hart (verletzende Tausche werden nie angenommen),
    ohne start kostet jede Verletzung PREF_WEIGHT.
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
    if start:
        assign = {r: [tm[r] for tm in start] for r in ROLES}

    if k < 2:
        return [{r: assign[r][0] for r in ROLES}]

    def team_strengths(a):
        return [sum(players[a[r][t]]["strength"] for r in ROLES) for t in range(k)]

    def spread(a):
        ts = team_strengths(a)
        m = sum(ts) / k
        c = sum((x - m) ** 2 for x in ts)
        if together or apart:
            teams = [{r: a[r][t] for r in ROLES} for t in range(k)]
            bad = len(pref_violations(teams, together, apart))
            if bad:
                c = math.inf if start else c + PREF_WEIGHT * bad
        return c

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
# Exakte Loesung mit Gurobi (Rolle + Team gemeinsam)
# ---------------------------------------------------------------------------
def _gurobi_model(players, pool, k, together, apart, all_play, time_limit):
    """Grundmodell: x[p, r, t] = 1, wenn Spieler p in Team t Rolle r spielt.
    Jede Rolle ist moeglich (nicht angegebene werden nur gezaehlt), Spieler-
    Praeferenzen sind harte Constraints. all_play=False: Ersatz erlaubt (Phase 1).
    """
    m = gp.Model("teams")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    x = {(p, r, t): m.addVar(vtype=GRB.BINARY)
         for p in pool for r in ROLES for t in range(k)}
    z = {(p, t): gp.quicksum(x[p, r, t] for r in ROLES) for p in pool for t in range(k)}
    plays = {p: gp.quicksum(z[p, t] for t in range(k)) for p in pool}
    for p in pool:
        m.addConstr(plays[p] == 1 if all_play else plays[p] <= 1)
    for r in ROLES:
        for t in range(k):
            m.addConstr(gp.quicksum(x[p, r, t] for p in pool) == 1)

    in_pool = set(pool)
    for i, j in together:  # beide spielen, und zwar im selben Team
        m.addConstr(plays[i] == 1)
        for t in range(k):
            m.addConstr(z[i, t] == z[j, t])
    for i, j in apart:
        if i in in_pool and j in in_pool:
            for t in range(k):
                m.addConstr(z[i, t] + z[j, t] <= 1)

    n_foreign = gp.quicksum(v for (p, r, _), v in x.items()
                            if role_cost(players[p], r) == W_FOREIGN)
    n_second = gp.quicksum(v for (p, r, _), v in x.items() if role_cost(players[p], r) == 1)
    return m, x, n_foreign, n_second


def _gurobi_teams(x, k):
    teams = [dict() for _ in range(k)]
    for (p, r, t), v in x.items():
        if v.X > 0.5:
            teams[t][r] = p
    return teams


def gurobi_select(players, k, together, apart, time_limit=60):
    """Phase 1: alle Praeferenzen erfuellt (hart), dann lexikographisch
    (1) Spieler auf nicht angegebener Rolle, (2) auf Zweitrolle minimieren,
    (3) bei Gleichstand sitzen die Schwaechsten aus.
    Liefert (aktive Spieler, min. fremde Rollen, min. Zweitrollen, Status)."""
    n = len(players)
    m, x, n_foreign, n_second = _gurobi_model(players, range(n), k, together, apart,
                                              all_play=False, time_limit=time_limit)
    rank = {i: r + 1 for r, i in enumerate(sorted(range(n),
                                                  key=lambda i: players[i]["strength"]))}
    plays_weighted = gp.quicksum(rank[p] * v for (p, _, _), v in x.items())
    m.ModelSense = GRB.MINIMIZE
    m.setObjectiveN(n_foreign, index=0, priority=3)
    m.setObjectiveN(n_second, index=1, priority=2)
    m.setObjectiveN(-plays_weighted, index=2, priority=1)
    m.optimize()
    status = GRB_STATUS.get(m.Status, str(m.Status))
    if m.SolCount == 0:
        return None, None, None, status
    teams = _gurobi_teams(x, k)
    active = sorted(i for tm in teams for i in tm.values())
    foreign, second = role_counts(teams, players)
    return active, foreign, second, status


def solve_gurobi(players, active, k, mode, max_foreign, max_second, time_limit=60,
                 start=None, together=(), apart=()):
    """Phase 2: bildet k Teams aus den aktiven Spielern. Anders als die Heuristik
    ist die Rolle nicht vorab fixiert: jeder Spieler darf jede Rolle bekommen,
    solange alle Praeferenzen erfuellt sind und hoechstens max_foreign Spieler
    auf einer nicht angegebenen bzw. max_second auf einer Zweitrolle landen -
    die Minima aus Phase 1, die Prioritaeten bleiben also exakt erhalten.

    mode "staerkste":    lexikographisch Team 1 maximal, dann Team 2, ...
    mode "ausgeglichen": minimiere sum_t (S_t - S_mittel)^2  (MIQP)
    start: optionale Startloesung (z. B. aus der Heuristik) - Gurobi wird dann
    hoechstens besser, nie schlechter als diese.
    """
    m, x, n_foreign, n_second = _gurobi_model(players, active, k, together, apart,
                                              all_play=True, time_limit=time_limit)
    m.addConstr(n_foreign <= max_foreign)
    m.addConstr(n_second <= max_second)

    S = [gp.quicksum(players[p]["strength"] * v for (p, _, tt), v in x.items() if tt == t)
         for t in range(k)]
    # Teams sind austauschbar -> Symmetrie brechen (Team 1 ist das staerkste)
    for t in range(k - 1):
        m.addConstr(S[t] >= S[t + 1])

    if mode == "staerkste":
        m.ModelSense = GRB.MAXIMIZE
        for t in range(k):
            m.setObjectiveN(S[t], index=t, priority=k - t)
    else:
        mean = sum(players[p]["strength"] for p in active) / k
        m.setObjective(gp.quicksum((S[t] - mean) * (S[t] - mean) for t in range(k)),
                       GRB.MINIMIZE)
    if start and {i for tm in start for i in tm.values()} == set(active):
        # Startteams nach Staerke sortieren, damit die Symmetrie-Constraints gelten
        order = sorted(start, key=lambda tm: -sum(players[i]["strength"] for i in tm.values()))
        for v in x.values():
            v.Start = 0
        for t, team in enumerate(order):
            for r, p in team.items():
                x[p, r, t].Start = 1
    m.optimize()
    status = GRB_STATUS.get(m.Status, str(m.Status))
    if m.SolCount == 0:
        return None, status, None

    gap = m.MIPGap if mode == "ausgeglichen" else None
    return _gurobi_teams(x, k), status, gap


def team_std(teams, players):
    """Streuung der Team-Summen (kleiner = ausgeglichener)."""
    sums = [sum(players[i]["strength"] for i in team.values()) for team in teams]
    mu = sum(sums) / len(sums)
    return (sum((x - mu) ** 2 for x in sums) / len(sums)) ** 0.5


# ---------------------------------------------------------------------------
# Ausgabe
# ---------------------------------------------------------------------------
ROLE_NOTE = {0: "", 1: "  (Zweitrolle)", W_FOREIGN: "  (nicht angegebene Rolle!)"}
ROLE_TYPE = {0: "Main", 1: "Zweitrolle", W_FOREIGN: "nicht angegeben"}


def print_teams(title, teams, players):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print("   (Stärke = gewichtete Kombination aus Rang, LP und Win Rate)")
    for t, team in enumerate(teams, 1):
        print(f"\nTeam {t}  (Ø Stärke {team_mean(team, players):+.3f})")
        for r in ROLES:
            p = players[team[r]]
            off = ROLE_NOTE[role_cost(p, r)]
            rank = f"{p['tier']} {p['division']}".strip()
            lp = p.get("lp")
            wr = p.get("win_rate")
            lp_s = f"{int(lp)} LP" if lp is not None and not pd.isna(lp) else "—"
            wr_s = f"{wr*100:.0f}% WR" if wr is not None and not pd.isna(wr) else "—"
            print(f"   {ROLE_LABEL[r]:<8} {p['riot_id']:<24} {rank:<14}"
                  f"{lp_s:>7} {wr_s:>7}   Stärke {p['strength']:+.3f}{off}")
    means = [team_mean(t, players) for t in teams]
    if len(means) > 1:
        print(f"\n  Spanne der Team-Ø-Stärke: {max(means) - min(means):.3f} "
              f"(kleiner = ausgeglichener)")


def print_preferences(teams, players, together, apart):
    total = len(together) + len(apart)
    if not total:
        return
    bad = pref_violations(teams, together, apart)
    print(f"  Spieler-Präferenzen erfüllt: {total - len(bad)}/{total}")
    for i, j, kind in bad:
        print(f"    verletzt: {players[i]['riot_id']} {kind} {players[j]['riot_id']}")


def teams_to_dataframe(teams, players, mode, reserves=()):
    rec = []
    for t, team in enumerate(teams, 1):
        for r in ROLES:
            p = players[team[r]]
            rec.append({"modus": mode, "team": t, "rolle": ROLE_LABEL[r],
                        "riot_id": p["riot_id"], "rang": f"{p['tier']} {p['division']}".strip(),
                        "lp": p.get("lp"), "win_rate": p.get("win_rate"),
                        "main_rolle": p["main"] == r,
                        "rollen_typ": ROLE_TYPE[role_cost(p, r)],
                        "staerke": round(p["strength"], 4)})
    for i in reserves:
        p = players[i]
        rec.append({"modus": mode, "team": "Ersatz", "rolle": ROLE_LABEL[p["main"]],
                    "riot_id": p["riot_id"], "rang": f"{p['tier']} {p['division']}".strip(),
                    "lp": p.get("lp"), "win_rate": p.get("win_rate"),
                    "main_rolle": True, "rollen_typ": "Main",
                    "staerke": round(p["strength"], 4)})
    return pd.DataFrame(rec)


# ---------------------------------------------------------------------------
# Ablauf
# ---------------------------------------------------------------------------
def build_all(players, k, together=(), apart=(), time_limit=60):
    """Bildet alle Aufstellungen. Prioritaeten: (1) alle Spieler-Praeferenzen
    erfuellt (hart), (2) moeglichst wenige nicht angegebene Rollen, (3) moeglichst
    wenige Zweitrollen, (4) Ziel des Modus. Liefert [(name, titel, teams, ersatz)]."""
    n = len(players)
    role_of, _, infeasible = assign_roles(players, k)
    start = None
    if together or apart or infeasible:
        # Praeferenzen (bzw. Rollenmangel) -> Rollen und Ersatz gemeinsam waehlen
        role_of, start = assign_roles_pref(players, k, together, apart)
    reserves = [i for i in range(n) if role_of[i] is None]
    foreign = sum(1 for i in range(n) if role_of[i] is not None
                  and role_cost(players[i], role_of[i]) == W_FOREIGN)
    second = sum(1 for i in range(n) if role_of[i] is not None
                 and role_cost(players[i], role_of[i]) == 1)
    print(f"\nRollenverteilung (Heuristik): {n - len(reserves) - foreign - second} auf "
          f"Main-Rolle, {second} auf Zweitrolle, {foreign} auf nicht angegebener Rolle.")

    strongest = build_strongest(players, role_of, k, together, apart, start=start)
    balanced = build_balanced(players, role_of, k, together=together, apart=apart,
                              start=start)
    results = [("staerkste_teams", "STÄRKSTE TEAMS (Stärke gebündelt)", strongest, reserves),
               ("ausgeglichene_teams", "AUSGEGLICHENE TEAMS (Stärke angeglichen)",
                balanced, reserves)]
    if not HAS_GUROBI:
        print("\n[Hinweis] gurobipy nicht verfügbar - nur heuristische Lösung.")
        return results

    print(f"\nLöse mit Gurobi (Zeitlimit je Schritt {time_limit}s) ...")
    active, g_foreign, g_second, status = gurobi_select(players, k, together, apart,
                                                        time_limit)
    if active is None:
        print(f"[Gurobi] keine Aufstellung gefunden, die alle Präferenzen erfüllt "
              f"({status}).")
        return results
    g_reserves = [i for i in range(n) if i not in set(active)]
    print(f"Gurobi Phase 1 [{status}]: {g_second} auf Zweitrolle, {g_foreign} auf "
          "nicht angegebener Rolle.")
    for mode, name, title, warm in (
            ("staerkste", "staerkste_teams_gurobi", "STÄRKSTE TEAMS - GUROBI", strongest),
            ("ausgeglichen", "ausgeglichene_teams_gurobi", "AUSGEGLICHENE TEAMS - GUROBI",
             balanced)):
        teams, status, gap = solve_gurobi(players, active, k, mode, g_foreign, g_second,
                                          time_limit, warm, together, apart)
        if teams is None:
            print(f"\n[Gurobi {mode}] keine Lösung gefunden ({status}).")
            continue
        gap_s = f", Gap {gap * 100:.2f} %" if gap is not None else ""
        results.append((name, f"{title} [{status}{gap_s}]", teams, g_reserves))
    return results


def run(csv_path, time_limit=60):
    csv_path = Path(csv_path)
    players, problems = read_players_csv(csv_path)
    for msg in problems:
        print(f"[uebersprungen] {msg}")
    together, apart, pref_problems = build_preferences(players)
    for msg in pref_problems:
        print(f"[Präferenz] {msg}")
    if together or apart:
        print(f"Spieler-Präferenzen: {len(together)}x 'Mit', {len(apart)}x 'Nicht mit'.")

    n = len(players)
    if n < 5:
        raise SystemExit(f"Es werden mindestens 5 Spieler benoetigt (gefunden: {n}).")
    k = n // 5
    extra = f" + {n - 5 * k} Ersatz" if n % 5 else ""
    print(f"\n{n} Spieler -> {k} Team(s) à 5{extra}. Hole Ränge über die Riot API "
          f"(Region {PLATFORM}) ...")

    model = load_weights()
    api = RiotAPI(load_api_key())
    players = enrich_strength(players, api, model)

    unranked = [p["riot_id"] for p in players if not p["ranked"]]
    if unranked:
        print(f"\n[Hinweis] Ohne Solo/Duo-Rang (Stärke = Modell-Median geschätzt): "
              f"{', '.join(unranked)}")

    results = build_all(players, k, together, apart, time_limit)

    for _, title, teams, reserves in results:
        print_teams(title, teams, players)
        if reserves:
            print("\n  Ersatz: " + ", ".join(players[i]["riot_id"] for i in reserves))
        print_preferences(teams, players, together, apart)

    print("\n" + "=" * 60)
    print("VERGLEICH (Streuung der Team-Summen, kleiner = ausgeglichener)")
    print("=" * 60)
    for name, _, teams, _ in results:
        foreign, second = role_counts(teams, players)
        n_bad = len(pref_violations(teams, together, apart))
        pref_s = f"   Präferenzen verletzt: {n_bad}" if together or apart else ""
        print(f"  {name:<28} std {team_std(teams, players):.3f}   Zweitrolle {second}"
              f"   nicht angegeben {foreign}{pref_s}")

    saved = []
    for name, _, teams, reserves in results:
        out = csv_path.with_name(f"{csv_path.stem}_{name}.csv")
        teams_to_dataframe(teams, players, name, reserves).to_csv(
            out, index=False, sep=";", encoding="utf-8-sig")
        saved.append(out.name)
    print(f"\nGespeichert: {', '.join(saved)}")


if __name__ == "__main__":
    # Windows-Konsole auf UTF-8, damit Umlaute korrekt erscheinen
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) not in (2, 3):
        print("Aufruf: python team_builder.py <pfad/zur/spieler.csv> [gurobi-zeitlimit-s]")
        raise SystemExit(1)
    run(sys.argv[1], time_limit=int(sys.argv[2]) if len(sys.argv) == 3 else 60)
