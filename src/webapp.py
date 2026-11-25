"""
Weboberflaeche fuer den Team-Builder.

Startet einen lokalen Webserver (nur Python-Standardbibliothek) und oeffnet die
Oberflaeche im Browser. Die gesamte Logik (CSV, Riot API, Rollen, Heuristik,
Gurobi) kommt unveraendert aus team_builder.py - die Seite ist nur eine
andere Ein- und Ausgabe.

Aufruf:  python webapp.py [port] [--no-browser]      (Standard-Port: 8765)

Endpunkte:
  GET  /                Oberflaeche (src/web/)
  GET  /api/status      API-Key, Gurobi, Region
  GET  /api/example     players_example.csv als Spielerliste
  POST /api/key         API-Key fuer diese Sitzung setzen (nur im Arbeitsspeicher)
  POST /api/parse-csv   CSV (Rohbytes) -> Spielerliste fuer die Tabelle
  POST /api/build       Spielerliste -> Teams; Antwort als NDJSON-Stream mit
                        Fortschritt (stage/progress/wait/log) und result/error
"""
import json
import math
import os
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

import team_builder as tb
from main import WEIGHTS_PATH

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
WEB_DIR = SRC / "web"
EXAMPLE_CSV = ROOT / "players_example.csv"
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_BODY = 5 * 1024 * 1024
RANK_TTL = 15 * 60  # Raenge 15 min cachen -> erneutes Berechnen ist sofort fertig
ROLE_KIND = {0: "main", 1: "second", tb.W_FOREIGN: "foreign"}


# ---------------------------------------------------------------------------
# API-Key
# ---------------------------------------------------------------------------
_session_key = None
_key_checks = {}


def current_key():
    """(key, quelle) - Sitzungs-Key zuerst, sonst wie tb.load_api_key."""
    if _session_key:
        return _session_key, "Sitzung"
    if os.getenv("RIOT_API_KEY"):
        return os.getenv("RIOT_API_KEY").strip(), "Umgebungsvariable RIOT_API_KEY"
    try:
        key = tb.load_api_key()
    except RuntimeError:
        return None, None
    env = ROOT / "Riot-Crawler" / ".env"
    if env.exists() and "RIOT_API_KEY" in env.read_text(encoding="utf-8"):
        return key, "Riot-Crawler/.env"
    return key, "riot.txt"


def check_key(key):
    """True = gueltig, False = abgelehnt, None = nicht pruefbar (Netz/Limit)."""
    hit = _key_checks.get(key)
    if hit and time.time() - hit[1] < 300:
        return hit[0]
    url = (f"https://{tb.REGION}.api.riotgames.com/riot/account/v1/accounts/"
           f"by-riot-id/{quote('keycheck')}/{quote('0000')}")
    try:
        r = requests.get(url, headers={"X-Riot-Token": key}, timeout=8)
    except requests.RequestException:
        return None
    if r.status_code in (401, 403):
        valid = False
    elif r.status_code in (200, 404):
        valid = True
    else:
        return None
    _key_checks[key] = (valid, time.time())
    return valid


def status_payload():
    key, source = current_key()
    return {
        "key": {"present": bool(key), "source": source,
                "valid": check_key(key) if key else False},
        "gurobi": tb.HAS_GUROBI,
        "region": tb.PLATFORM.upper().rstrip("0123456789"),  # euw1 -> EUW
        "weights": WEIGHTS_PATH.exists(),
    }


# ---------------------------------------------------------------------------
# Riot API mit Cache (PUUIDs dauerhaft, Raenge RANK_TTL Sekunden)
# ---------------------------------------------------------------------------
_cache_lock = threading.Lock()
_puuid_cache, _rank_cache = {}, {}


class CachedRiotAPI(tb.RiotAPI):
    def puuid(self, name, tag):
        key = tb._id_key(f"{name}#{tag}")
        with _cache_lock:
            if key in _puuid_cache:
                return _puuid_cache[key]
        value = super().puuid(name, tag)
        if value:
            with _cache_lock:
                _puuid_cache[key] = value
        return value

    def solo_rank(self, puuid):
        with _cache_lock:
            hit = _rank_cache.get(puuid)
            if hit and time.time() - hit[1] < RANK_TTL:
                return hit[0]
        value = super().solo_rank(puuid)
        if not self.auth_failed:
            with _cache_lock:
                _rank_cache[puuid] = (value, time.time())
        return value


# ---------------------------------------------------------------------------
# Spielerliste <-> Tabelle
# ---------------------------------------------------------------------------
def table_rows(df):
    """CSV-DataFrame -> Zeilen fuer die Tabelle. Anders als players_from_table
    werden fehlerhafte Zeilen NICHT verworfen, sondern mit Hinweis uebernommen,
    damit man sie in der Oberflaeche korrigieren kann."""
    cols = tb.player_columns(df)
    rows, notes = [], []
    for _, row in df.iterrows():
        riot_id = row[cols["id"]].strip()
        if not riot_id:
            continue
        main = tb.normalize_role(row[cols["main"]])
        if not main and row[cols["main"]].strip():
            notes.append(f"{riot_id}: Main-Rolle '{row[cols['main']].strip()}' nicht erkannt.")
        seconds = []
        if cols["sec"]:
            for part in row[cols["sec"]].split(","):
                r = tb.normalize_role(part)
                if part.strip() and not r:
                    notes.append(f"{riot_id}: Rolle '{part.strip()}' nicht erkannt.")
                if r and r != main and r not in seconds:
                    seconds.append(r)
        prefs = {key: [s.strip() for s in row[col].split(",") if s.strip()] if col else []
                 for key, col in (("with", cols["with"]), ("not_with", cols["not_with"]))}
        rows.append({"riot_id": riot_id, "main": main or "", "seconds": seconds, **prefs})
    return rows, notes


def rows_to_table(rows):
    """Zeilen aus der Oberflaeche -> DataFrame im CSV-Format von team_builder."""
    recs = []
    for r in rows:
        recs.append({
            "RIOT ID": str(r.get("riot_id") or "").strip(),
            "Main Position": str(r.get("main") or ""),
            "Second Positions": ",".join(r.get("seconds") or []),
            "With": ",".join(r.get("with") or []),
            "Not With": ",".join(r.get("not_with") or []),
        })
    return pd.DataFrame(recs, columns=["RIOT ID", "Main Position", "Second Positions",
                                       "With", "Not With"])


# ---------------------------------------------------------------------------
# Ergebnis -> JSON
# ---------------------------------------------------------------------------
def _num(x):
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _clean(obj):
    """NaN/Inf -> None, numpy -> Python (JSON.parse kennt kein NaN)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _num(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _member(p, role):
    return {
        "riot_id": p["riot_id"],
        "role": role or p["main"],
        "kind": ROLE_KIND[tb.role_cost(p, role)] if role else "bench",
        "tier": p["tier"], "division": p["division"],
        "lp": _num(p.get("lp")), "win_rate": _num(p.get("win_rate")),
        "strength": p["strength"], "ranked": p["ranked"],
    }


def result_payload(res, players, together, apart):
    teams = res["teams"]
    team_objs = []
    for team in teams:
        total = sum(players[i]["strength"] for i in team.values())
        team_objs.append({"members": [_member(players[team[r]], r) for r in tb.ROLES],
                          "sum": total, "mean": total / len(team)})
    means = [t["mean"] for t in team_objs]
    foreign, second = tb.role_counts(teams, players)
    bad = tb.pref_violations(teams, together, apart)
    csv = tb.teams_to_dataframe(teams, players, res["name"], res["reserves"]).to_csv(
        index=False, sep=";")
    return {
        "id": res["name"],
        "mode": "strongest" if res["mode"] == "staerkste" else "balanced",
        "solver": res["solver"], "status": res["status"], "gap": res["gap"],
        "gurobi_note": res.get("gurobi_note"), "objective": res.get("objective"),
        "teams": team_objs,
        "reserves": [_member(players[i], None) for i in res["reserves"]],
        "metrics": {
            "spread": max(means) - min(means),
            "active": 5 * len(teams), "second": second, "foreign": foreign,
            "pref_total": len(together) + len(apart), "pref_ok": len(together) + len(apart) - len(bad),
            "violations": [f"{players[i]['riot_id']} {kind} {players[j]['riot_id']}"
                           for i, j, kind in bad],
        },
        "csv": csv,
    }


# ---------------------------------------------------------------------------
# Teams bilden (mit Fortschritt ueber emit)
# ---------------------------------------------------------------------------
class UserError(Exception):
    """Fehler mit verstaendlicher Meldung fuer die Oberflaeche."""


def run_build(rows, time_limit, emit):
    players, problems = tb.players_from_table(rows_to_table(rows))
    together, apart, pref_problems = tb.build_preferences(players)
    n = len(players)
    if n < 5:
        raise UserError(f"Es werden mindestens 5 gültige Spieler benötigt (gefunden: {n}).")
    k = n // 5

    key, _ = current_key()
    if not key:
        raise UserError("Kein Riot API Key hinterlegt.")
    if not WEIGHTS_PATH.exists():
        raise UserError("weights.json fehlt - bitte zuerst regression.py ausführen.")
    model = tb.load_weights()

    api = CachedRiotAPI(key, on_wait=lambda s: emit(type="wait", seconds=s))
    emit(type="stage", stage="ranks", total=n)
    try:
        tb.enrich_strength(players, api, model, on_progress=lambda done, total, p, ranked: emit(
            type="progress", stage="ranks", done=done, total=total,
            player=p["riot_id"], ranked=ranked))
    except requests.RequestException as e:
        raise UserError(f"Riot API nicht erreichbar ({type(e).__name__}). "
                        "Internetverbindung prüfen.") from e
    if api.auth_failed:
        _key_checks[key] = (False, time.time())
        raise UserError("Die Riot API lehnt den API Key ab - er ist ungültig oder "
                        "abgelaufen. Development Keys gelten nur 24 Stunden.")

    emit(type="stage", stage="teams")
    results = tb.build_all(players, k, together, apart, time_limit,
                           log=lambda m: emit(type="log", message=m.strip()),
                           on_stage=lambda s: emit(type="stage", stage=s,
                                                   time_limit=time_limit))
    return {
        "summary": {
            "n": n, "k": k, "reserves": n - 5 * k,
            "unranked": [p["riot_id"] for p in players if not p["ranked"]],
            "problems": problems + pref_problems,
            "with": len(together), "not_with": len(apart),
            "gurobi": tb.HAS_GUROBI,
        },
        "results": [result_payload(r, players, together, apart) for r in results],
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "TeamBuilder/1.0"

    def log_request(self, code="-", size="-"):  # nur fehlerhafte Anfragen loggen
        if str(getattr(code, "value", code))[:1] in "45":
            super().log_request(code, size)

    # -- Helfer -------------------------------------------------------------
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(_clean(obj), ensure_ascii=False))

    def _error(self, message, code=400):
        self._json({"error": message}, code)

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise UserError("Datei zu groß (max. 5 MB).")
        return self.rfile.read(length)

    def _local_only(self):
        """Nur Anfragen von dieser Seite selbst (Schutz vor fremden Webseiten,
        die an localhost posten, und vor DNS-Rebinding)."""
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            return False
        origin = self.headers.get("Origin")
        if origin and origin.split("://", 1)[-1].split(":")[0] not in ("127.0.0.1", "localhost"):
            return False
        return True

    # -- GET ----------------------------------------------------------------
    def do_GET(self):
        if not self._local_only():
            return self._error("Nur lokal erreichbar.", 403)
        path = self.path.split("?", 1)[0]
        if path in STATIC:
            name, ctype = STATIC[path]
            return self._send(200, (WEB_DIR / name).read_bytes(), ctype)
        if path == "/api/status":
            return self._json(status_payload())
        if path == "/api/example":
            if not EXAMPLE_CSV.exists():
                return self._error("players_example.csv nicht gefunden.", 404)
            rows, notes = table_rows(tb.read_table(EXAMPLE_CSV))
            return self._json({"players": rows, "notes": notes})
        self._error("Nicht gefunden.", 404)

    # -- POST ---------------------------------------------------------------
    def do_POST(self):
        global _session_key
        if not self._local_only():
            return self._error("Nur lokal erreichbar.", 403)
        path = self.path.split("?", 1)[0]
        try:
            raw = self._body()
            if path == "/api/parse-csv":
                rows, notes = table_rows(tb.read_table(raw))
                if not rows:
                    raise UserError("Die CSV enthält keine Spieler.")
                return self._json({"players": rows, "notes": notes})
            if path == "/api/key":
                key = str(json.loads(raw or b"{}").get("key") or "").strip()
                if not key:
                    raise UserError("Bitte einen API Key eingeben.")
                valid = check_key(key)
                if valid is False:
                    raise UserError("Die Riot API lehnt diesen Key ab - ungültig oder abgelaufen.")
                _session_key = key
                return self._json(status_payload())
            if path == "/api/build":
                body = json.loads(raw or b"{}")
                return self._stream_build(body.get("players") or [],
                                          max(5, min(600, int(body.get("time_limit") or 60))))
        except UserError as e:
            return self._error(str(e))
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            return self._error(f"CSV konnte nicht gelesen werden: {e}")
        self._error("Nicht gefunden.", 404)

    def _stream_build(self, rows, time_limit):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        lock = threading.Lock()

        def emit(**obj):
            line = json.dumps(_clean(obj), ensure_ascii=False) + "\n"
            with lock:
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()

        try:
            emit(type="result", **run_build(rows, time_limit, emit))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return  # Browser hat die Seite verlassen
        except (UserError, SystemExit, ValueError) as e:
            # SystemExit: team_builder meldet so unerfuellbare Praeferenzen
            emit(type="error", message=str(e))
        except Exception as e:
            traceback.print_exc()
            emit(type="error", message=f"Unerwarteter Fehler: {type(e).__name__}: {e}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    port = int(args[0]) if args else 8765
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Team Builder läuft auf {url}  (Beenden mit Strg+C)")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBeendet.")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
