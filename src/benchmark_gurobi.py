"""
Benchmark: exakter MIQP-Solver (Gurobi) vs. Simulated Annealing (Meta-Heuristik).

Beide loesen DASSELBE Problem der ausgeglichenen Teambildung: bei fester
Rollenzuordnung (aus assign_roles, Main bevorzugt) die k Teams so bilden, dass
die Varianz der Teamstaerken minimal wird. Das ist exakt das Ziel, das die SA
in team_builder.build_balanced optimiert - hier zusaetzlich beweisbar optimal
durch Gurobi.

Zielfunktion (identisch fuer beide):
    minimiere  sum_t (S_t - S_mittel)^2        (Varianz der Team-Summen)
mit S_t = Summe der Spielerstaerken in Team t. Da die Gesamtstaerke konstant ist,
ist das aequivalent zu  minimiere sum_t S_t^2  -> konvexes MIQP.

Constraints (identisch fuer beide):
    - jeder Spieler in genau einem Team
    - jedes Team hat je Rolle genau einen Spieler (auf seiner zugewiesenen Rolle)

Der Benchmark skaliert die Spielerzahl hoch und misst je Groesse:
    SA-Laufzeit & -Guete  vs.  Gurobi-Laufzeit, -Guete, Optimalitaetsluecke.
So wird sichtbar, ab welcher Spielerzahl der exakte Solver "zu lange" braucht
und die Meta-Heuristik die sinnvollere Wahl ist.

Voraussetzung: gurobipy + gueltige (akademische) Lizenz. Ohne Gurobi laeuft nur
der SA-Teil.
"""
import sys
import time
from pathlib import Path

import numpy as np

import team_builder as tb
from team_builder import ROLES

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except Exception:
    HAS_GUROBI = False

GRB_STATUS = {2: "optimal", 9: "Zeitlimit", 11: "unterbrochen",
              3: "unzulaessig", 4: "unbeschraenkt", 13: "suboptimal"}


# ---------------------------------------------------------------------------
# Zielfunktion (fuer beide Verfahren identisch berechnet)
# ---------------------------------------------------------------------------
def spread(teams, players):
    """Varianz-Zaehler: Summe der quadrierten Abweichungen der Team-Summen."""
    sums = [sum(players[i]["strength"] for i in team.values()) for team in teams]
    m = sum(sums) / len(sums)
    return sum((s - m) ** 2 for s in sums)


def team_std(teams, players):
    """Streuung der Team-Summen (interpretierbar, kleiner = ausgeglichener)."""
    return (spread(teams, players) / len(teams)) ** 0.5


# ---------------------------------------------------------------------------
# Synthetische Spielerpools mit realistischen Staerken
# ---------------------------------------------------------------------------
def load_strength_pool():
    """Reale Spielerstaerken aus dem Datensatz (fuer realistische Verteilung)."""
    from data_prep import load_clean_data
    from main import load_weights, compute_individual_strength
    data = load_clean_data().drop_duplicates("puuid")
    data = compute_individual_strength(data, load_weights())
    return data["individual_strength"].to_numpy()


def make_pool(n, strength_pool, seed):
    """n Spieler (Vielfaches von 5): reale Staerken gezogen, Main-Rollen im
    Round-Robin (garantiert eine gueltige Rollenverteilung) plus 0-2 zufaellige
    Zweitrollen."""
    rng = np.random.default_rng(seed)
    strengths = rng.choice(strength_pool, size=n, replace=False)
    players = []
    for i in range(n):
        main = ROLES[i % 5]  # Round-Robin -> je Rolle genau n/5 Mains
        others = [r for r in ROLES if r != main]
        n_sec = int(rng.integers(0, 3))
        seconds = list(rng.choice(others, size=n_sec, replace=False)) if n_sec else []
        players.append({"riot_id": f"P{i}", "main": main, "seconds": seconds,
                        "allowed": set([main] + seconds), "strength": float(strengths[i]),
                        "tier": "", "division": "", "ranked": True})
    return players, n // 5


# ---------------------------------------------------------------------------
# Exakter Solver (Gurobi, MIQP)
# ---------------------------------------------------------------------------
def solve_balanced_gurobi(players, role_of, k, time_limit=None):
    n = len(players)
    groups = {r: [p for p in range(n) if role_of[p] == r] for r in ROLES}

    m = gp.Model("balanced_teams")
    m.Params.OutputFlag = 0
    if time_limit:
        m.Params.TimeLimit = time_limit

    y = m.addVars(n, k, vtype=GRB.BINARY, name="y")
    # jeder Spieler in genau einem Team
    for p in range(n):
        m.addConstr(gp.quicksum(y[p, t] for t in range(k)) == 1)
    # je Rolle und Team genau ein Spieler
    for r in ROLES:
        for t in range(k):
            m.addConstr(gp.quicksum(y[p, t] for p in groups[r]) == 1)

    # Zentrierte Zielfunktion: Summe (S_t - Mittelwert)^2 = Varianz-Zaehler.
    # Der Mittelwert ist konstant (alle Spieler zugeordnet), also ist der
    # Minimierer identisch zu Summe S_t^2 - aber die Zielgroesse IST hier die
    # Varianz. So bezieht sich Gurobis relatives MIPGap auf die Varianz selbst
    # (nicht auf die riesige Konstante k*Mittelwert^2) und "optimal" ist echt.
    mean = sum(players[p]["strength"] for p in range(n)) / k
    S = [gp.quicksum(players[p]["strength"] * y[p, t] for p in range(n))
         for t in range(k)]
    m.setObjective(gp.quicksum((S[t] - mean) * (S[t] - mean) for t in range(k)),
                   GRB.MINIMIZE)
    m.optimize()

    teams = [dict() for _ in range(k)]
    if m.SolCount > 0:
        for p in range(n):
            for t in range(k):
                if y[p, t].X > 0.5:
                    teams[t][role_of[p]] = p
    return {
        "teams": teams,
        "runtime": m.Runtime,
        "gap": (m.MIPGap if m.SolCount > 0 else None),
        "status": GRB_STATUS.get(m.Status, str(m.Status)),
        "spread": (spread(teams, players) if m.SolCount > 0 else None),
    }


# ---------------------------------------------------------------------------
# Zeitbudgetierte Meta-Heuristik (Multi-Start SA) - nutzt das volle Zeitbudget
# ---------------------------------------------------------------------------
def build_balanced_timed(players, role_of, k, time_budget, seed=42,
                         iters_per_restart=8000):
    """Startet die SA wiederholt (Multi-Start) bis das Zeitbudget aufgebraucht ist
    und behaelt die beste Loesung. So bekommt die Heuristik dieselben Sekunden wie
    der Solver - Voraussetzung fuer einen fairen 'wer ist in X s besser'-Vergleich.
    """
    start = time.perf_counter()
    best, best_sp, run = None, float("inf"), 0
    while time.perf_counter() - start < time_budget:
        teams = tb.build_balanced(players, role_of, k,
                                  iters=iters_per_restart, seed=seed + run)
        sp = spread(teams, players)
        if sp < best_sp:
            best_sp, best = sp, teams
        run += 1
    return best, best_sp, run


# ---------------------------------------------------------------------------
# Duell: wer erreicht in T Sekunden die niedrigere Varianz?
# ---------------------------------------------------------------------------
def run_duel(sizes=(20, 50, 100, 150, 200, 250, 300), time_budget=30, n_seeds=5):
    """SA vs. Gurobi bei gleichem Zeitbudget, mehrere zufaellige Instanzen je
    Groesse. Aggregiert: Median-Varianz beider Verfahren, wie oft die SA gewinnt
    (x/n_seeds) und wie oft Gurobi beweisbar optimal war. So wird der Uebergangs-
    bereich robust charakterisiert statt von einer Einzelinstanz verrauscht.
    """
    if not HAS_GUROBI:
        print("gurobipy nicht verfuegbar - Duell nicht moeglich.")
        return

    print("Lade reale Spielerstaerken ...")
    pool = load_strength_pool()
    est = len(sizes) * n_seeds * 2 * time_budget / 60
    print(f"\nDUELL: SA (Multi-Start, volles Budget) vs. Gurobi - je {time_budget}s, "
          f"{n_seeds} Instanzen je Groesse")
    print(f"Laufzeit-Obergrenze grob ~{est:.0f} min.")
    print("=" * 84)
    print(f"{'n':>5}{'k':>4}{'SA-Var (Med)':>15}{'GRB-Var (Med)':>15}"
          f"{'SA-Siege':>11}{'Gurobi opt.':>13}")
    print("-" * 84)

    crossover_maj, crossover_all = None, None
    for n in sizes:
        sa_vars, grb_vars, sa_wins, n_opt, valid = [], [], 0, 0, 0
        for s in range(1, n_seeds + 1):
            players, k = make_pool(n, pool, seed=s)
            role_of, _, infeasible = tb.assign_roles(players, k)
            if infeasible:
                continue
            valid += 1
            _, sa_sp, _ = build_balanced_timed(players, role_of, k, time_budget, seed=s)
            g = solve_balanced_gurobi(players, role_of, k, time_limit=time_budget)
            sa_vars.append(sa_sp / k)
            grb_vars.append(g["spread"] / k if g["spread"] is not None else np.nan)
            if g["spread"] is None or sa_sp < g["spread"] - 1e-9:
                sa_wins += 1
            if g["status"] == "optimal":
                n_opt += 1
        if valid == 0:
            print(f"{n:>5}{n // 5:>4}  (keine gueltige Rollenverteilung)")
            continue

        med_sa = float(np.median(sa_vars))
        med_grb = float(np.nanmedian(grb_vars)) if np.any(~np.isnan(grb_vars)) else float("nan")
        print(f"{n:>5}{n // 5:>4}{med_sa:>15.4f}{med_grb:>15.4f}"
              f"{f'{sa_wins}/{valid}':>11}{f'{n_opt}/{valid}':>13}")

        if sa_wins > valid / 2 and crossover_maj is None:
            crossover_maj = n
        if sa_wins == valid and crossover_all is None:
            crossover_all = n

    print("-" * 84)
    if crossover_maj:
        print(f"=> Ab {crossover_maj} Spielern gewinnt die SA in {time_budget}s die "
              f"MEHRHEIT der Instanzen.")
    if crossover_all:
        print(f"=> Ab {crossover_all} Spielern gewinnt die SA ALLE Instanzen.")
    if not crossover_maj:
        print(f"=> Im getesteten Bereich gewinnt Gurobi meist - groessere n testen.")
    print("SA-Siege x/N: in so vielen der N Instanzen war die SA-Varianz niedriger.")
    print("Gurobi opt.: in so vielen Instanzen wurde die Optimalitaet in der Zeit bewiesen.")


# ---------------------------------------------------------------------------
# Einzel-Vergleich auf einer realen Spieler-CSV
# ---------------------------------------------------------------------------
def run_on_csv(csv_path, time_limit=60):
    """SA vs. Gurobi auf genau den Spielern einer CSV (reale Ränge über die API)."""
    players, problems = tb.read_players_csv(csv_path)
    for msg in problems:
        print(f"[übersprungen] {msg}")
    n = len(players)
    if n < 5 or n % 5 != 0:
        raise SystemExit(f"Es werden Vielfache von 5 Spielern benötigt (gefunden: {n}).")
    k = n // 5

    print(f"{n} Spieler -> {k} Team(s). Hole Ränge über die Riot API ...")
    model = tb.load_weights()
    api = tb.RiotAPI(tb.load_api_key())
    players = tb.enrich_strength(players, api, model)

    role_of, off_main, infeasible = tb.assign_roles(players, k)
    if infeasible:
        namen = ", ".join(p["riot_id"] for p in infeasible)
        raise SystemExit(f"Keine gültige Rollenverteilung möglich. Betroffen: {namen}.")
    print(f"\nRollenverteilung: {n - off_main} auf Main-Rolle, {off_main} auf Zweitrolle.")

    # --- Simulated Annealing ---
    t0 = time.perf_counter()
    sa_teams = tb.build_balanced(players, role_of, k)
    sa_time = time.perf_counter() - t0
    sa_sp, sa_std = spread(sa_teams, players), team_std(sa_teams, players)

    # --- Gurobi ---
    print("\n" + "=" * 60)
    print("VERGLEICH AUF DEN REALEN SPIELERN")
    print("=" * 60)
    print(f"  Simulated Annealing:  std {sa_std:>7.3f}   Zeit {sa_time:>7.3f}s")
    if HAS_GUROBI:
        g = solve_balanced_gurobi(players, role_of, k, time_limit=time_limit)
        grb_std = (g["spread"] / k) ** 0.5 if g["spread"] is not None else float("nan")
        gap = g["gap"] * 100 if g["gap"] is not None else float("nan")
        print(f"  Gurobi (MIQP):        std {grb_std:>7.3f}   Zeit {g['runtime']:>7.3f}s"
              f"   Gap {gap:.2f}%   [{g['status']}]")
        if g["spread"] and g["spread"] > 1e-9:
            ratio = sa_sp / g["spread"]
            if ratio <= 1.001:
                print("  -> SA hat dieselbe (optimale) Lösung gefunden.")
            else:
                print(f"  -> SA ist um {(ratio - 1) * 100:.1f} % schlechter (Spread-Verhältnis).")
        else:
            print("  -> Perfekt ausgeglichen möglich (Spread ~ 0).")
        tb.print_teams("SIMULATED ANNEALING", sa_teams, players)
        tb.print_teams("GUROBI (beweisbar optimal)", g["teams"], players)
    else:
        print("  Gurobi (MIQP):        gurobipy nicht verfügbar - kein Vergleich.")
        tb.print_teams("SIMULATED ANNEALING", sa_teams, players)


# ---------------------------------------------------------------------------
# Benchmark-Schleife (synthetische Skalierung)
# ---------------------------------------------------------------------------
def run_benchmark(sizes=(10, 15, 20, 25, 30, 40, 50, 60, 80, 100),
                  time_limit=30, seed=42):
    if not HAS_GUROBI:
        print("gurobipy nicht verfuegbar - installiere Gurobi (+ Lizenz):")
        print("  pip install gurobipy   und   akademische Lizenz aktivieren.")
        print("Es laeuft nur der SA-Teil (kein Vergleich moeglich).\n")

    print("Lade reale Spielerstaerken ...")
    pool = load_strength_pool()
    print(f"Staerke-Pool: {len(pool)} Spieler, "
          f"Range {pool.min():+.2f} .. {pool.max():+.2f}\n")

    print(f"Gurobi-Zeitlimit je Instanz: {time_limit}s  ('Zeitlimit' = zu lange)")
    print("=" * 92)
    print(f"{'n':>4}{'k':>4}{'SA s':>8}{'SA std':>9}"
          f"{'GRB s':>9}{'GRB std':>9}{'Gap %':>8}{'Status':>12}{'SA/GRB':>9}")
    print("-" * 92)

    for n in sizes:
        players, k = make_pool(n, pool, seed)
        role_of, off_main, infeasible = tb.assign_roles(players, k)
        if infeasible:
            print(f"{n:>4}{k:>4}  (keine gueltige Rollenverteilung - uebersprungen)")
            continue

        # --- Simulated Annealing ---
        t0 = time.perf_counter()
        sa_teams = tb.build_balanced(players, role_of, k)
        sa_time = time.perf_counter() - t0
        sa_spread = spread(sa_teams, players)
        sa_std = team_std(sa_teams, players)

        # --- Gurobi ---
        if HAS_GUROBI:
            g = solve_balanced_gurobi(players, role_of, k, time_limit=time_limit)
            grb_std = (g["spread"] / k) ** 0.5 if g["spread"] is not None else float("nan")
            gap_pct = g["gap"] * 100 if g["gap"] is not None else float("nan")
            ratio = (sa_spread / g["spread"]) if (g["spread"] and g["spread"] > 1e-9) else 1.0
            print(f"{n:>4}{k:>4}{sa_time:>8.2f}{sa_std:>9.3f}"
                  f"{g['runtime']:>9.2f}{grb_std:>9.3f}{gap_pct:>8.2f}"
                  f"{g['status']:>12}{ratio:>9.3f}")
        else:
            print(f"{n:>4}{k:>4}{sa_time:>8.2f}{sa_std:>9.3f}"
                  f"{'-':>9}{'-':>9}{'-':>8}{'kein Gurobi':>12}{'-':>9}")

    print("-" * 92)
    print("SA std / GRB std: Streuung der Team-Summen (kleiner = ausgeglichener).")
    print("SA/GRB (Spread-Verhaeltnis): 1,00 = SA hat das Optimum erreicht; groesser = SA schlechter.")
    print("Status 'Zeitlimit' + Gap>0: Gurobi hat die Optimalitaet nicht mehr bewiesen -> ab hier lohnt die Heuristik.")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = sys.argv[1:]
    # "duel [budget]"  -> SA vs Gurobi bei gleichem Zeitbudget (Crossover suchen)
    # <csv> [limit]    -> nur die Spieler dieser CSV
    # [limit]          -> synthetischer Skalierungs-Benchmark
    if args and args[0] == "duel":
        tb_budget = int(args[1]) if len(args) > 1 else 30
        seeds = int(args[2]) if len(args) > 2 else 5
        run_duel(time_budget=tb_budget, n_seeds=seeds)
    elif args and (args[0].lower().endswith(".csv") or Path(args[0]).exists()):
        tl = int(args[1]) if len(args) > 1 else 60
        run_on_csv(args[0], time_limit=tl)
    else:
        tl = int(args[0]) if args else 30
        run_benchmark(time_limit=tl)
