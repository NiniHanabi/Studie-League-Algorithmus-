"""
Grafiken zur empirischen Faktor-Auswahl.

Liest die Ergebnisse aus weights.json (von regression.py erzeugt) und erstellt
zwei druckfertige Abbildungen fuer die Arbeit:

  Abbildung 1: Inkrementeller Beitrag jedes Faktors (Delta-AUC) mit der
               Aufnahme-Schwelle - zeigt, ab welchem Cutoff Faktoren verworfen
               werden.
  Abbildung 2: Forward-Selection-Kurve - die kreuzvalidierte AUC steigt mit den
               ersten drei Faktoren und laeuft danach in ein Plateau.

Speichert je Abbildung als PNG (300 dpi) und PDF (Vektor) in ../figures/.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

from data_prep import TIER_ORDER

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.json"
STRAT_PATH = Path(__file__).resolve().parent / "stratification.json"
TIER_PATH = Path(__file__).resolve().parent / "tier_analysis.json"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# --- Farben (aus der validierten Referenz-Palette) ---
BLUE = "#2a78d6"     # aufgenommen (Signal)
GRAY = "#9e9d99"     # verworfen (Rauschen, tritt zurueck)
ORANGE = "#eb6834"   # Schwellenlinie
INK = "#0b0b0b"      # Primaertext
INK2 = "#52514e"     # Sekundaertext
GRID = "#e6e5e1"     # zuruecktretendes Raster
REJECT_ZONE = "#f4f3f0"  # sehr helle Verwerfungszone

# Sequenzielle Blau-Rampe fuer die GEORDNETEN Elo-Baender (hell = niedrig, dunkel = hoch)
BAND_COLOR = {"low": "#86b6ef", "mid": "#3987e5", "high": "#184f95"}

# Anzeigenamen der Faktoren (deutsch)
LABELS = {
    "rank": "Rang (Tier + Division)",
    "league_points": "League Points",
    "winrate": "Win Rate",
    "matches": "Anzahl Matches",
    "summoner_level": "Summoner Level",
    "champion_mastery": "Champion Mastery",
    "flex_rank": "Flex-Rang",
    # Teamfaktoren
    "role_decomposition": "Rollen-Zerlegung (5 Rollen-Elos)",
    "role_synergies": "Rollen-Synergien (JG×MID …)",
    "lp_variance": "LP-Varianz / Team-Streuung",
    "weakest_link": "Schwächstes Glied (Minimum)",
}

AUC_THRESHOLD = 0.002


def de(x, nachkomma=3):
    """Zahl mit deutschem Dezimalkomma formatieren."""
    return f"{x:.{nachkomma}f}".replace(".", ",")


def _style_axes(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)


def plot_cutoff(model):
    """Abbildung 1: Delta-AUC je Faktor mit Aufnahme-Schwelle.

    Zwei Gruppen: individuelle Faktoren (Zuwachs im Forward-Aufbau) und
    Teamfaktoren (Zuwachs ueber dem Team-Mittelwert-Modell). Beide auf derselben
    ΔAUC-Skala - keiner der Teamfaktoren erreicht die Schwelle.
    """
    ind = sorted(model["candidate_auc_table"],
                 key=lambda r: r["delta_auc"], reverse=True)
    team = sorted(model["team_factor_table"],
                  key=lambda r: r["delta_auc"], reverse=True)

    # Layout von oben nach unten: Ueberschrift + Balkenzeilen je Gruppe
    layout = [("header", "INDIVIDUELLE FAKTOREN")]
    layout += [("bar", r) for r in ind]
    layout += [("header", "TEAMFAKTOREN  (Zuwachs über Team-Mittelwert)")]
    layout += [("bar", r) for r in team]

    n = len(layout)
    y_of = {i: n - 1 - i for i in range(n)}  # Index 0 ganz oben

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    _style_axes(ax)

    # Verwerfungszone (alles links der Schwelle) sehr hell hinterlegen
    ax.axvspan(-1, AUC_THRESHOLD, color=REJECT_ZONE, zorder=0)

    tick_pos, tick_lab, tick_bold = [], [], []
    for i, (kind, payload) in enumerate(layout):
        yi = y_of[i]
        tick_pos.append(yi)
        if kind == "header":
            tick_lab.append(payload)
            tick_bold.append(True)
            continue
        r = payload
        d, a = r["delta_auc"], r["accepted"]
        tick_lab.append(LABELS.get(r["factor"], r["factor"]))
        tick_bold.append(False)
        ax.barh(yi, d, color=BLUE if a else GRAY, height=0.62, zorder=3)
        # Direkte Wertebeschriftung (Achse verzerrt -> exakte Zahl an den Balken).
        # Grosse Werte 3, winzige 4 Nachkommastellen (sonst "+0,000").
        # Negative Balken werden RECHTS von 0 beschriftet (freie Verwerfungszone),
        # damit die Zahl nicht mit den langen Faktornamen links kollidiert.
        nk = 3 if abs(d) >= 0.001 else 4
        x_lab = (d + 0.0004) if d >= 0 else 0.0004
        ax.text(x_lab, yi, f"{'+' if d >= 0 else '−'}{de(abs(d), nk)}",
                va="center", ha="left", fontsize=9,
                color=INK if a else INK2, fontweight="bold" if a else "normal")

    # Trennlinie zwischen den Gruppen (oberhalb der Team-Ueberschrift)
    team_header_i = next(i for i, (k, _) in enumerate(layout) if k == "header" and i > 0)
    ax.axhline(y_of[team_header_i] + 0.6, color=GRID, linewidth=1.0, zorder=1)

    # Schwellenlinie
    ax.axvline(AUC_THRESHOLD, color=ORANGE, linestyle="--", linewidth=1.6, zorder=4)
    ax.text(AUC_THRESHOLD, n - 0.3, f"  Aufnahme-Schwelle  {de(AUC_THRESHOLD)}",
            color=ORANGE, va="center", ha="left", fontsize=9, fontweight="bold")

    # Symlog: Bereich um 0 linear (macht die winzigen Werte sichtbar), darueber
    # logarithmisch (baendigt den grossen Rang-Wert). Knick genau bei 0,002.
    ax.set_xscale("symlog", linthresh=AUC_THRESHOLD, linscale=0.9)
    ax.set_xlim(-0.0015, 0.25)
    ax.set_ylim(-0.6, n - 0.2)
    ax.set_xticks([0, 0.002, 0.01, 0.05, 0.1])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, 3)))
    ax.xaxis.set_minor_locator(plt.NullLocator())  # symlog-Zwischenticks weg

    ax.set_yticks(tick_pos)
    labels = ax.set_yticklabels(tick_lab, fontsize=10)
    for lab, bold in zip(labels, tick_bold):
        lab.set_fontweight("bold" if bold else "normal")
        lab.set_color(INK2 if bold else INK)
        if bold:
            lab.set_fontsize(9)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)

    ax.set_xlabel("Zuwachs an kreuzvalidierter AUC gegenüber dem jeweiligen "
                  "Basismodell (ΔAUC, symlog)", color=INK2, fontsize=9.5)

    fig.tight_layout(rect=[0, 0, 1, 0.9])  # oben Platz fuer Titel reservieren
    fig.text(0.012, 0.965, "Inkrementeller Beitrag der Faktoren zur Vorhersagegüte",
             color=INK, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.012, 0.925,
             "5-fach kreuzvalidiert auf 75 % der Matches · blau = aufgenommen, "
             "grau = verworfen",
             color=INK2, fontsize=9.5, ha="left", va="top")
    return fig


def plot_forward(model):
    """Abbildung 2: Forward-Selection-Kurve (AUC steigt, dann Plateau)."""
    table = {r["factor"]: r for r in model["candidate_auc_table"]}
    order_accepted = ["rank", "league_points", "winrate"]
    rejected = ["matches", "summoner_level", "champion_mastery", "flex_rank"]

    # Staircase der aufgenommenen Faktoren, Start bei AUC 0,5 (Muenzwurf)
    xs = ["Basis"] + [LABELS[f].split(" (")[0] for f in order_accepted]
    ys = [0.5] + [table[f]["cv_auc"] for f in order_accepted]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    _style_axes(ax)

    x_idx = list(range(len(xs)))
    ax.plot(x_idx, ys, color=BLUE, linewidth=2.2, marker="o",
            markersize=7, zorder=3, label="aufgenommene Faktoren (kumuliert)")
    for xi, yv in zip(x_idx, ys):
        ax.annotate(de(yv, 4), (xi, yv), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9,
                    color=INK, fontweight="bold")

    # Plateau-Bereich nach Win Rate markieren
    plateau = table["winrate"]["cv_auc"]
    ax.axhspan(plateau - 0.004, plateau + 0.004, color=REJECT_ZONE, zorder=0)
    ax.axhline(plateau, color=GRAY, linestyle="--", linewidth=1.2, zorder=1)

    # Verworfene Kandidaten als Punkte im Plateau (jeweils Kern + dieser Faktor)
    rx = len(xs) - 1 + 0.6
    for i, f in enumerate(rejected):
        ax.scatter(rx + i * 0.55, table[f]["cv_auc"], color=GRAY, s=45,
                   zorder=3, marker="D")
    ax.scatter([], [], color=GRAY, s=45, marker="D",
               label="verworfener Kandidat (kein Zuwachs)")
    ax.text(rx + (len(rejected) - 1) * 0.275, plateau - 0.012,
            "Plateau:\nMatches, Level, Mastery, Flex", ha="center", va="top",
            color=INK2, fontsize=8.5)

    ax.set_xticks(x_idx)
    ax.set_xticklabels(["Basis\n(0,5)", "+ Rang", "+ League\nPoints",
                        "+ Win Rate"], color=INK, fontsize=9.5)
    ax.set_xlim(-0.3, rx + len(rejected) * 0.55 + 0.2)
    ax.set_ylim(0.48, 0.78)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, 2)))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.set_ylabel("Kreuzvalidierte AUC", color=INK2, fontsize=9.5)
    ax.set_title("Vorhersagegüte steigt nur mit den ersten drei Faktoren",
                 color=INK, fontsize=13, fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.055,
            "Danach flaches Plateau – weitere Faktoren heben die AUC nicht "
            "über das Rauschen",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    return fig


def plot_common(model):
    """Abbildung 3: Alle Faktoren gegen EIN gemeinsames Basismodell.

    Streng vergleichbare Darstellung - jeder Balken misst den Beitrag zur
    kreuzvalidierten AUC relativ zum finalen Modell (Rang + LP + Win Rate):
    Kernfaktoren per Leave-one-out (einzigartiger Beitrag), alle uebrigen als
    Zuwachs beim Hinzufuegen. Dadurch sind individuelle und Teamfaktoren direkt
    in ihrer Balkenlaenge vergleichbar. Lineare Achse (kein grosser Ausreisser).
    """
    cb = model["common_baseline"]
    core = sorted(cb["core_leave_one_out"].items(), key=lambda x: -x[1])
    extras = sorted(cb["extra_individual"].items(), key=lambda x: -x[1])
    team = sorted(cb["team"].items(), key=lambda x: -x[1])

    layout = [("header", "KERNFAKTOREN  (einzigartiger Beitrag · Leave-one-out)")]
    layout += [("bar", f, v) for f, v in core]
    layout += [("header", "WEITERE INDIVIDUELLE FAKTOREN  (Zuwachs über finales Modell)")]
    layout += [("bar", f, v) for f, v in extras]
    layout += [("header", "TEAMFAKTOREN  (Zuwachs über finales Modell)")]
    layout += [("bar", f, v) for f, v in team]

    n = len(layout)
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    _style_axes(ax)
    ax.axvspan(-1, AUC_THRESHOLD, color=REJECT_ZONE, zorder=0)

    tick_pos, tick_lab, tick_bold = [], [], []
    header_ys = []
    for i, row in enumerate(layout):
        yi = n - 1 - i
        tick_pos.append(yi)
        if row[0] == "header":
            tick_lab.append(row[1])
            tick_bold.append(True)
            if i > 0:
                header_ys.append(yi + 0.6)
            continue
        _, f, v = row
        a = v > AUC_THRESHOLD
        tick_lab.append(LABELS.get(f, f))
        tick_bold.append(False)
        ax.barh(yi, v, color=BLUE if a else GRAY, height=0.62, zorder=3)
        nk = 3 if abs(v) >= 0.001 else 4
        # Kernfaktoren (ueber Schwelle): Label am Balkenende. Sub-Schwellen-Faktoren:
        # alle rechts der Schwellenlinie buendig ausrichten (Werte-Spalte, Linie frei).
        x_lab = (v + 0.001) if a else (AUC_THRESHOLD + 0.001)
        ax.text(x_lab, yi, f"{'+' if v >= 0 else '−'}{de(abs(v), nk)}",
                va="center", ha="left", fontsize=9,
                color=INK if a else INK2, fontweight="bold" if a else "normal")

    for hy in header_ys:
        ax.axhline(hy, color=GRID, linewidth=1.0, zorder=1)

    ax.axvline(AUC_THRESHOLD, color=ORANGE, linestyle="--", linewidth=1.6, zorder=4)
    ax.text(AUC_THRESHOLD, n - 0.3, f"  Relevanz-Schwelle  {de(AUC_THRESHOLD)}",
            color=ORANGE, va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_xlim(-0.004, 0.062)
    ax.set_ylim(-0.6, n - 0.2)
    # 0,002-Tick weglassen (steht zu dicht an 0); die Schwelle markiert die Linie
    ax.set_xticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, 3)))
    ax.set_yticks(tick_pos)
    labels = ax.set_yticklabels(tick_lab, fontsize=10)
    for lab, bold in zip(labels, tick_bold):
        lab.set_fontweight("bold" if bold else "normal")
        lab.set_color(INK2 if bold else INK)
        if bold:
            lab.set_fontsize(9)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)

    ax.set_xlabel("Beitrag zur kreuzvalidierten AUC gegenüber dem finalen Modell (ΔAUC)",
                  color=INK2, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.text(0.012, 0.965,
             "Alle Faktoren gegen dasselbe Basismodell (Rang + LP + Win Rate)",
             color=INK, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.012, 0.925,
             f"Gemeinsame Referenz: finales Modell mit CV-AUC {de(cb['baseline_auc'], 3)}"
             " · nur die drei Kernfaktoren erreichen die Schwelle",
             color=INK2, fontsize=9.5, ha="left", va="top")
    return fig


def plot_stratification(strat):
    """Abbildung 4: Faktor-Gewichte ueber die drei Elo-Baender.

    Zwei Panels. Links: standardisierte Gewichte je Faktor, gruppiert nach Band
    (zeigt, dass Rang und League Points ueber die Elo-Baender die Rollen tauschen,
    Win Rate dagegen stabil bleibt). Rechts: Anteil "staerkeres Team gewinnt"
    je Band (Vorhersageguete ueber die Niveaus).
    """
    bands = strat["bands"]
    labels = strat["band_label"]
    order = ["low", "mid", "high"]
    factors = [("rank", "Rang"), ("league_points", "League Points"),
               ("winrate", "Win Rate")]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(11, 5.3), gridspec_kw={"width_ratios": [3, 1.15]})
    _style_axes(axA)
    _style_axes(axB)

    # --- Panel A: gruppierte Balken (Faktor x Band) ---
    xf = list(range(len(factors)))
    w = 0.26
    for j, b in enumerate(order):
        vals = [bands[b]["weights"][f] for f, _ in factors]
        xs = [x + (j - 1) * w for x in xf]
        axA.bar(xs, vals, width=w, color=BAND_COLOR[b], label=labels[b], zorder=3)
        for x, v in zip(xs, vals):
            axA.text(x, v + 0.06, de(v, 2), ha="center", va="bottom",
                     fontsize=8, color=INK)
    axA.axhline(0, color=GRID, linewidth=1.0)
    axA.set_xticks(xf)
    axA.set_xticklabels([name for _, name in factors], fontsize=10, color=INK)
    axA.set_ylim(0, 4.2)
    axA.yaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, 1)))
    axA.set_ylabel("Standardisiertes Gewicht (pro bandinterner Streuung)",
                   color=INK2, fontsize=9.5)
    axA.grid(axis="y", color=GRID, linewidth=0.8)
    axA.set_axisbelow(True)
    axA.legend(frameon=False, fontsize=9, loc="upper left", title="Elo-Band",
               title_fontsize=9)

    # --- Panel B: staerkeres Team gewinnt je Band ---
    xs = list(range(len(order)))
    vals = [bands[b]["stronger_team_wins"] for b in order]
    axB.bar(xs, vals, width=0.62, color=[BAND_COLOR[b] for b in order], zorder=3)
    for x, v in zip(xs, vals):
        axB.text(x, v + 0.006, f"{v*100:.1f} %".replace(".", ","),
                 ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")
    axB.set_xticks(xs)
    axB.set_xticklabels([labels[b] for b in order], fontsize=9, color=INK, rotation=12)
    axB.set_ylim(0.5, 0.76)
    axB.yaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, 2)))
    axB.set_title("Stärkeres Team gewinnt\n(Achsenstart 0,50 = Zufall)", color=INK,
                  fontsize=10.5, fontweight="bold", loc="left")
    axB.grid(axis="y", color=GRID, linewidth=0.8)
    axB.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.text(0.012, 0.965,
             "Faktor-Gewichte über die Elo-Bänder: Rang und League Points tauschen die Rollen",
             color=INK, fontsize=12.5, fontweight="bold", ha="left", va="top")
    fig.text(0.012, 0.925,
             "Getrennte Regression je Band · Gewicht = Effekt pro bandinterner "
             "Standardabweichung · Win Rate bleibt als Einziges stabil",
             color=INK2, fontsize=9.5, ha="left", va="top")
    return fig


def plot_tier(tier_data):
    """Abbildung 5: Heatmap Tier x Faktor (Delta-AUC über dem Kernmodell).

    Zeigt für jedes der 10 Tiers den inkrementellen Beitrag jedes Kandidaten.
    Kernbotschaft: Kein Feld übersteigt den Rauschpegel seines Tiers (im
    Spaltenkopf als ±Wert angegeben) -> die bunten Zellen sind Stichproben-
    Rauschen, kein tier-spezifisches Signal. Werte in AUC × 10⁻³ (Tausendstel).
    """
    tiers = tier_data["tiers"]
    threshold = tier_data["threshold"]
    factors = [("matches", "Matches"), ("summoner_level", "Summoner Level"),
               ("champion_mastery", "Champion Mastery"), ("flex", "Flex-Rang"),
               ("role_decomposition", "Rollen-Wichtigkeit"),
               ("role_synergies", "Rollen-Synergien"),
               ("lp_variance", "LP-Varianz"), ("weakest_link", "Schwächstes Glied")]
    tshort = {"IRON": "Iron", "BRONZE": "Bronze", "SILVER": "Silver", "GOLD": "Gold",
              "EMERALD": "Emerald", "PLATINUM": "Platinum", "DIAMOND": "Diamond",
              "MASTER": "Master", "GRANDMASTER": "GM", "CHALLENGER": "Challenger"}
    tier_keys = sorted([t for t in tiers if tiers[t]["core_auc"] is not None],
                       key=lambda t: TIER_ORDER[t])

    nrows, ncols = len(factors), len(tier_keys)
    M = np.full((nrows, ncols), np.nan)         # Delta-AUC in Tausendstel
    credible = np.zeros((nrows, ncols), dtype=bool)
    for j, t in enumerate(tier_keys):
        std = tiers[t]["core_std"]
        for i, (fkey, _) in enumerate(factors):
            v = tiers[t]["deltas"].get(fkey)
            if v is None:
                continue
            M[i, j] = v * 1000
            credible[i, j] = (v > threshold) and (std is not None and v > std)

    # Diverging: rot (schadet) -> neutral -> blau (hilft); auf ±20 begrenzt
    cmap = LinearSegmentedColormap.from_list(
        "dauc", ["#b5271b", "#f0efec", "#184f95"])
    cmap.set_bad("#e6e5e1")
    norm = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    mesh = ax.pcolormesh(np.arange(ncols + 1), np.arange(nrows + 1),
                         np.ma.masked_invalid(M), cmap=cmap, norm=norm,
                         edgecolors="white", linewidth=2)

    # Zellbeschriftung + Rahmen um glaubwürdige Zellen
    for i in range(nrows):
        for j in range(ncols):
            if np.isnan(M[i, j]):
                ax.text(j + 0.5, i + 0.5, "n/a", ha="center", va="center",
                        fontsize=8, color=INK2)
                continue
            iv = int(round(M[i, j]))
            txt = f"{iv:+d}" if iv != 0 else "0"
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=9,
                    color="white" if abs(iv) > 10 else INK)
            if credible[i, j]:
                ax.add_patch(Rectangle((j, i), 1, 1, fill=False,
                                       edgecolor="black", linewidth=2.2))

    ax.set_xticks([j + 0.5 for j in range(ncols)])
    ax.set_xticklabels(
        [f"{tshort[t]}\nn={tiers[t]['n']}\n±{tiers[t]['core_std']*1000:.0f}"
         for t in tier_keys], fontsize=8.5, color=INK)
    ax.set_yticks([i + 0.5 for i in range(nrows)])
    ax.set_yticklabels([name for _, name in factors], fontsize=10, color=INK)
    ax.invert_yaxis()                 # erster Faktor oben
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(0, ncols)
    ax.set_ylim(nrows, 0)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02,
                        ticks=[-20, -10, 0, 10, 20], extend="both")
    cbar.ax.set_yticklabels(["−20", "−10", "0", "+10", "+20"])
    cbar.set_label("ΔAUC × 10⁻³  (blau = hilft, rot = schadet)",
                   color=INK2, fontsize=9)
    cbar.outline.set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.text(0.012, 0.965,
             "Faktor-Beitrag je Tier (ΔAUC) – alles im Rauschen",
             color=INK, fontsize=13, fontweight="bold", ha="left", va="top")
    fig.text(0.012, 0.925,
             "Spaltenkopf: n und Rauschpegel ±(CV-Streuung) je Tier · kein Feld "
             "übersteigt seinen Rauschpegel (kein Rahmen) → kein tier-spezifisches Signal",
             color=INK2, fontsize=9.5, ha="left", va="top")
    return fig


def main():
    model = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    FIG_DIR.mkdir(exist_ok=True)

    figures = {
        "abb1_faktor_cutoff": plot_cutoff(model),
        "abb2_forward_selection": plot_forward(model),
        "abb3_gemeinsames_basismodell": plot_common(model),
    }
    if STRAT_PATH.exists():
        strat = json.loads(STRAT_PATH.read_text(encoding="utf-8"))
        figures["abb4_elo_stratifizierung"] = plot_stratification(strat)
    if TIER_PATH.exists():
        tier_data = json.loads(TIER_PATH.read_text(encoding="utf-8"))
        figures["abb5_tier_faktoren"] = plot_tier(tier_data)
    for name, fig in figures.items():
        for ext in ("png", "pdf"):
            path = FIG_DIR / f"{name}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)
        print(f"gespeichert: {FIG_DIR / (name + '.png')}  (+ .pdf)")


if __name__ == "__main__":
    main()
