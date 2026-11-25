# Studie-League-Algorithmus

This repository contains two related but separate Python projects in one monorepo:

1. `src/` holds the correlation and team-strength logic for the study project.
2. `Riot-Crawler/` holds the Riot API crawler, dataset builder, and analysis scripts.

Keeping both projects in the same repository is a good fit here because they share the same research topic and data flow, but they should stay clearly separated by folder, entry point, and documentation.

## Repository Layout

```text
.
├── src/                     # Strength model, team builder and benchmarks
├── players_example.csv      # Input template for the team builder
└── Riot-Crawler/            # Data collection and analysis pipeline
	├── crawler/             # Riot API access and fetching
	├── analysis/            # Dataset building and statistical analysis
	└── data/                # Raw and processed data artifacts
```

## Project 1: Strength Model and Team Builder

The code in `src/` derives empirical factor weights from the crawled match data, uses them to calculate individual player strength, and forms League teams from a player list.

| Module | Purpose |
| --- | --- |
| `data_prep.py` | Loads, cleans and splits `matches.csv` from the crawler |
| `regression.py` | Derives the factor weights via logistic regression (cross-validated AUC) and writes `weights.json` |
| `main.py` | Calculates individual and team strength with the weights from `weights.json` |
| `stratification.py`, `tier_analysis.py` | Check whether the weights are stable across Elo bands and individual tiers |
| `plots.py` | Figures for the factor selection (Delta-AUC, forward selection, tier heatmap) |
| `team_builder.py` | Builds teams from a player CSV (see below) |
| `benchmark_gurobi.py` | Benchmark: Simulated Annealing vs. the exact Gurobi solver |
| `webapp.py` | Local web interface for the team builder (see below) |

Required packages for this part:

```bash
pip install numpy pandas scipy scikit-learn matplotlib requests
pip install gurobipy   # optional, needs a (free academic) Gurobi license
```

Run the strength model from the repository root with:

```bash
python src/regression.py   # once, creates src/weights.json
python src/main.py
```

### Team Builder

`team_builder.py` reads a player list, fetches each player's Solo/Duo rank via the Riot API (EUW), calculates their strength with `weights.json`, and builds two lineups:

1. **Strongest teams**: strength is bundled (team 1 is the strongest).
2. **Balanced teams**: team strengths are as equal as possible.

Both modes are solved twice: heuristically (sorting and Simulated Annealing) and, if `gurobipy` is installed, exactly with Gurobi.

```bash
python src/team_builder.py players_example.csv        # Gurobi time limit 60 s per step
python src/team_builder.py players_example.csv 120    # custom time limit in seconds
```

A Riot API key is required, read from `RIOT_API_KEY`, `Riot-Crawler/.env` or `riot.txt`. Development keys expire after 24 hours.

**Web interface:** the same algorithm with a browser UI. Enter players in a table or import a CSV (drag and drop works), then view the teams, KPIs and strength comparison and download them as CSV. It needs no extra packages and runs only on `127.0.0.1`.

```bash
python src/webapp.py              # opens http://127.0.0.1:8765/
python src/webapp.py 9000 --no-browser
```

The API key can also be entered in the UI. That key is only kept in memory for the running session. Ranks are cached for 15 minutes, so recalculating after editing preferences is instant.

**Input format** (see [players_example.csv](players_example.csv)), semicolon- or comma-separated, header row required:

| Column | Required | Content |
| --- | --- | --- |
| `RIOT ID` | yes | `Name#Tagline` |
| `Main Position` | yes | one role: Top, Jungle, Mid, ADC (or Bot), Support |
| `Second Positions` | no | further roles, comma-separated |
| `With` | no | Riot IDs this player must be in the same team with, comma-separated |
| `Not With` | no | Riot IDs this player must not be in the same team with |

- Role names accept common synonyms (`jgl`, `bot`, `supp`, ...).
- The preference columns may also be called `Mit` / `Nicht mit` or written as `not_with`, `Not-With`, `NotWith`.
- Riot IDs in preferences ignore case and spaces.
- A preference applies to both players, so it only has to be entered once.

**Priorities** (strictly in this order):

1. **Preferences are hard constraints** and are always fulfilled 100 %. `With` means both play in the same team (neither is a reserve). If necessary, a player gets a role they did not list.
2. As few players as possible on a role they did not list.
3. As few players as possible on their second role.
4. The goal of the mode (strongest or balanced teams).

**Things to know:**

- **Player count:** If the number of players is not a multiple of 5, the surplus players become reserves. They are chosen by the priorities above; on a tie, the weakest players sit out.
- **Impossible preferences:** If the preferences cannot be fulfilled, the program stops with an explanation. This happens for a `With` group of more than 5 players, a `With` chain that contradicts a `Not With`, or more players in groups than there are slots.
- **Unranked players:** Players without a Solo/Duo rank get the model's median strength.

**Output:** the teams per variant, the fulfilled preferences, and a comparison of the spread of team strengths. Four files are written next to the input CSV:

- `<name>_staerkste_teams.csv`
- `<name>_ausgeglichene_teams.csv`
- `<name>_staerkste_teams_gurobi.csv`
- `<name>_ausgeglichene_teams_gurobi.csv`

Each file lists team, role, rank, strength and role type (main, second role, or not listed) per player, plus the reserves.

### Benchmark

```bash
python src/benchmark_gurobi.py            # scaling benchmark on synthetic player pools
python src/benchmark_gurobi.py duel 30    # SA vs. Gurobi with the same time budget
python src/benchmark_gurobi.py players.csv 60   # comparison on the players of a CSV
```

## Project 2: Riot-Crawler

The `Riot-Crawler/` folder is a separate pipeline for collecting Riot data and turning it into analysis-ready datasets.

Typical workflow:

1. Crawl matches and rank data with `Riot-Crawler/main.py`.
2. Build the dataset with `Riot-Crawler/analysis/build_dataset.py`.
3. Run the statistical analyses in `Riot-Crawler/analysis/`.
4. Generate the report with `Riot-Crawler/analysis/generate_report.py`.

Run it from inside the crawler folder:

```bash
cd Riot-Crawler
pip install -r requirements.txt
python main.py
```

The analysis scripts also use `matplotlib` and `scikit-learn`. Install them as needed:

```bash
pip install matplotlib scikit-learn
```

## Data And Outputs

Generated data should stay out of version control:

- crawler caches such as account and rank caches
- raw match dumps
- processed CSV and Parquet files
- analysis reports and temporary outputs

If you need to share a sample dataset, add it intentionally instead of committing the full generated output tree.

## Recommended Working Style

Use separate virtual environments for the two projects if you work on both frequently. That keeps the crawler dependencies and the correlation scripts isolated while still letting the repo stay unified.