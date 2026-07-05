# Studie-League-Algorithmus

This repository contains two related but separate Python projects in one monorepo:

1. `src/` holds the correlation and team-strength logic for the study project.
2. `Riot-Crawler/` holds the Riot API crawler, dataset builder, and analysis scripts.

Keeping both projects in the same repository is a good fit here because they share the same research topic and data flow, but they should stay clearly separated by folder, entry point, and documentation.

## Repository Layout

```text
.
├── src/                     # Correlation and scoring logic
└── Riot-Crawler/            # Data collection and analysis pipeline
	├── crawler/             # Riot API access and fetching
	├── analysis/            # Dataset building and statistical analysis
	└── data/                # Raw and processed data artifacts
```

## Project 1: Correlation Model

The code in `src/` is the lightweight algorithmic part of the repo. It focuses on calculating player and team strength based on normalized features such as rank, LP, win rate, match volume, role consistency, and team composition.

Run it from the repository root with:

```bash
python src/main.py
```

Required packages for this part:

```bash
pip install numpy pandas
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