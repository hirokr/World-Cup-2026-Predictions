**World Cup 2026 Predictions — Cards, Scores & Results**

**Overview:** This repository contains data and notebooks used to predict FIFA World Cup 2026 match outcomes: yellow/red cards, scores, and final results. It uses historical match and player data (2016–2025) to build features, train models, and evaluate predictions.

**Data:**

- **data/**: primary CSV files used for modelling and analysis (goalscorers, results, shootouts, etc.).
- **data/worldcup/data-csv/**: extracted World Cup datasets (matches, players, teams, squads, stadiums, bookings, etc.).

**Key Files:**

- **football_data_scraper_2016_2026.py**: data scraping / ingestion script used to collect and prepare data.
- **wc2026_prediction_notebook.ipynb**: main notebook with EDA, feature engineering, modeling, and evaluation.
- **cleaning.ipynb**: exploratory cleaning steps and lightweight transformations.

**Quick Start:**

1. Create a Python virtual environment and install dependencies (create `requirements.txt` if not present):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) Re-run the scraper to refresh raw data:

```bash
python football_data_scraper_2016_2026.py
```

3. Open and run the notebooks to reproduce analysis and model training:

```bash
jupyter lab
# then open wc2026_prediction_notebook.ipynb or cleaning.ipynb
```

**Modeling & Approach:**

- Exploratory data analysis to understand distributions and match-level signals.
- Feature engineering at team- and match-level (recent form, head-to-head, player suspensions/injuries where available).
- Separate tasks: regression for scores, classification for win/draw/loss, and classification/regression for card counts.
- Cross-validation and careful time-aware splitting to avoid look-ahead bias.

**Reproduce Results:**

- Run the cells in `wc2026_prediction_notebook.ipynb` in order. The notebook contains sections for data loading, preprocessing, model training, and evaluation. Checkpoint or save intermediate datasets back to `data/` to speed iteration.

**Notes & Next Steps:**

- Add a `requirements.txt` listing packages used (pandas, scikit-learn, xgboost, jupyter, etc.).
- Add a simple CLI or script to train and export models for batch predictions.
- Add evaluation notebooks that produce submission-ready CSVs for leaderboard/testing.

**Contact:**
For questions or collaboration, open an issue or contact the repo owner.

**Final Submission Pipeline:**

```bash
# Refresh official Elo ratings and FIFA squad lists
python refresh_external_data.py

# Retrain, run 50,000 tournament simulations, and regenerate the notebook
python build_submission.py
```

Optional current-data files are stored under `data/`:

- `market_odds_2026.csv`: decimal or American home/draw/away odds.
- `international_match_stats_2026.csv`: national-team corner and card estimates.
- `injuries_2026.csv`: confirmed absences with numeric impact values.
- `referee_assignments_2026.csv`: match officials and historical card rates.

Empty optional files have no effect. Populated rows are validated and blended
automatically. Generated backtests are written to `comp-notebook/`.
