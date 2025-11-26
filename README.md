# Machine Learning Basket

This repository contains code and datasets to build models to predict team success (e.g., making the playoffs) using player and team features derived from historical basketball seasons.

---

## Table of Contents
- Project Overview
- Data
- Features Description
- Setup & Installation
- How to Run
- Results Summary (Models)
- Project Structure
- Contributing
- License

---

## Project Overview

This project implements a pipeline to:

- Clean and organize raw data across seasons
- Create player-level and team-level features
- Train classification and regression models to predict team outcomes (playoffs, performance, etc.)
- Evaluate models across time (year-based, time-series cross-validation)

The repository includes scripts for: cleaning (`src/cleaning.py`), data organization (`src/organize_year.py`), feature creation (`src/features.py`), model training and evaluation (`src/modeling.py`), and orchestration (`src/main.py`).

---

## Data

Data files are included under `data/raw` and more organized versions of the data are generated during the cleaning and organization steps (and saved under `data/year/`). The raw files include:

- `awards_players.csv`
- `coaches.csv`
- `players_teams.csv`
- `players.csv`
- `series_post.csv`
- `teams_post.csv`
- `teams.csv`

There is also a `data/year11` directory used for tests or specific year analyses.

Note: The repository expects the data to be in `./data/` when running the scripts. The `main.py` orchestrator calls the cleaning, organization, preparation and model prediction steps and saves outputs under `data/year/combined/prediction` and `data/year/combined/result`.

---

## Features Description

Player features (some feature names are Portuguese inspired):

- `awards_score` — Score for awards received by the player.
- `regular_season_score` — Score derived from regular season stats.
- `post_season_score` — Score derived from post season stats.
- `FG%`, `3P%`, `FT%` — Shooting percentages.
- `efficiency` — Field goal efficiency based score.
- `total_experience` — Experience score (years played + awards, etc.).
- `player_score` — Overall player quality score.
- `pre_season_player_score` — The average player score from previous seasons.

Team features:

- `o_score` — Offensive score for the current season.
- `d_score` — Defensive score for the current season.
- `pf_score` — Performance score (e.g., wins/losses ratio).
- `team_stats_score` — Team statistics score for the current year.
- `pre_season_team_stats_score` — Average team stats score across previous seasons (excludes current year).
- `all_time_team_players_score` — Average player_score for the team's roster across previous seasons (excludes current year).
- `current_team_players_score` — Roster player_score for the current season before start (uses players' `pre_season_player_score`).
- `team_players_score` — Average of `player_score` including current year.
- `team_score` — Combined score defined as 0.5 * team_players_score + 0.5 * team_stats_score.

These features are calculated during the data preparation pipeline and combined into `data/year/combined/combined_team_data.csv` before modeling.

---

## Setup & Installation

1. Install Python 3.10+ (3.11 recommended).
2. (Optional but recommended) Create and activate a virtual environment.

   Windows PowerShell example:
   ```powershell
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies. (The `requirements.txt` file may be empty or incomplete; below is a minimal list of packages required by the code.)

   Example dependencies to install:
   ```powershell
   pip install pandas numpy scikit-learn matplotlib imbalanced-learn xgboost lightgbm catboost
   ```

4. (Optional) Add more packages depending on scripts you run, notably `imblearn` for SMOTE and `matplotlib` for plotting.

---

## How to Run

From the repository root, run:

```powershell
python -m src.main
```

This will:

- Run the cleaning pipeline to produce cleaned datasets
- Organize data by season/year
- Prepare yearly combined team data
- Remove previous predictions under `./data/year/combined/prediction`
- Run prediction models for the configured models (e.g., logistic_regression, svm, catboost)
- Save predictions under `./data/year/combined/prediction` and results under `./data/year/combined/result`.

You can also run specific scripts manually, e.g., `src/modeling.py` functions if you want to run one model at a time or run the time-series CV for regression or classification.

---

## Results Summary (Model Performance)

Below are the classification results that were included in the original README. These were calculated using different models — Random Forest, XGB, Logistic Regression, and SVM — across years 8, 9, and 10 for different features and pre-season/team metrics.

### Random Forest Classifier
- `pre_season_team_stats_score` - Year 8: 0.588 | Year 9: 0.729 | Year 10: 0.575
- `current_team_players_score` - Year 8: 0.375 | Year 9: 0.531 | Year 10: 0.625
- `all_time_team_players_score` - Year 8: 0.4 | Year 9: 0.521 | Year 10: 0.662
- `pre_season_team_score` - Year 8: 0.5 | Year 9: 0.562 | Year 10: 0.325

### XGB Classifier
- `pre_season_team_stats_score` - Year 8: 0.55 | Year 9: 0.76 | Year 10: 0.538
- `current_team_players_score` - Year 8: 0.425 | Year 9: 0.688 | Year 10: 0.588
- `all_time_team_players_score` - Year 8: 0.438 | Year 9: 0.615 | Year 10: 0.588
- `pre_season_team_score` - Year 8: 0.512 | Year 9: 0.708 | Year 10: 0.525

### Logistic Regression
- `pre_season_team_stats_score` - Year 8: 0.538 | Year 9: 0.854 | Year 10: 0.512
- `current_team_players_score` - Year 8: 0.4 | Year 9: 0.667 | Year 10: 0.9
- `all_time_team_players_score` - Year 8: 0.55 | Year 9: 0.792 | Year 10: 0.65
- `pre_season_team_score` - Year 8: 0.55 | Year 9: 0.833 | Year 10: 0.587

### SVM
- `pre_season_team_stats_score` - Year 8: 0.538 | Year 9: 0.854 | Year 10: 0.512
- `current_team_players_score` - Year 8: 0.6 | Year 9: 0.667 | Year 10: 0.1
- `all_time_team_players_score` - Year 8: 0.55 | Year 9: 0.792 | Year 10: 0.65
- `pre_season_team_score` - Year 8: 0.55 | Year 9: 0.833 | Year 10: 0.587

These scores represent performance metrics (AUC, accuracy, etc.) for the selected features. The exact metric used for the reported score depends on the experiments executed — the code usually prints the AUC-ROC and detailed classification reports.

---

## Project Structure

Key source files:

- `src/main.py` — Orchestrator for cleaning, organizing, and running models
- `src/cleaning.py` — Cleaning steps to process raw files
- `src/organize_year.py` — Functions to reorganize data by years and perform year-specific transformations
- `src/features.py` — Feature engineering (player/team features)
- `src/data_preparation.py` — Combine and prepare features with targets for each year
- `src/modeling.py` — Model training, evaluation, and prediction functions
- `src/all_models.py` — Helper script to aggregate predictions from multiple models

---

## Contributing

Contributions are welcome. If you add features, please:

1. Open an issue to discuss major changes or new experiments
2. Add tests where possible and ensure existing code still runs
3. Update this README if you change how to run or install the project

---

## License

This repository currently does not include a license file. Please add a `LICENSE` if you plan to open source this code for public reuse.