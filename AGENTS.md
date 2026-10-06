# Project context: Chess Rating Predictor (DAT158, HVL)

You are a coding agent working in a GitHub Codespace. Your job is to set up and build a complete, reproducible machine learning project. Read this whole file before doing anything, and keep it as the source of truth for the project. Work step by step, and stop to ask me before making major design decisions that are not covered here.

## 1. What we are building

A website where a user submits a chess game, and an ML model estimates the playing strength (rating) of White and Black **from the game itself**, without seeing the real ratings.

Example output: `White: ~1650 (±200) · Black: ~1480 (±200)`. If the submitted game contained real ratings, the app hides them from the model, and afterwards reveals them for comparison: "Model guessed 1620, actual rating was 1580".

## 2. Course requirements (must all be satisfied)

This is the final project in DAT158 (Machine Learning) at HVL. Deadline: **30 October 2026**.

- The model must be trained by us, following the full ML project lifecycle (problem framing, data, EDA, features, baseline, model comparison, evaluation, deployment). Using libraries like scikit-learn is fine; using a pre-trained third-party model as-is is not.
- Everything must live in a **public GitHub repository**: code, report, and documentation.
- **Anyone must be able to reproduce the results and deploy the website** from the README alone.
- The final product is a **website** that takes user input and returns an ML-based result, preferably **deployed publicly** (we will use Hugging Face Spaces with Gradio).
- A short **report** documenting the work goes in the repo, following the course template (max 3500 words, see §10).
- The assignment says groups of 2–3; **this project is done solo** (confirm with the lecturer). The submission on Canvas is a link to the repo.
- The course follows Géron, *Hands-On Machine Learning* (ch. 2 End-to-End ML Project, **Appendix A: ML Project Checklist**). Use the checklist as a guide for the lifecycle and refer to course concepts (bias/variance, regularization, ensembles, learning curves, random vs. grid search) when justifying decisions.
- Besides the live deployment, record a short **demo video/screencast** as a backup in case the Space is down when graded.
- AI coding tools are allowed, but **sources and AI usage must be cited**. Keep a running log in `report/ai_usage.md` of what you (the agent) built, so I can document it honestly.

The grading weight is on the ML process and the reasoning behind decisions, not on frontend polish. Keep the frontend simple and spend effort on correct, well-documented ML.

## 3. Data

**Source:** Lichess open database, https://database.lichess.org/ — released under **Creative Commons CC0**, free to use for any purpose. Cite it in the README and report.

**Files:** Monthly standard rated games, `.pgn.zst` format, URL pattern:
`https://database.lichess.org/standard/lichess_db_standard_rated_YYYY-MM.pgn.zst`

**Critical constraint — file size:** Recent monthly files are tens of GB compressed with ~90 million games. **Never download a full file.** Instead, stream the file over HTTP, decompress on the fly with `zstandard`, parse with `python-chess`, and stop after N games. Codespace disk and memory are limited.

**Dataset size:** The final model uses the **first 250,000 games that pass the filters** in the chosen month (= 500,000 per-player rows). Streaming stops as soon as N is reached. Use a small N (e.g. 20,000) during development so iterations are fast. N is set in `config.yaml` and can be overridden on the command line. Because the monthly file is chronological, these games come from the first day or two of the month — document this as a limitation.

**Use a recent month** (2024 or later) so games include `%clk` clock annotations. A fraction of games also include `%eval` engine evaluations.

**Filtering rules** (apply while streaming, count how many games each rule removes, and log the counts — they go in the report):
- Standard chess only, rated games only.
- One time control family: **blitz** (use the `Event` header containing "Blitz", and/or `TimeControl` base time of 180–300 seconds). Store the exact `TimeControl` anyway.
- Both `WhiteElo` and `BlackElo` present and numeric.
- Exclude games with `Termination` = "Abandoned" or "Rules infraction".
- Exclude games shorter than 10 full moves (20 plies).

Save the processed dataset as Parquet in `data/processed/`. Raw/processed data must be in `.gitignore`; only small sample files (e.g. `data/sample/sample_games.pgn`, a few dozen games) are committed.

## 4. Data leakage — the most important rule

PGN headers contain the answer. The following must **never** be used as features or passed to the model, in training or in the app:

`WhiteElo`, `BlackElo`, `WhiteRatingDiff`, `BlackRatingDiff`, `WhiteTitle`, `BlackTitle`, player names (`White`, `Black`), `Site`/URLs, and anything derived from them.

Implementation requirements:
- Feature extraction must work **only from the moves, the result, and move comments (clocks/evals)**. It should not read rating-related headers at all.
- Additionally provide `strip_rating_headers(pgn: str) -> tuple[str, dict]` that removes the headers above and returns the cleaned PGN plus the removed true ratings (for the reveal feature in the app). Apply it in the backend; never rely on the frontend alone.
- Write a unit test that proves features are identical whether or not rating headers are present.

## 5. Problem framing

**Task:** Regression — predict a player's rating.

**Row design:** One row **per player per game** (each game produces two rows). Each row contains the player's own features, the opponent's features, and a `color` flag. Target: that player's rating. This doubles the data and lets one model serve both colors.

**Train/test split:** Use `GroupShuffleSplit` grouped by **game ID**, so both rows from the same game always land in the same split. Note in the report that the same Lichess user may appear in both splits; optionally run a second evaluation grouped by username (usernames used **only** for grouping, never as features).

Fix random seeds everywhere and keep all settings in `config.yaml`.

## 6. Features

All feature code lives in **one module, `src/features.py`**, used by **both** training and the app, so a pasted game is processed exactly like training data.

**Feature set A — always available (required, used by the deployed model):**
- Game length in plies, result (from the player's perspective: win/draw/loss), whether the game ended in checkmate (detect with `board.is_checkmate()`), whether the player was checkmated.
- Move statistics per player: number and rate of captures, checks, promotions; castling (yes/no, which side, on which move); number of queen moves in the first 10 moves; number of distinct pieces moved in the opening.
- Material balance over time (computed by replaying moves): material difference at move 20, 30, 40 and at the end; the largest material deficit the player recovered from.
- Clock features (if `%clk` present, otherwise NaN): average and standard deviation of time per move, time used in the first 10 moves, time remaining at the end as a fraction of base time, fraction of moves played with under 10 seconds left, base time and increment parsed from the clocks or TimeControl.
- Opening: ECO code from the header **only if present**, otherwise derive it from the moves using the Lichess `chess-openings` dataset (https://github.com/lichess-org/chess-openings, CC0), otherwise "unknown". Treat as categorical.

**Feature set B — engine-based (optional, phase 2):**
- From `%eval` comments: average centipawn loss per player, number of inaccuracies/mistakes/blunders (use clear thresholds, e.g. 50/100/300 cp swings, and document them), accuracy in the first 15 moves vs later.
- Only build this after set A works end-to-end. Train it as a separate model on the subset of games with evals, and compare it to set A in the report. The deployed app uses set A unless we later add Stockfish to the app.

The tree-based models handle NaN natively; for linear models, impute and add "is_missing" indicator columns.

## 7. Modelling and evaluation

Compare, in this order:
1. **Baseline:** `DummyRegressor` (predict the training mean). Every other model must be compared against it.
2. **Ridge regression** (with scaling, one-hot encoding for categoricals) — simple, interpretable.
3. **Random forest.**
4. **Gradient boosting:** `HistGradientBoostingRegressor` (scikit-learn), optionally LightGBM.

Use scikit-learn `Pipeline`/`ColumnTransformer` so preprocessing is saved together with the model. Do light hyperparameter tuning for the best model with cross-validation on the training set only (keep it cheap: `RandomizedSearchCV` with a small budget, grouped CV).

**Metrics:** MAE (primary, in rating points), RMSE, R². Report all models in one table.

**Experiments for the report** (save figures to `report/figures/`, tables as CSV/Markdown):
- Model comparison table.
- **Learning curve:** MAE of the best model trained on 10k, 50k, 100k, 200k, (500k) games. This justifies why we did not use all 90M games.
- Feature importance (permutation importance on the test set).
- Error analysis: MAE by true rating bucket (e.g. <1000, 1000–1500, 1500–2000, 2000+) and by game length. Expect the model to regress toward the mean; show it.
- Prediction vs. actual scatter plot.
- (Phase 2) Set A vs. set A+B.

The uncertainty interval shown in the app (`±X`) should come from the test-set MAE (or a quantile model, if time allows) — not a made-up number.

Save the final model with `joblib` to `models/`. It must be small enough to commit (keep it well under 100 MB) so the Space can load it directly.

## 8. The web app

**Framework:** Gradio, deployed to **Hugging Face Spaces**. Entry point `app/app.py` (also expose a root-level `app.py` or configure the Space so it runs).

**Inputs (tabs or one form):**
1. Paste PGN text.
2. Upload a `.pgn` file.
3. Paste a Lichess game URL or game ID → fetch the PGN from the Lichess API: `GET https://lichess.org/game/export/{gameId}?clocks=true&evals=true` (plain text PGN, no auth needed). Handle errors and rate limits gracefully.

**Processing:** strip rating headers (keep the true values aside) → `features.py` → model → predictions.

**Outputs:**
- Estimated rating for White and Black with the ± interval.
- If true ratings were found in the input: show them after the prediction, with the error.
- A small note on limitations (one game is limited evidence; model trained on Lichess blitz; ratings on other sites/time controls differ).
- Validate input: clear error messages for invalid PGN, non-standard variants, or very short games.

Include 2–3 example games (from the sample file) as Gradio examples so the demo works with one click.

Keep the UI simple. No custom frontend framework needed.

## 9. Repository structure

```
chess-rating-predictor/
├── AGENTS.md                 # this file
├── README.md                 # what, why, how to reproduce, how to deploy, live link
├── config.yaml               # month, N games, filters, seeds, paths
├── requirements.txt          # pinned versions
├── Makefile                  # make data / features / train / evaluate / app / test
├── data/
│   ├── sample/sample_games.pgn   # small, committed
│   ├── raw/                  # gitignored
│   └── processed/            # gitignored
├── src/
│   ├── download.py           # stream + filter + parse N games from Lichess
│   ├── pgn_utils.py          # strip_rating_headers, parsing helpers
│   ├── features.py           # shared feature extraction (training + app)
│   ├── dataset.py            # build per-player rows, train/test split
│   ├── train.py              # train + compare models, save best
│   └── evaluate.py           # metrics, figures, tables for the report
├── notebooks/
│   └── 01_eda.ipynb          # exploratory data analysis
├── models/                   # final model (committed)
├── app/
│   └── app.py                # Gradio app
├── tests/
│   ├── test_pgn_utils.py
│   └── test_features.py
└── report/
    ├── report.md             # report skeleton (see below)
    ├── ai_usage.md           # log of AI-assisted work
    └── figures/
```

## 10. Report skeleton (`report/report.md`)

Create the skeleton with headings and short TODO notes only. **Do not invent results, numbers, or conclusions** — results sections are filled in from actual output of `evaluate.py`.

The report **must follow the official course template** ("Template DAT158 assignment 2.docx") and stay **under 3500 words**. Title page: project name, group members' names, date. Use exactly these numbered sections (the template's guiding questions are listed so each one gets answered or consciously skipped):

1. **Describe the problem**
   - *Scope:* goal; why ML is a promising solution; who uses it and how; existing solutions / how it is done today (e.g. Kaggle "Finding Elo" — verify and cite); how a human would do it without ML; "business objective"; system components (download → features → model → app) and how changes in one affect the others; resources needed (compute, people).
   - *Metrics:* ML metrics (MAE primary, RMSE, R²) and software metrics (prediction latency in the app); how they connect to the objective; the minimal performance for success (e.g. clearly beating the DummyRegressor baseline — define the threshold before seeing results).
2. **Data** — source, license (CC0), how labels (ratings) are obtained and how accurate/consistent they are (Glicko-2, provisional ratings, rating noise); filtering steps with counts; final dataset size and how much data is needed (learning curve); privacy/ethics (usernames, never used as features); representation, cleaning, feature engineering, scaling; leakage prevention; EDA highlights.
3. **Modeling** — models explored; baseline (DummyRegressor, optionally a simple non-ML heuristic and a rough human-level estimate if a source exists); all metrics in tables/graphs; learning curve; hyperparameter search; error analysis and feature importance, and how they informed improvements.
4. **Deployment** — how the model is deployed (Gradio on HF Spaces); how predictions are used; monitoring and maintenance (data drift as the Lichess rating pool changes, retraining on newer months); planned improvements; link to the live site and screenshots.
5. **References** — all sources (Lichess database, chess-openings, Géron, scikit-learn, Kaggle, etc.) and the AI usage statement.

Place limitations/discussion where the template allows (end of Modeling or Deployment) rather than adding extra top-level sections.

## 11. How to work

- Start by setting up the environment (Python 3.13 (matches the local machine; pin the same version on HF Spaces), virtualenv, pinned `requirements.txt`: `python-chess`, `zstandard`, `requests`, `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`, `pyyaml`, `gradio`, `pytest`, optionally `lightgbm`).
- Build in this order, and verify each step before moving on:
  1. `download.py` working on a small N, with filter counts logged.
  2. `pgn_utils.py` + tests (leakage test included).
  3. `features.py` + tests, run on the sample file.
  4. `dataset.py` producing the per-player table.
  5. `train.py` with baseline + models; print the comparison table.
  6. `evaluate.py` with all figures and tables.
  7. Minimal Gradio app running locally in the Codespace.
  8. Deployment instructions for Hugging Face Spaces in the README.
  9. Report skeleton and AI usage log.
- Get a thin end-to-end version working first (small data, baseline + one model, bare app), then improve.
- Keep functions small, typed, and documented. Comment the *why*, not the *what*.
- Make every script runnable from the Makefile and configurable via `config.yaml`, so the README instructions are just a few commands.
- After each major step, update `report/ai_usage.md` with a one-line summary of what you produced.
- If something is ambiguous or a step fails in a way that changes the plan, stop and ask me.
