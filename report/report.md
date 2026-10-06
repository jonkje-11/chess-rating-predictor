# Chess Rating Predictor

Jonas _TODO: full name_, _TODO: date_

<!-- Follows the DAT158 assignment 2 template. Max 3500 words. Results are filled in only from evaluate.py output. -->

## 1: Describe the problem

### Scope

- TODO: Goal: estimate both players' rating from a single game, without seeing the ratings.
- TODO: Why ML is a promising approach, and how a human (e.g. a coach looking at a game) would do this without ML.
- TODO: Who would use it and how (players curious about a game, coaches, games from offline/OTB play without a rating).
- TODO: Existing solutions / related work (verify and cite, e.g. Kaggle "Finding Elo").
- TODO: Business objective (adapted to a student project) and system components: download → features → model → app, and how a change in one affects the others.
- TODO: Resources: one person, a laptop, free hosting on Hugging Face Spaces.

### Metrics

- TODO: ML metrics: MAE in rating points (primary), RMSE, R².
- TODO: Software metrics: prediction latency in the app.
- TODO: Minimum performance for success, defined before seeing results (relative to the DummyRegressor baseline).

## 2: Data

- TODO: Source and licence (Lichess open database, CC0), month used, streaming approach.
- TODO: Labels: Lichess Glicko-2 ratings: how they are produced, provisional ratings, noise.
- TODO: Filtering steps with counts (from download.py log).
- TODO: Final dataset size; how much data is needed (learning curve).
- TODO: Privacy/ethics: usernames are public but never used as features.
- TODO: Representation: per-player rows, feature engineering, missing values, scaling.
- TODO: Data leakage prevention.
- TODO: EDA highlights (from notebooks/01_eda.ipynb).
- TODO: Limitation: games come from the first days of one month.

## 3: Modeling

- TODO: Models explored: DummyRegressor baseline, Ridge, Random Forest, HistGradientBoosting (LightGBM optional).
- TODO: Train/test split grouped by game; optional evaluation grouped by player.
- TODO: Hyperparameter search (random search, grouped CV).
- TODO: Model comparison table.
- TODO: Learning curve.
- TODO: Feature importance (permutation importance).
- TODO: Error analysis by rating bucket and game length; regression toward the mean.
- TODO: Limitations.

## 4: Deployment

- TODO: Gradio app on Hugging Face Spaces; inputs, processing, outputs.
- TODO: How the ± interval is derived.
- TODO: Monitoring and maintenance (rating pool drift, retraining on newer months).
- TODO: Planned improvements.
- TODO: Link and screenshots.

## 5: References

- Lichess open database. https://database.lichess.org/ (CC0)
- lichess-org/chess-openings. https://github.com/lichess-org/chess-openings (CC0)
- Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3rd ed.). O'Reilly. Ch. 2 and Appendix A.
- Pedregosa et al. Scikit-learn: Machine Learning in Python. JMLR 12, 2011.
- TODO: further sources.

### AI usage statement

TODO: summary of [ai_usage.md](ai_usage.md).
