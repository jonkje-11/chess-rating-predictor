# Chess Rating Predictor: estimating playing strength from a single game

**Jonas** _(TODO: full name)_, 30 October 2026. DAT158 Machine Learning, assignment 2 (project work), HVL.

Repository: https://github.com/jonkje-11/chess-rating-predictor · Live app: _TODO: Hugging Face Spaces link_

---

## 1: Describe the problem

### Scope

**Goal.** Build a website where a user submits one chess game (PGN text, a `.pgn` file or a Lichess link), and a machine-learning model estimates the rating of both players **from the game itself**, without seeing their real ratings. Example output: *White ~1650 (±225), Black ~1480 (±225)*. If the input contains the real ratings, they are hidden from the model and shown afterwards for comparison.

**Users and use.** Club players and online players who want a strength estimate for a game played without a rating (over-the-board casual games, games against friends, games from another site), coaches who want a quick read on an unknown student, and curious players who want to test whether "the moves give away the rating". The estimate is meant as an indication, not an official rating.

**How it is done today.** Ratings come from rating systems (Elo, Glicko-2) that use only *results* over many games, never the content of a game. Estimating strength from the moves is done informally by experienced players or coaches, who look at opening knowledge, tactical mistakes and time use; this needs expertise and is subjective. A related public challenge is the Kaggle competition *Finding Elo* (Kaggle, n.d.), where both players' Elo had to be predicted from a single game. Research such as Maia (McIlroy-Young et al., 2020) shows that move choices differ systematically between skill levels on Lichess, which supports the idea that a game carries a skill signal.

**Why machine learning.** The signal is spread over many weak indicators (time use, opening choice, material swings, game length) whose combination is hard to write down as rules, while millions of labelled examples (games with ratings) are freely available. This is a textbook supervised regression setting.

**"Business" objective.** For a free tool, the objective is a credible estimate that users find useful and come back to: estimates should usually be within one rating "class" (about ±200–250 points) and never wildly off for typical players. The product also has an educational purpose: showing what a single game can and cannot reveal.

**System components.** The system is a pipeline: (1) `download.py` streams and filters games from the Lichess database, (2) `features.py` turns a game into features, (3) `dataset.py` builds training rows and the split, (4) `train.py`/`evaluate.py` train, tune and evaluate models, and (5) `predict.py` + a Gradio app serve predictions. Changes propagate: a new filter changes the training distribution; a feature change requires rebuilding the dataset *and* retraining, and the app must use the identical feature code. That is why one module (`features.py`) is shared by training and the app, and why the model file stores the expected feature columns.

**Resources.** One student, a laptop (16 cores, no GPU), roughly one to two weeks of work, free hosting on Hugging Face Spaces, and free CC0 data. No paid compute was needed.

### Metrics

- **ML metrics.** Mean absolute error (MAE) in rating points is the primary metric because it is directly interpretable for users ("off by 225 points on average") and is the quantity shown as ± in the app. RMSE (penalises large misses) and R² (share of variance explained) are reported alongside.
- **Software metrics.** Prediction latency per request in the app (target: well under one second on free CPU hardware) and model file size (must be small enough to commit and load in a free Space).
- **Minimal success criterion.** The model must clearly beat the `DummyRegressor` baseline that predicts the mean rating; we set the bar at **at least 15% lower test MAE** than the baseline, measured on games and (separately) on players the model has never seen. _(TODO: confirm this threshold is the one you want to state.)_

## 2: Data

**Source and licence.** Games come from the Lichess open database (Lichess, 2026a), monthly files of all rated standard games, released under CC0. We use **September 2026**. A month is about 29 GB compressed (~90 million games), so the file is **never downloaded**: `download.py` streams it over HTTP, decompresses with `zstandard` on the fly, filters on header text, and stops after 250,000 kept games (about 180 MB transferred, 93 seconds).

**Labels.** The target is each player's Lichess **blitz rating** at the time of the game (`WhiteElo`/`BlackElo` headers). Lichess uses Glicko-2: new players start at 1500 with a large uncertainty, and a rating is shown as provisional while the rating deviation is above 110 (Lichess, 2026b). The labels are therefore noisy: a new account's rating can be hundreds of points from its true strength, and ratings move after every game. This label noise puts a floor under any achievable error.

**Filtering** (counts logged by `download.py`):

| Step | Games | Share of streamed |
|---|---:|---:|
| Streamed from the start of the file | 562,581 | 100% |
| Removed: not blitz (bullet, rapid, classical, …) | 302,489 | 53.8% |
| Removed: shorter than 20 plies | 8,686 | 1.5% |
| Removed: abandoned / rules infraction | 709 | 0.1% |
| Removed: base time outside 180–300 s | 697 | 0.1% |
| **Kept** | **250,000** | 44.4% |

**Dataset.** Each game gives **two rows**, one per player, with the player's own features (`own_*`), the opponent's features (`opp_*`), game-level features and a `color` flag: 500,000 rows, 58 features. One model therefore serves both colours. EDA (`notebooks/01_eda.ipynb`) shows:

- Ratings have mean 1622 and standard deviation 360 (range 400–3060).
- **Lichess pairs players of similar strength**: White and Black ratings correlate at 0.95 (median difference 30 points). The opponent's play therefore carries information about a player's own rating, which is why opponent features are included.
- **Time control is self-selected and informative**: mean rating 1713 in 3+0 versus 1439 in 5+3.
- **Stronger players move faster in the opening**: median time on the first 10 moves falls from 42 s (<1000) to 17 s (2000+).
- All games have clock annotations (`%clk`); only 9.8% have engine evaluations (`%eval`).

![Rating distribution](figures/eda_rating_distribution.png)

**Feature engineering** (`src/features.py`, feature set A). Features are computed by replaying the moves with `python-chess`:

- **Result:** result from the player's perspective and checkmate given or received.
- **Move statistics:** captures, checks and promotions (counts and rates); castling (side and move number); queen moves and number of distinct pieces moved in the first 10 moves.
- **Material balance:** balance at moves 20, 30 and 40 and at the end; the largest material deficit, and the largest deficit the player recovered from.
- **Clock use:** mean and standard deviation of time per move, time used on the first 10 moves, remaining time as a fraction of the base time, and share of moves made with under 10 s left.
- **Time control and termination:** base time, increment and termination type.
- **Opening:** the ECO code, taken from the header or derived from the moves with the Lichess opening book (Lichess, 2026c).

Missing values are natural: material at move 40 is missing in 67% of rows because the game ended earlier. Tree models handle NaN natively; for Ridge, missing values are median-imputed with indicator columns, numeric features are standardised and categoricals one-hot encoded, all inside scikit-learn `Pipeline`s, so preprocessing is saved with the model.

**Leakage prevention.** PGN headers contain the answer (`WhiteElo`, `BlackElo`, rating changes, titles, usernames, game URL). Protection is layered:

1. Feature code can only read an allow-list of four headers (`Result`, `TimeControl`, `ECO`, `Termination`), and any other access raises an error.
2. The app strips rating and identity headers **in the backend** before computing features.
3. Unit tests show that features and predictions are identical when rating headers are removed, or changed to other values, for all 40 committed sample games.
4. The train/test split is **grouped by game**, so the two rows of one game never end up on different sides. Otherwise a test row's opponent features would have been seen as own features in training.

**Privacy and ethics.** The data is public and CC0, but usernames identify real people. They are used only to group the "unseen players" evaluation and are never model inputs. The app has no database and does not keep submitted games (uploads only pass through Gradio's temporary file cache). A rating estimate from one game could be misused to judge people. The app therefore shows the uncertainty and a limitations note with every estimate.

**Limitations of the sample.** The file is chronological, so the 250,000 games all come from **1 September 2026, 00:00–07:15 UTC** (night in Europe, evening in the Americas). The player population at that hour may differ from the full month, and the model is restricted to Lichess blitz.

## 3: Modeling

**Split and selection protocol.** 80/20 train/test split grouped by game (400,000 / 100,000 rows). Models are compared on a grouped **validation** split (10% of the training games); the test set is used only to report final numbers. The best model is then tuned with grouped cross-validation on the training set and refit on the full training set.

**Models.** In order of complexity:

1. `DummyRegressor` (training mean)
2. Ridge regression
3. Random forest (100 trees, min. 20 samples per leaf)
4. `HistGradientBoostingRegressor` (HGB), with native categorical support and early stopping

These cover the course's progression from a regularised linear model to bagging and boosting ensembles.

**Baselines.** Besides the mean predictor, a non-ML heuristic predicts the mean training rating of the game's time control. It reaches MAE 281.4 (R² 0.07), only slightly better than the mean, which shows that the features below add far more than the time-control choice alone. No published human-level benchmark for this task was found, so human performance is not estimated.

**Results** (test set, 100,000 rows):

| Model | Validation MAE | Test MAE | Test RMSE | Test R² | Train MAE | Size |
|---|---:|---:|---:|---:|---:|---:|
| Mean baseline (Dummy) | 289.1 | 292.1 | 361.0 | 0.00 | 291.5 | 0 MB |
| Mean per time control (heuristic) | – | 281.4 | 348.2 | 0.07 | – | – |
| Ridge | 232.4 | 234.7 | 293.9 | 0.34 | 233.4 | 0.02 MB |
| Random forest | 236.9 | 238.7 | 297.7 | 0.32 | 219.5 | 78 MB |
| HGB (defaults) | 223.2 | 224.7 | 282.2 | 0.39 | 209.1 | 1.7 MB |
| **HGB tuned, refit on all training data (deployed)** | – | **224.8** | **282.1** | **0.39** | 216.7 | 1.4 MB |

HGB is best on validation and reduces MAE by **23%** relative to the mean baseline, which meets the success criterion. The random forest overfits more (largest train/test gap), is the weakest ensemble and is 78 MB, too large to deploy comfortably. Ridge is surprisingly close, so most of the signal is roughly additive.

**Hyperparameter search.** We ran a random search (course module 2: random rather than grid search, to cover more values per parameter with a small budget):

- **Budget:** 12 candidates × 3-fold grouped CV on 100,000 training games, about 5 minutes in total.
- **Parameters:** learning rate, number of iterations, number of leaves, minimum leaf size and L2 regularisation.

CV MAE varied only between 226.1 and 235.6 across the candidates. The best setting has few leaves (16) and strong L2 regularisation (4.3), i.e. a *simpler* model. It matches the default model's test MAE (224.8 vs. 224.7) with less overfitting (train MAE 216.7 vs. 209.1) and a smaller file. We deploy it for that reason. In bias–variance terms, the remaining error is mostly **bias / irreducible noise**, not variance, so more tuning has little to gain.

**Learning curve.** We trained the tuned HGB and Ridge on 10k–200k training games (all training games = 200k) and evaluated on the same test set. HGB improves from MAE 235.1 (10k games) to 228.4 (50k) and 224.8 (200k): each doubling of data now gains only about 1.5–2.5 points, and the gap between training and test error shrinks from 57 to 8 points, so the model is no longer variance-limited. Ridge is flat at about 235 with training error equal to test error, the signature of a high-bias model. This justifies using 250k games instead of the full 90 million per month: more data would buy at most a few points, while better *information per game* (see set B below) buys much more.

![Learning curve](figures/learning_curve.png)

**Generalisation to unseen players.** In the main split, 89.8% of test rows belong to players who also have (other) games in training. Usernames are not features, but a player's style could still be memorised indirectly. We therefore re-split so that 20% of usernames are held out entirely. Trained on 320,312 rows without any game involving a held-out player and evaluated on 99,841 rows of those players, the model reaches MAE **224.8** (R² 0.39), the same as on the standard split. The model has not memorised individual players; it generalises to new ones.

**Error analysis.** Error depends strongly on the true rating and shows clear **regression toward the mean**:

| True rating | Rows | MAE | Mean prediction | Bias |
|---|---:|---:|---:|---:|
| < 1000 | 5,040 | 437 | 1314 | +437 |
| 1000–1499 | 30,659 | 232 | 1498 | +204 |
| 1500–1999 | 49,716 | 163 | 1667 | −75 |
| 2000+ | 14,585 | 346 | 1825 | −341 |

When the evidence is weak, the MAE-minimising answer is close to the population mean, so beginners are overestimated and strong players underestimated. Error falls with game length, from MAE 243 for games under 40 plies to 213–216 for games over 100 plies, because longer games carry more evidence. Possible improvements are a model that sees several games per player, sample weighting of rare rating ranges, or reporting a quantile interval that widens at the extremes.

![Predicted vs actual](figures/pred_vs_actual.png)

**Feature importance.** Permutation importance on 20,000 test rows (increase in MAE when one feature is shuffled):

- The opening code (`eco`, +23 points) is the strongest single feature, because opening choice reflects knowledge.
- Clock features follow: the player's variability of time per move (+10), mean time per move (+8) and time used on the first 10 moves (+7), for both the player and the opponent. Base time (+4.5) captures the self-selection seen in the EDA.
- Material and move statistics matter less individually.
- Because own and opponent features are correlated, permutation importance *understates* each of them (shuffling one leaves its partner intact). Features with zero importance, such as `eco_group` and `castled`, are redundant given `eco` and `castle_move`, not useless.

![Permutation importance](figures/permutation_importance.png)

**Engine features (set B).** About 10% of games carry engine evaluations. On these games we computed average centipawn loss (overall, first 15 moves and later) and counts of inaccuracies, mistakes and blunders. The thresholds are a loss of at least 50, 100 and 300 centipawns per move, with evaluations capped at ±1000. Both models were trained with the tuned HGB settings on the same grouped split, restricted to games with evaluations (39,150 training rows, 9,760 test rows):

| Model (test rows with engine evals) | Test MAE | RMSE | R² |
|---|---:|---:|---:|
| Deployed model (set A, trained on all 400k rows) | 260.5 | 322.7 | 0.44 |
| Set A, trained on eval subset | 262.4 | 326.8 | 0.42 |
| **Set A+B, trained on eval subset** | **242.7** | **301.5** | **0.51** |

Engine features lower MAE by about 20 points (7.5%) at equal training size and raise R² from 0.42 to 0.51: move *quality* carries information that clocks and material cannot. (Errors are higher on this subset than overall because games that were analysed by an engine have a wider rating spread.) The deployed app still uses set A, because most submitted games have no evaluations and computing them would require running Stockfish on the server, but this is the most promising improvement.

**Limitations.**
- One game is limited evidence.
- The labels are noisy (provisional ratings).
- The sample covers seven hours of one day.
- The model only knows Lichess blitz: ratings from other sites or time controls are on different scales.
- Very short or unusual games (e.g. an early resignation) give the least information.

## 4: Deployment

**Architecture.**
- **App:** a Gradio app (`app/app.py`) with a custom theme and CSS, hosted on Hugging Face Spaces (free CPU). The repository itself is the Space, configured through the YAML block in `README.md`.
- **Inputs:** three tabs for pasting a PGN, uploading a file, or giving a Lichess URL/ID. For a link, the game is fetched from the Lichess export API (`/game/export/{id}?clocks=true`), with clear messages for unknown games, rate limiting (HTTP 429) and network errors.
- **Separation:** all ML logic is in `src/predict.py` (`predict_pgn(pgn) -> dict`, plain JSON data). The UI only formats the output, so it can later be replaced (e.g. a custom Tailwind page or a frontend on Vercel calling the Space's API) without touching the model.

**What happens on a request.**
1. Strip rating and identity headers, keeping the true values aside.
2. Validate the game (standard chess, legal moves, at least 20 plies).
3. Compute features with the same `features.py` used in training.
4. Predict both rows with the saved pipeline and clip to the 400–3100 range seen in training.
5. Show the estimates, the final position and, if available, the true ratings with the error.

The **± interval is the test-set MAE** (225 points) rather than a guessed number; the app states that about 58% of test estimates fall within it. The model file is 0.6 MB, and latency on the 40 sample games is 25 ms median (35 ms max) on a laptop CPU, well within the software target. Three example games (low, mid and high rated) from the committed sample make the demo usable in one click.

![App](figures/app_result.png)

**Monitoring and maintenance.** The Lichess rating pool drifts over time: rating inflation or deflation, new players, and changes in popular openings and time controls. Monitoring would track:
- the distribution of predictions and of input features (e.g. share of games without clocks, unknown ECO codes);
- whenever true ratings are available in the input, the live MAE against the true ratings, which is a free source of labelled feedback.

Retraining is a single command per month (`download → dataset → train → evaluate`), and the pinned `requirements.txt` and fixed seeds make it reproducible.

**Planned improvements.**
- A quantile model for honest, rating-dependent intervals.
- Engine features in the app (running Stockfish server-side), if set B proves worth it.
- Sampling games across the whole month instead of the first hours.
- Supporting other time controls with separate models.

## 5: References

- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly. Chapter 2 and Appendix A (Machine Learning Project Checklist).
- Kaggle (n.d.). *Finding Elo* [competition]. https://www.kaggle.com/c/finding-elo
- Lichess (2026a). *Lichess open database* (CC0). https://database.lichess.org/
- Lichess (2026b). *Frequently asked questions: ratings (Glicko-2, provisional ratings)*. https://lichess.org/faq
- Lichess (2026c). *chess-openings* (CC0). https://github.com/lichess-org/chess-openings
- McIlroy-Young, R., Sen, S., Kleinberg, J., & Anderson, A. (2020). Aligning superhuman AI with human behavior: chess as a model system. *Proceedings of KDD '20*. https://arxiv.org/abs/2006.01855
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR* 12, 2825–2830.
- Fiekas, N. *python-chess*. https://github.com/niklasf/python-chess
- Gradio (2026). https://www.gradio.app/ · Hugging Face Spaces. https://huggingface.co/spaces

**AI usage statement.** Claude Code (Anthropic, Claude Opus 5.5) was used as a coding assistant. It did the following, under the student's direction and review:
- wrote most of the code, tests, figures and README;
- drafted this report from the actual outputs of `evaluate.py`;
- proposed design choices, which the student decided on.

A dated log of what the AI produced is in `report/ai_usage.md`. All numbers in this report come from the scripts in the repository and can be reproduced with the commands in the README.
