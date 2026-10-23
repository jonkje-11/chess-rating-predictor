# ♞ Chess Rating Predictor

Estimate both players' **Lichess blitz rating from a single game**, using only the moves and the clock times. The model never sees the real ratings; if the game contains them, the app shows them afterwards for comparison.

Final project in DAT158 Machine Learning Engineering, Western Norway University of Applied Sciences (HVL), autumn 2026.

- **Live app:** https://chess-rating-predictor-d9ki9wewfxjdndrfkqec4k.streamlit.app
  (free hosting: if the app has been idle it may ask you to wake it up, which takes about 30 seconds)
- **Report:** [docs/report/](docs/report/) _(TODO: file name)_

## Results

Trained on 250,000 Lichess blitz games from September 2026 and evaluated once on 50,000 games the model never saw (100,000 player rows). MAE = mean absolute error in rating points.

| Model | Test MAE |
|---|---:|
| Always guess the average rating (baseline) | 291 |
| **Gradient boosting, tuned (used in the app)** | **219** |

The success criterion (test MAE ≤ 250 and better than the baseline) was set before the first test run and is met. The error is uneven: the model is accurate for players rated 1200–1999 and pulls uncertain guesses toward the middle, so beginners are overestimated and strong players underestimated. See the report for the full analysis.

## Quick start

Requires [uv](https://docs.astral.sh/uv/), which installs Python 3.13. On macOS and Linux the steps are short `make` commands; on Windows, see [Windows](#windows) below.

```bash
git clone <this repository>
cd <repository folder>
make setup   # Python 3.13 environment + libraries
make test    # all tests, a few seconds
make app     # the website on http://localhost:8501
```

The trained model is included in `models/`, so the app works right away without downloading any data.

> **Why uv and not Homebrew's Python?** On macOS 26.2, Homebrew's python@3.13 (3.13.16) cannot run pip because its XML module needs a newer system library. uv's Python 3.13 bundles its own copy and works.

## Reproduce every result

```bash
make all     # about 1 hour in total
```

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `make download` | Streams the Lichess September 2026 file (29 GB) over HTTP and keeps the first 250,000 rated blitz games that pass the filters. The full file is never downloaded. | ~20 min |
| 2 | `make dataset` | Removes rating headers, computes 34 features per player, builds two rows per game and splits train/test by game. | ~15 min |
| 3 | `make train` | Compares a baseline, Ridge, random forest and gradient boosting on a validation split, tunes gradient boosting, saves `models/model.joblib`. | ~10 min |
| 4 | `make evaluate` | Test-set results, error analysis, feature importance and the learning curve, saved to `docs/report/`. | ~3 min |

All settings (month, number of games, filters, seed, split shares) are in [config.yaml](config.yaml). A run overwrites the data, the model and the files in `docs/report/`, so do quick test runs with few games (e.g. `python -m src.download 2000`) in a separate copy of the project. To use newer data, change `month`, for example to `"2026-10"`.

## Windows

Windows has no `make`, so run the same steps directly in PowerShell (a standalone PowerShell window or the terminal in VS Code both work).

**1. Install uv** (once per computer):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

The installer adds uv to your PATH, but only terminals opened afterwards see it. Either close and reopen the terminal (in VS Code, quit VS Code completely and reopen it; a new terminal tab is not enough), or run this in the current terminal:

```powershell
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
```

Check with `uv --version`.

**2. Go to the project folder.** It must be the folder that contains `README.md` and `requirements.txt`. If you downloaded the ZIP from GitHub instead of cloning, Windows extracts it into a nested folder, so go one level deeper:

```powershell
cd chess-rating-predictor-main\chess-rating-predictor-main   # ZIP download
# or: cd <repository folder>                                  # git clone
ls requirements.txt                                            # should be found
```

**3. Set up, test and run** (all commands from that folder):

```powershell
uv python install 3.13
uv venv --python 3.13 --seed .venv
.venv\Scripts\python -m pip install -r requirements.txt

.venv\Scripts\python -m pytest              # tests
.venv\Scripts\streamlit run app/app.py      # the website on http://localhost:8501

# Reproduce every result (about 1 hour), in this order:
.venv\Scripts\python -m src.download
.venv\Scripts\python -m src.dataset
.venv\Scripts\python -m src.train
.venv\Scripts\python -m src.evaluate
```

## Deploy the app

1. Push this repository to GitHub (public).
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account and choose **Create app**.
3. Select the repository, branch `main`, and main file `app/app.py`.
4. Under **Advanced settings**, choose Python **3.13**, then **Deploy**.

Streamlit installs `requirements.txt` and loads the model from `models/`. Free apps sleep after a period without visitors and wake up on the next visit.

## How it works

```
Lichess (29 GB/month) ─stream─▶ download ─▶ dataset ─▶ train ─▶ models/model.joblib
                                filter        features    compare,        │
                                250k games    2 rows/game tune            ▼
                                                          evaluate    predict ◀── app
                                                                      strip ratings → features → model
```

- **No rating leakage.** Ratings, rating changes, titles, names and links are removed before anything else sees a game (`src/pgn_utils.py`). The feature code may only read an allow-list of safe headers, and tests prove the features are identical with real, removed or fake ratings.
- **One feature module** (`src/features.py`) is used both for training and in the app, so a pasted game is processed exactly like the training data.

| Folder | Contents |
|---|---|
| `src/` | download, rating stripping, features, dataset, training, evaluation, prediction |
| `tests/` | leakage, feature and prediction tests |
| `app/` | the Streamlit website (UI only) |
| `models/` | the trained model and its test results |
| `data/sample/` | 40 games for the tests; 12 unseen test-set games for the app's examples |
| `data/openings/` | the Lichess opening list (see `SOURCE.md`) |
| `docs/report/` | the report, figures and tables |

## Data sources

- Games: [Lichess open database](https://database.lichess.org/), released under CC0.
- Opening names: [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings), CC0 (version in `data/openings/SOURCE.md`).
