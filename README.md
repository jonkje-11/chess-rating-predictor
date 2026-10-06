# Chess Rating Predictor

Estimate the playing strength (Lichess blitz rating) of both players **from the moves of a single game**, without looking at their real ratings.

> Final project in DAT158 Machine Learning, Western Norway University of Applied Sciences (HVL), autumn 2026.

- **Live demo:** _TODO: Hugging Face Spaces link_
- **Report:** [report/report.md](report/report.md)
- **AI usage log:** [report/ai_usage.md](report/ai_usage.md)

## How it works

1. A sample of rated blitz games is streamed from the [Lichess open database](https://database.lichess.org/) (CC0) and filtered.
2. Features are computed **only** from the moves, the result and move comments (clock times). Rating headers are never read (see "Data leakage" below).
3. Each game gives two rows (one per player); a regression model is trained to predict that player's rating.
4. A Gradio web app accepts a PGN, a `.pgn` file or a Lichess game link and returns the estimated ratings.

## Reproduce the results

Requires Python 3.13 and about 2 GB of free disk space. The Lichess monthly file (~29 GB) is **streamed**, never downloaded in full.

```bash
git clone https://github.com/jonkje-11/chess-rating-predictor.git
cd chess-rating-predictor
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All settings (month, number of games, filters, seed) live in [config.yaml](config.yaml).

| Step | Make (Linux/macOS/Codespaces) | Plain Python (any OS) |
|---|---|---|
| Stream + filter games | `make data` | `python -m src.download` |
| Build per-player dataset | `make features` | `python -m src.dataset` |
| Train and compare models | `make train` | `python -m src.train` |
| Figures and tables for the report | `make evaluate` | `python -m src.evaluate` |
| Run the web app locally | `make app` | `python app/app.py` |
| Run tests | `make test` | `python -m pytest -q` |

_TODO: expected runtimes once measured._

## Deploy to Hugging Face Spaces

_TODO (step 8)._

## Data leakage

PGN headers contain the answer (`WhiteElo`, `BlackElo`, rating diffs, titles, player names). These are never used as features. The app strips them in the backend before prediction and only uses them afterwards to show the true rating next to the estimate. A unit test checks that features are identical with and without these headers.

## Data and licences

- Game data: [Lichess open database](https://database.lichess.org/), released under CC0.
- Opening names/ECO codes: [lichess-org/chess-openings](https://github.com/lichess-org/chess-openings), CC0.

## Project structure

```
src/        download, PGN utilities, feature extraction, dataset, training, evaluation
app/        Gradio web app
tests/      unit tests (incl. leakage test)
notebooks/  exploratory data analysis
models/     final trained model
report/     report, figures, AI usage log
data/       sample/ is committed; raw/ and processed/ are generated
```
