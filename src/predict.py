"""Prediction backend used by the web app (and any future frontend/API).

`predict_pgn` returns plain JSON-serialisable data so the UI layer can be swapped (Gradio today,
FastAPI/Vercel later) without touching ML code. Rating headers are stripped HERE, in the backend,
before features are computed; the true values are only used afterwards for the comparison.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from src.config import ROOT, load_config
from src.features import extract_features, player_rows
from src.pgn_utils import PGNError, extract_lichess_game_id, parse_game, strip_rating_headers, validate_game

MODEL_PATH = ROOT / "models" / "model.joblib"
LICHESS_EXPORT_URL = "https://lichess.org/game/export/{game_id}?clocks=true&evals=true"

LIMITATIONS = (
    "One game is limited evidence, so estimates are rough. The model was trained on Lichess blitz games "
    "(3-5 minute time controls); ratings on other sites (e.g. FIDE or Chess.com) and in other time controls "
    "are on different scales."
)


@lru_cache(maxsize=1)
def load_bundle(path: Path = MODEL_PATH) -> dict:
    """Load the trained model bundle once per process."""
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def _round_to(x: float, step: int = 10) -> int:
    return int(step * round(x / step))


def predict_pgn(pgn: str) -> dict:
    """Estimate both players' ratings from a single PGN game.

    Raises PGNError with a user-facing message for invalid input.
    """
    start = time.perf_counter()
    cfg = load_config()
    bundle = load_bundle()

    clean_pgn, truth = strip_rating_headers(pgn)
    parsed = parse_game(clean_pgn)
    validate_game(parsed, cfg["filters"]["min_plies"])
    final = parsed.end()
    features = extract_features(clean_pgn)
    X = pd.DataFrame(player_rows(features))[bundle["feature_columns"]]
    for col in bundle["categorical_columns"]:
        X[col] = X[col].astype("category")
    raw = bundle["pipeline"].predict(X)
    lo, hi = cfg["app"]["min_rating"], cfg["app"]["max_rating"]
    estimates = np.clip(raw, lo, hi)
    margin = _round_to(bundle["interval"]["mae"], 5)

    players = {}
    for color, est in zip(("white", "black"), estimates):
        actual = truth[f"{color}_elo"]
        players[color] = {
            "estimate": _round_to(est),
            "low": _round_to(max(lo, est - margin)),
            "high": _round_to(min(hi, est + margin)),
            "actual": actual,
            "error": None if actual is None else _round_to(est) - actual,
        }

    game = features["game"]
    return {
        "white": players["white"],
        "black": players["black"],
        "margin": margin,
        "margin_coverage": round(bundle["interval"]["coverage_of_mae"], 2),
        "had_true_ratings": truth["white_elo"] is not None or truth["black_elo"] is not None,
        "game": {
            "plies": game["plies"],
            "eco": game["eco"],
            "termination": game["termination"],
            "time_control": None if np.isnan(game["base_time"]) else
            f"{int(game['base_time'] // 60)}+{0 if np.isnan(game['increment']) else int(game['increment'])}",
            "has_clocks": not np.isnan(features["white"]["avg_time_per_move"]),
            "result": parsed.headers.get("Result", "*"),
            "final_fen": final.board().fen(),
            "last_move": final.move.uci() if final.move else None,
        },
        "model": {
            "name": bundle["model_name"],
            "test_mae": round(bundle["test_metrics"]["MAE"], 1),
            "trained_on": bundle["trained_on"],
        },
        "limitations": LIMITATIONS,
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
    }


def fetch_lichess_pgn(url_or_id: str, timeout: float = 10.0) -> str:
    """Download a game's PGN (with clocks) from the public Lichess API. Raises PGNError on failure."""
    game_id = extract_lichess_game_id(url_or_id)
    try:
        resp = requests.get(LICHESS_EXPORT_URL.format(game_id=game_id), timeout=timeout,
                            headers={"Accept": "application/x-chess-pgn"})
    except requests.RequestException as exc:
        raise PGNError(f"Could not reach Lichess ({exc.__class__.__name__}). Try again or paste the PGN.") from exc
    if resp.status_code == 404:
        raise PGNError(f"Lichess has no game with ID '{game_id}'.")
    if resp.status_code == 429:
        raise PGNError("Lichess is rate-limiting requests right now. Wait a minute and try again, or paste the PGN.")
    if resp.status_code != 200 or not resp.text.strip():
        raise PGNError(f"Lichess returned an error (HTTP {resp.status_code}).")
    return resp.text
