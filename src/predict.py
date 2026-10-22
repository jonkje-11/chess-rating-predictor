"""Predict both players' ratings for one game. Used by the app; contains no UI code."""

import io
import json
from functools import cache
from pathlib import Path
import re
import requests
import chess.pgn
import joblib
import pandas as pd

from src.dataset import build_rows
from src.features import extract_features
from src.pgn_utils import strip_rating_headers

MODELS_DIR = Path(__file__).parent.parent / "models"
MIN_PLIES = 20  # same minimum as the training data


class InvalidGame(ValueError):
    """The input can't be used. The message is shown to the user as it is."""


@cache
def load_model():
    """Load the model and its test results once, then reuse them."""
    saved = joblib.load(MODELS_DIR / "model.joblib")
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    return saved["model"], saved["features"], metrics


def check_game(pgn):
    """Raise InvalidGame with a clear message if the game can't be estimated."""
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None or game.errors:
        raise InvalidGame("This doesn't look like a valid PGN chess game.")
    if game.headers.get("Variant", "Standard") != "Standard":
        raise InvalidGame("Only standard chess is supported, not variants like Chess960.")
    plies = game.end().ply()
    if plies < MIN_PLIES:
        raise InvalidGame(f"The game is too short ({plies} half-moves). At least {MIN_PLIES} are needed.")
    time_control = game.headers.get("TimeControl", "-")
    if "+" not in time_control:
        raise InvalidGame("The game has no time control, so it can't be compared with blitz games.")
    base, increment = time_control.split("+")
    if not 180 <= int(base) + 40 * int(increment) <= 479:
        raise InvalidGame("The model is trained on blitz games (about 3 to 8 minutes). This game is not blitz.")
    return game

LICHESS_EXPORT = "https://lichess.org/game/export/{game_id}?clocks=true&evals=true"


def lichess_game_id(text):
    """Get the 8-character game ID from a Lichess link (or from a bare ID)."""
    text = text.strip()
    match = re.search(r"lichess\.org/([A-Za-z0-9]{8})", text)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9]{8}([A-Za-z0-9]{4})?", text):
        return text[:8]
    raise InvalidGame("That doesn't look like a Lichess game link, e.g. https://lichess.org/abcd1234")


def fetch_lichess_pgn(link_or_id):
    """Download one game as PGN text from the Lichess API (no login needed)."""
    game_id = lichess_game_id(link_or_id)
    try:
        response = requests.get(LICHESS_EXPORT.format(game_id=game_id), timeout=10)
    except requests.RequestException:
        raise InvalidGame("Could not reach Lichess. Check the internet connection and try again.")
    if response.status_code == 404:
        raise InvalidGame(f"Lichess has no game with the ID {game_id}.")
    if response.status_code == 429:
        raise InvalidGame("Lichess is limiting requests right now. Wait a minute and try again.")
    if response.status_code != 200:
        raise InvalidGame(f"Lichess answered with an error ({response.status_code}). Try again later.")
    return response.text


def predict_pgn(pgn):
    """Estimate both players' ratings from the text of one game.

    Returns a plain dictionary, so the app (or any other frontend) can display it directly.
    """
    cleaned, removed = strip_rating_headers(pgn)
    game = check_game(cleaned)
    model, features, metrics = load_model()

    white, black = extract_features(cleaned)
    rows = pd.DataFrame(build_rows(white, black))
    estimates = model.predict(rows[features])

    result = {"margin": round(metrics["mae"]), "final_fen": game.end().board().fen()}
    for color, estimate in zip(("White", "Black"), estimates):
        true = removed.get(color + "Elo", "")
        entry = {"estimate": round(float(estimate)), "true_rating": int(true) if true.isdigit() else None}
        if entry["true_rating"] is not None:
            entry["error"] = entry["estimate"] - entry["true_rating"]
        result[color.lower()] = entry
    return result
