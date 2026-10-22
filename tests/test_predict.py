"""Tests for src/predict.py: the full path the app uses."""

import time
from pathlib import Path

import pytest

from src.pgn_utils import strip_rating_headers
from src.predict import InvalidGame, lichess_game_id, predict_pgn

SAMPLE_FILE = Path("data/sample/sample_games.pgn")


def first_sample_game():
    text = SAMPLE_FILE.read_text(encoding="utf-8")
    return "[Event " + text.split("[Event ")[1]


def test_prediction_does_not_depend_on_ratings():
    pgn = first_sample_game()
    cleaned, removed = strip_rating_headers(pgn)
    faked = pgn.replace(f'[WhiteElo "{removed["WhiteElo"]}"]', '[WhiteElo "3333"]')
    estimates = [predict_pgn(text)["white"]["estimate"] for text in (pgn, cleaned, faked)]
    assert estimates[0] == estimates[1] == estimates[2]


def test_true_ratings_are_revealed_after_prediction():
    result = predict_pgn(first_sample_game())
    assert result["white"]["true_rating"] == 1365
    assert result["white"]["error"] == result["white"]["estimate"] - 1365


def test_no_true_rating_when_game_has_none():
    cleaned, removed = strip_rating_headers(first_sample_game())
    assert predict_pgn(cleaned)["white"]["true_rating"] is None


@pytest.mark.parametrize("bad_input", [
    "this is not a chess game",
    '[Event "Rated Blitz game"]\n[TimeControl "180+0"]\n\n1. e4 e5 2. Nf3 Nc6 *',
    first_sample_game().replace('[TimeControl "300+0"]', '[TimeControl "900+10"]'),
    '[Variant "Chess960"]\n' + first_sample_game(),
])
def test_bad_input_gives_clear_error(bad_input):
    with pytest.raises(InvalidGame):
        predict_pgn(bad_input)


def test_lichess_links_and_ids():
    assert lichess_game_id("https://lichess.org/abcd1234") == "abcd1234"
    assert lichess_game_id("https://lichess.org/abcd1234/black") == "abcd1234"
    assert lichess_game_id("abcd1234WXYZ") == "abcd1234"
    with pytest.raises(InvalidGame):
        lichess_game_id("https://chess.com/game/123")


def test_prediction_is_fast_enough():
    predict_pgn(first_sample_game())  # first call loads the model
    start = time.time()
    predict_pgn(first_sample_game())
    assert time.time() - start < 1.0  # success condition: answer within 1 second
