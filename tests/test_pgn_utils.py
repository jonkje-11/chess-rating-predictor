"""Tests for src/pgn_utils.py: prove that no leaky header reaches the model."""

from pathlib import Path

from src.pgn_utils import LEAKY_HEADERS, strip_rating_headers

SAMPLE_FILE = Path("data/sample/sample_games.pgn")

EXAMPLE_GAME = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abcd1234"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "1650"]
[BlackElo "1480"]
[WhiteRatingDiff "+6"]
[BlackRatingDiff "-6"]
[WhiteTitle "FM"]
[TimeControl "180+2"]

1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 1-0
"""


def load_sample_games():
    """Split the sample file into a list with one PGN text per game."""
    text = SAMPLE_FILE.read_text(encoding="utf-8")
    return ["[Event " + part for part in text.split("[Event ") if part.strip()]


def test_removed_values_are_returned():
    cleaned, removed = strip_rating_headers(EXAMPLE_GAME)
    assert removed["WhiteElo"] == "1650"
    assert removed["BlackElo"] == "1480"
    assert removed["White"] == "alice"


def test_leaky_headers_are_gone():
    cleaned, removed = strip_rating_headers(EXAMPLE_GAME)
    for name in LEAKY_HEADERS:
        assert f"[{name} " not in cleaned
    assert "1650" not in cleaned
    assert "alice" not in cleaned


def test_safe_parts_are_kept():
    cleaned, removed = strip_rating_headers(EXAMPLE_GAME)
    assert '[Result "1-0"]' in cleaned
    assert '[TimeControl "180+2"]' in cleaned
    assert "1. e4 { [%clk 0:03:00] }" in cleaned


def test_no_leaky_header_survives_in_sample_games():
    games = load_sample_games()
    assert len(games) == 40
    for game in games:
        cleaned, removed = strip_rating_headers(game)
        for name in LEAKY_HEADERS:
            assert f"[{name} " not in cleaned
        assert removed["WhiteElo"].isdigit()
        assert removed["BlackElo"].isdigit()
