"""Tests for src/features.py. Most important: features never depend on ratings or names."""

import math
from pathlib import Path

import pytest

from src.features import extract_features, header, read_game
from src.pgn_utils import strip_rating_headers

SAMPLE_FILE = Path("data/sample/sample_games.pgn")


def load_sample_games():
    """Split the sample file into a list with one PGN text per game."""
    text = SAMPLE_FILE.read_text(encoding="utf-8")
    return ["[Event " + part for part in text.split("[Event ") if part.strip()]


def same(a, b):
    """Compare two feature values. NaN counts as equal to NaN (normally NaN != NaN)."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def assert_same_features(first, second):
    assert first.keys() == second.keys()
    for name in first:
        assert same(first[name], second[name]), name


def fake_ratings_and_names(pgn):
    """Replace ratings, names and titles with absurd values, to see if features change."""
    lines = []
    for line in pgn.splitlines():
        if line.startswith(("[WhiteElo ", "[BlackElo ")):
            line = line.split(" ")[0] + ' "3333"]'
        elif line.startswith(("[White ", "[Black ")):
            line = line.split(" ")[0] + ' "somebody_else"]'
        lines.append(line)
    lines.insert(1, '[WhiteTitle "GM"]')
    return "\n".join(lines)


def test_features_identical_without_rating_headers():
    for pgn in load_sample_games():
        cleaned, removed = strip_rating_headers(pgn)
        for original, stripped in zip(extract_features(pgn), extract_features(cleaned)):
            assert_same_features(original, stripped)


def test_features_identical_with_fake_ratings_and_names():
    for pgn in load_sample_games():
        for original, faked in zip(extract_features(pgn), extract_features(fake_ratings_and_names(pgn))):
            assert_same_features(original, faked)


def test_header_allow_list_blocks_ratings():
    game = read_game(load_sample_games()[0])
    with pytest.raises(ValueError):
        header(game, "WhiteElo")


def test_both_players_get_the_same_feature_names():
    white, black = extract_features(load_sample_games()[0])
    assert white.keys() == black.keys()


def test_known_values_for_first_sample_game():
    # Checked by hand against the moves and clocks of the first sample game.
    white, black = extract_features(load_sample_games()[0])
    assert white["base_time"] == 300
    assert white["opening_time"] == 45
    assert white["castle_move"] == 13
    assert black["castle_move"] == 11
    assert white["score"] == 1.0
    assert black["lost_on_time"] == 1
    assert white["book_moves"] == 4
    assert white["eco"] == "A22"

def test_zero_base_time_game_does_not_crash():
    # 0+5: no starting time, 5 seconds per move. Counts as blitz (0 + 40 * 5 = 200 s).
    pgn = """[Event "Rated Blitz game"]
[Result "1-0"]
[TimeControl "0+5"]

1. e4 { [%clk 0:00:05] } 1... e5 { [%clk 0:00:05] } 2. Nf3 { [%clk 0:00:08] } 2... Nc6 { [%clk 0:00:09] } 1-0
"""
    white, black = extract_features(pgn)
    assert white["base_time"] == 0
    assert white["time_left_share"] >= 0
