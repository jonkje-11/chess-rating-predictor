"""Tests for src/pgn_utils.py, including that no rating/identity header survives stripping."""

from pathlib import Path

import pytest

from src.pgn_utils import (
    LEAKY_HEADERS,
    PGNError,
    extract_lichess_game_id,
    parse_game,
    strip_rating_headers,
    validate_game,
)

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_games.pgn"

# A 24-ply game with every leaky header, wrapped movetext and a comment that starts a line with "[".
FULL_PGN = """[Event "Rated Blitz game"]
[Site "https://lichess.org/AbCdEfGh"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "1650"]
[BlackElo "1480"]
[WhiteRatingDiff "+6"]
[BlackRatingDiff "-6"]
[WhiteTitle "FM"]
[TimeControl "180+2"]
[ECO "C50"]

1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4
6. cxd4 Bb4+ 7. Nc3 Nxe4 8. O-O Bxc3 9. d5 Bf6 10. Re1 Ne7 11. Rxe4 d6 12. Bg5 {
[%clk 0:02:10] } Bxg5 1-0
"""


def test_strip_removes_all_leaky_headers_and_returns_truth():
    clean, truth = strip_rating_headers(FULL_PGN)
    for tag in LEAKY_HEADERS:
        assert f"[{tag} " not in clean
    assert truth["white_elo"] == 1650
    assert truth["black_elo"] == 1480
    assert truth["removed"]["WhiteTitle"] == "FM"


def test_strip_keeps_non_leaky_headers_and_moves():
    clean, _ = strip_rating_headers(FULL_PGN)
    assert '[TimeControl "180+2"]' in clean
    assert '[ECO "C50"]' in clean
    # The comment line starting with "[%clk" is movetext and must survive.
    assert "[%clk 0:02:10]" in clean
    original_moves = list(parse_game(FULL_PGN).mainline_moves())
    assert list(parse_game(clean).mainline_moves()) == original_moves


def test_strip_without_ratings_returns_none():
    clean, truth = strip_rating_headers("1. e4 e5 2. Nf3 Nc6 *")
    assert truth["white_elo"] is None and truth["black_elo"] is None
    assert "1. e4" in clean


def test_strip_on_every_sample_game():
    games = SAMPLE.read_text(encoding="utf-8").strip().split("\n\n[Event ")
    assert len(games) >= 20
    for i, g in enumerate(games):
        pgn = g if i == 0 else "[Event " + g
        clean, truth = strip_rating_headers(pgn)
        assert truth["white_elo"] is not None
        assert not any(f"[{tag} " in clean for tag in LEAKY_HEADERS)


@pytest.mark.parametrize("bad", ["", "   ", "hello world", "1. e4 e5 2. Ke3 Qh4 3. Kxh4 *"])
def test_parse_rejects_invalid(bad):
    with pytest.raises(PGNError):
        parse_game(bad)


def test_validate_rejects_short_game_and_variants():
    with pytest.raises(PGNError, match="too short"):
        validate_game(parse_game("1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"))
    with pytest.raises(PGNError, match="standard chess"):
        validate_game(parse_game('[Variant "Atomic"]\n\n1. e4 e5 *'))
    validate_game(parse_game(FULL_PGN))  # 24 plies, standard: no error


@pytest.mark.parametrize("text", [
    "https://lichess.org/AbCdEfGh",
    "https://lichess.org/AbCdEfGh/black",
    "https://lichess.org/AbCdEfGh1234",
    "lichess.org/game/export/AbCdEfGh?clocks=true",
    "AbCdEfGh",
    "  AbCdEfGh1234 ",
])
def test_extract_lichess_game_id(text):
    assert extract_lichess_game_id(text) == "AbCdEfGh"


def test_extract_lichess_game_id_rejects_garbage():
    with pytest.raises(PGNError):
        extract_lichess_game_id("https://example.com/whatever")
