"""Tests for src/features.py. The leakage tests are the most important ones in the project."""

import math
import re
from pathlib import Path

import chess.pgn
import pytest

from src.features import eco_from_moves, extract_features, player_rows
from src.pgn_utils import strip_rating_headers

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_games.pgn"


def sample_games() -> list[str]:
    parts = SAMPLE.read_text(encoding="utf-8").strip().split("\n\n[Event ")
    return [p if i == 0 else "[Event " + p for i, p in enumerate(parts)]


def same(a, b) -> bool:
    """Equality that treats NaN == NaN."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def assert_rows_equal(rows_a: list[dict], rows_b: list[dict]) -> None:
    assert len(rows_a) == len(rows_b)
    for ra, rb in zip(rows_a, rows_b):
        assert ra.keys() == rb.keys()
        for k in ra:
            assert same(ra[k], rb[k]), f"feature {k} differs: {ra[k]} vs {rb[k]}"


@pytest.mark.parametrize("pgn", sample_games())
def test_features_identical_with_and_without_rating_headers(pgn):
    stripped, truth = strip_rating_headers(pgn)
    assert truth["white_elo"] is not None  # the sample really did contain ratings
    assert_rows_equal(player_rows(extract_features(pgn)), player_rows(extract_features(stripped)))


@pytest.mark.parametrize("pgn", sample_games()[:10])
def test_features_unchanged_when_ratings_and_names_are_altered(pgn):
    """Changing the answer in the headers must not change a single feature."""
    altered = re.sub(r'\[(WhiteElo|BlackElo) "\d+"\]', r'[\1 "3000"]', pgn)
    altered = re.sub(r'\[(White|Black) "[^"]*"\]', r'[\1 "someone_else"]', altered)
    altered = re.sub(r'\[(WhiteRatingDiff|BlackRatingDiff) "[^"]*"\]', r'[\1 "+99"]', altered)
    assert altered != pgn
    assert_rows_equal(player_rows(extract_features(pgn)), player_rows(extract_features(altered)))


def test_no_feature_name_mentions_rating_or_identity():
    rows = player_rows(extract_features(sample_games()[0]))
    for name in rows[0]:
        assert not re.search(r"elo|rating|title|name|site|url", name, re.I), name


SCHOLARS_MATE_PLUS = """[Result "1-0"]
[TimeControl "180+0"]

1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 2. Bc4 { [%clk 0:02:58] } 2... Nc6 { [%clk 0:02:55] }
3. Qh5 { [%clk 0:02:50] } 3... Nf6 { [%clk 0:02:40] } 4. Qxf7# { [%clk 0:02:45] } 1-0
"""


def test_checkmate_result_and_counts():
    f = extract_features(SCHOLARS_MATE_PLUS)
    w, b = f["white"], f["black"]
    assert f["game"]["plies"] == 7
    assert f["game"]["ended_in_checkmate"] == 1
    assert (w["delivered_mate"], w["was_mated"], b["was_mated"]) == (1, 0, 1)
    assert (w["result_score"], b["result_score"]) == (1.0, 0.0)
    assert w["captures"] == 1 and w["checks"] == 1
    assert w["queen_moves_opening"] == 2
    assert w["distinct_pieces_opening"] == 3   # e-pawn, bishop, queen
    assert w["material_end"] == 1.0            # won the f7 pawn
    assert math.isnan(w["material_m20"])       # game ended before move 20


def test_clock_features():
    f = extract_features(SCHOLARS_MATE_PLUS)
    w, b = f["white"], f["black"]
    assert (f["game"]["base_time"], f["game"]["increment"]) == (180.0, 0.0)
    # White clocks 180, 178, 170, 165 -> spent 0, 2, 8, 5
    assert w["avg_time_per_move"] == pytest.approx(15 / 4)
    assert w["time_first10"] == pytest.approx(15)
    assert w["end_time_frac"] == pytest.approx(165 / 180)
    assert b["time_first10"] == pytest.approx(0 + 5 + 15)


def test_missing_clocks_give_nan_not_errors():
    f = extract_features("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 *")
    assert math.isnan(f["white"]["avg_time_per_move"])
    assert math.isnan(f["white"]["result_score"])
    assert f["white"]["castled"] == 1 and f["white"]["castle_side"] == "kingside"
    assert f["white"]["castle_move"] == 5.0
    assert f["game"]["termination"] == "unknown"


def test_eco_derived_from_moves_when_header_missing():
    # Ruy Lopez, Morphy Defense (C70-C99 family) - derived from the opening book.
    f = extract_features("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 *")
    assert f["game"]["eco"].startswith("C")
    assert f["game"]["eco_group"] == "C"
    game = chess.pgn.read_game(__import__("io").StringIO("1. e4 *"))
    assert eco_from_moves(list(game.mainline_moves())) == "B00"


def test_player_rows_are_symmetric():
    rows = player_rows(extract_features(SCHOLARS_MATE_PLUS))
    white, black = rows
    assert (white["color"], black["color"]) == (1, 0)
    assert white["own_captures"] == black["opp_captures"]
    assert white["opp_was_mated"] == black["own_was_mated"] == 1
