"""End-to-end tests for the prediction backend used by the app (requires models/model.joblib)."""

import json
import re
from pathlib import Path

import pytest
import requests

import src.predict as predict
from src.pgn_utils import PGNError

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "model.joblib").exists(), reason="no trained model")


def first_sample_game() -> str:
    text = (ROOT / "data" / "sample" / "sample_games.pgn").read_text(encoding="utf-8")
    return text.strip().split("\n\n[Event ")[0]


def test_predict_returns_json_serialisable_estimates_and_truth():
    pgn = first_sample_game()
    res = predict.predict_pgn(pgn)
    json.dumps(res)  # the API contract: plain data, so the frontend can be swapped
    white_elo = int(re.search(r'\[WhiteElo "(\d+)"\]', pgn).group(1))
    assert res["white"]["actual"] == white_elo
    assert res["white"]["error"] == res["white"]["estimate"] - white_elo
    assert 400 <= res["black"]["estimate"] <= 3100
    assert res["white"]["low"] < res["white"]["estimate"] < res["white"]["high"]
    assert res["had_true_ratings"] is True


def test_prediction_does_not_depend_on_rating_headers():
    """End-to-end leakage check: changing the real ratings must not change the estimate."""
    pgn = first_sample_game()
    altered = re.sub(r'\[(WhiteElo|BlackElo) "\d+"\]', r'[\1 "2900"]', pgn)
    a, b = predict.predict_pgn(pgn), predict.predict_pgn(altered)
    assert a["white"]["estimate"] == b["white"]["estimate"]
    assert a["black"]["estimate"] == b["black"]["estimate"]
    assert b["white"]["actual"] == 2900  # ...but the reveal uses the header values


def test_prediction_without_headers_has_no_truth():
    pgn = first_sample_game()
    moves_only = pgn.split("\n\n", 1)[1]
    res = predict.predict_pgn(moves_only)
    assert res["white"]["actual"] is None and res["had_true_ratings"] is False


@pytest.mark.parametrize("bad", ["", "not a game", "1. e4 e5 2. Nf3 Nc6 *", '[Variant "Chess960"]\n\n1. e4 e5 *'])
def test_invalid_input_raises_user_facing_error(bad):
    with pytest.raises(PGNError):
        predict.predict_pgn(bad)


class FakeResponse:
    def __init__(self, status: int, text: str = ""):
        self.status_code, self.text = status, text


@pytest.mark.parametrize("status, message", [(404, "no game"), (429, "rate-limiting"), (500, "HTTP 500")])
def test_lichess_fetch_errors_are_friendly(monkeypatch, status, message):
    monkeypatch.setattr(predict.requests, "get", lambda *a, **k: FakeResponse(status))
    with pytest.raises(PGNError, match=message):
        predict.fetch_lichess_pgn("https://lichess.org/AbCdEfGh")


def test_lichess_fetch_network_error(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(predict.requests, "get", boom)
    with pytest.raises(PGNError, match="Could not reach Lichess"):
        predict.fetch_lichess_pgn("AbCdEfGh")
