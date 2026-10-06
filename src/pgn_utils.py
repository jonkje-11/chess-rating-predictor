"""PGN helpers shared by training and the app: leakage-safe header stripping, parsing and validation."""

from __future__ import annotations

import io
import re

import chess
import chess.pgn

# Headers that reveal (or identify a source of) the target. They are removed before any feature
# extraction in the app, and features.py never reads them in training either (AGENTS.md section 4).
LEAKY_HEADERS = frozenset({
    "WhiteElo", "BlackElo", "WhiteRatingDiff", "BlackRatingDiff", "WhiteTitle", "BlackTitle",
    "White", "Black", "WhiteFideId", "BlackFideId", "WhiteTeam", "BlackTeam", "Site", "LichessURL",
})

HEADER_LINE_RE = re.compile(r'^\s*\[(\w+)\s+"(.*)"\]\s*$')
# Lichess game URLs may carry a 4-character player suffix ("/AbCdEfGh1234") or "/black"; the game ID is the first 8.
LICHESS_URL_RE = re.compile(r"lichess\.org/(?:embed/)?(?:game/export/)?([A-Za-z0-9]{8})(?:[A-Za-z0-9]{4})?(?![A-Za-z0-9])")
LICHESS_BARE_ID_RE = re.compile(r"^([A-Za-z0-9]{8})(?:[A-Za-z0-9]{4})?$")


class PGNError(ValueError):
    """Raised with a user-facing message when a PGN cannot be used."""


def _to_int(value: str | None) -> int | None:
    return int(value) if value and value.strip().isdigit() else None


def strip_rating_headers(pgn: str) -> tuple[str, dict]:
    """Remove rating/identity headers from a PGN.

    Returns the cleaned PGN and a dict with the true ratings (``white_elo``, ``black_elo``, ints or None)
    plus all removed header values (``removed``), so the app can reveal them after predicting.
    Only the header block before the first movetext line is touched, so comments such as
    ``{ [%clk 0:03:00] }`` in wrapped movetext are never mistaken for headers.
    """
    kept: list[str] = []
    removed: dict[str, str] = {}
    in_headers = True
    for line in pgn.splitlines():
        m = HEADER_LINE_RE.match(line) if in_headers else None
        if m and m.group(1) in LEAKY_HEADERS:
            removed[m.group(1)] = m.group(2)
            continue
        if in_headers and line.strip() and not m:
            in_headers = False
        kept.append(line)
    truth = {
        "white_elo": _to_int(removed.get("WhiteElo")),
        "black_elo": _to_int(removed.get("BlackElo")),
        "removed": removed,
    }
    return "\n".join(kept).strip() + "\n", truth


def parse_game(pgn: str) -> chess.pgn.Game:
    """Parse the first game in a PGN string, raising PGNError with a readable message on failure."""
    if not pgn or not pgn.strip():
        raise PGNError("The PGN is empty.")
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        raise PGNError("Could not find a chess game in the input.")
    if game.errors:
        raise PGNError(f"The PGN contains an illegal or unreadable move: {game.errors[0]}")
    # python-chess happily parses free text as a game with zero moves.
    if game.next() is None:
        raise PGNError("No chess moves were found in the input.")
    return game


def validate_game(game: chess.pgn.Game, min_plies: int = 20) -> None:
    """Reject games the model was not trained for (variants, custom start positions, very short games)."""
    variant = game.headers.get("Variant", "Standard")
    if variant.lower() not in ("standard", "chess") or "FEN" in game.headers:
        raise PGNError(f"Only standard chess from the normal starting position is supported (got '{variant}').")
    plies = sum(1 for _ in game.mainline_moves())
    if plies < min_plies:
        raise PGNError(f"The game is too short ({plies} half-moves); at least {min_plies} are needed.")


def extract_lichess_game_id(text: str) -> str:
    """Return the 8-character game ID from a Lichess URL or a bare ID ('.../AbCdEfGh/black' -> 'AbCdEfGh')."""
    text = text.strip()
    m = LICHESS_URL_RE.search(text) or LICHESS_BARE_ID_RE.match(text)
    if not m:
        raise PGNError("That does not look like a Lichess game URL or game ID.")
    return m.group(1)
