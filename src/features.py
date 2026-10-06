"""Feature extraction shared by training (src/dataset.py) and the app (src/predict.py).

Features come ONLY from the moves, the result, move comments (clocks) and a small allow-list of
non-identifying headers. Rating headers, names and URLs are never read: every header access goes
through `_header`, which refuses anything outside `ALLOWED_HEADERS` (AGENTS.md section 4).

Feature set A (always available). Engine-based set B is added in a later phase.
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from statistics import mean, pstdev

import chess
import chess.pgn

from src.pgn_utils import parse_game

OPENINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "openings"

# The only headers feature code may read. None of them carry rating or player identity.
ALLOWED_HEADERS = frozenset({"Result", "TimeControl", "ECO", "Termination"})

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
MATERIAL_CHECKPOINTS = (20, 30, 40)  # full moves
OPENING_MOVES = 10                    # per player, for opening-phase features
LOW_TIME_SECONDS = 10.0

COLORS = {"white": chess.WHITE, "black": chess.BLACK}


def _header(game: chess.pgn.Game, name: str) -> str | None:
    """Read a header, but only from the leakage-safe allow-list."""
    if name not in ALLOWED_HEADERS:
        raise KeyError(f"Header '{name}' is not allowed in feature extraction (possible leakage).")
    value = game.headers.get(name)
    return value if value not in (None, "", "?", "-") else None


def _material(board: chess.Board, color: chess.Color) -> int:
    return sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values() if p.color == color)


def parse_time_control(tc: str | None) -> tuple[float, float]:
    """'180+2' -> (180.0, 2.0); unknown -> (nan, nan)."""
    if tc and "+" in tc:
        base, inc = tc.split("+", 1)
        if base.isdigit() and inc.isdigit():
            return float(base), float(inc)
    return math.nan, math.nan


@lru_cache(maxsize=1)
def _opening_book() -> dict[str, str]:
    """Map position EPD -> ECO code from the Lichess chess-openings dataset (CC0)."""
    book: dict[str, str] = {}
    for path in sorted(OPENINGS_DIR.glob("*.tsv")):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                board = chess.Board()
                for token in row["pgn"].split():
                    if not token[0].isdigit():
                        board.push_san(token)
                book[board.epd()] = row["eco"]
    return book


def eco_from_moves(moves: list[chess.Move], max_plies: int = 40) -> str | None:
    """Return the ECO code of the deepest known opening position reached in the game."""
    book = _opening_book()
    board = chess.Board()
    eco = None
    for move in moves[:max_plies]:
        board.push(move)
        eco = book.get(board.epd(), eco)
    return eco


def _result_score(result: str | None, color: chess.Color) -> float:
    """Score from this player's perspective: 1 win, 0.5 draw, 0 loss, NaN unknown."""
    white_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result or "", math.nan)
    return white_score if color == chess.WHITE else 1.0 - white_score


def _recovered_deficit(diffs: list[int]) -> float:
    """Largest material deficit the player later climbed back from to level material or better."""
    best, running_min = 0, 0
    for d in diffs:
        running_min = min(running_min, d)
        if d >= 0:
            best = max(best, -running_min)
    return float(best)


def _clock_features(clocks: list[float], base: float, inc: float) -> dict[str, float]:
    """Time-use statistics from the clock remaining after each of this player's moves."""
    nan = math.nan
    keys = ("avg_time_per_move", "std_time_per_move", "time_first10", "end_time_frac", "low_time_frac")
    if len(clocks) < 2:
        return dict.fromkeys(keys, nan)
    start = base if not math.isnan(base) else clocks[0]
    increment = 0.0 if math.isnan(inc) else inc
    # Time spent on a move = clock before - clock after + increment received for that move.
    spent = [max(0.0, prev - cur + increment) for prev, cur in zip([start] + clocks[:-1], clocks)]
    return {
        "avg_time_per_move": mean(spent),
        "std_time_per_move": pstdev(spent),
        "time_first10": sum(spent[:OPENING_MOVES]),
        "end_time_frac": clocks[-1] / start if start > 0 else nan,
        "low_time_frac": sum(c < LOW_TIME_SECONDS for c in clocks) / len(clocks),
    }


def extract_features(pgn: str) -> dict:
    """Compute feature set A for one game.

    Returns ``{"game": {...}, "white": {...}, "black": {...}}`` where the player dicts contain that
    player's own statistics. Use `player_rows` to turn this into model input rows.
    """
    game = parse_game(pgn)
    board = game.board()
    moves = list(game.mainline_moves())
    nodes = list(game.mainline())

    stats = {
        c: {"captures": 0, "checks": 0, "promotions": 0, "castle_move": math.nan, "castle_side": "none",
            "queen_moves_opening": 0, "clocks": []}
        for c in (chess.WHITE, chess.BLACK)
    }
    # Track each piece by its starting square, so "distinct pieces moved" follows pieces across moves.
    origin = {sq: sq for sq in board.piece_map()}
    opening_pieces: dict[chess.Color, set[int]] = {chess.WHITE: set(), chess.BLACK: set()}
    diffs: list[int] = []  # white material minus black material after each ply
    checkpoints: dict[int, int] = {}

    for ply, (move, node) in enumerate(zip(moves, nodes), start=1):
        color = board.turn
        s = stats[color]
        move_no = board.fullmove_number
        piece = board.piece_at(move.from_square)

        if board.is_capture(move):
            s["captures"] += 1
        if move.promotion:
            s["promotions"] += 1
        if board.is_castling(move):
            s["castle_move"] = float(move_no)
            s["castle_side"] = "kingside" if board.is_kingside_castling(move) else "queenside"
        if move_no <= OPENING_MOVES:
            opening_pieces[color].add(origin.get(move.from_square, move.from_square))
            if piece and piece.piece_type == chess.QUEEN:
                s["queen_moves_opening"] += 1

        # Update piece identity map (castling also moves the rook).
        start_id = origin.pop(move.from_square, move.from_square)
        origin.pop(move.to_square, None)
        if board.is_castling(move):
            rook_from, rook_to = _castling_rook_squares(board, move)
            origin[rook_to] = origin.pop(rook_from, rook_from)
        if board.is_en_passant(move):
            origin.pop(move.to_square + (-8 if color == chess.WHITE else 8), None)
        origin[move.to_square] = start_id

        board.push(move)
        if board.is_check():
            s["checks"] += 1
        clock = node.clock()
        if clock is not None:
            s["clocks"].append(float(clock))

        diffs.append(_material(board, chess.WHITE) - _material(board, chess.BLACK))
        if ply % 2 == 0 and ply // 2 in MATERIAL_CHECKPOINTS:
            checkpoints[ply // 2] = diffs[-1]

    result = _header(game, "Result")
    is_mate = board.is_checkmate()
    base, inc = parse_time_control(_header(game, "TimeControl"))
    white_clocks = stats[chess.WHITE]["clocks"]
    if math.isnan(base) and white_clocks:
        base = float(round(white_clocks[0]))  # first clock ~ base time when TimeControl is missing
    termination = (_header(game, "Termination") or "unknown").lower()
    eco = _header(game, "ECO") or eco_from_moves(moves) or "unknown"

    out: dict = {
        "game": {
            "plies": len(moves),
            "base_time": base,
            "increment": inc,
            "ended_in_checkmate": int(is_mate),
            "termination": termination,
            "eco": eco,
            "eco_group": eco[0] if eco != "unknown" else "unknown",
        }
    }
    for name, color in COLORS.items():
        s = stats[color]
        sign = 1 if color == chess.WHITE else -1
        n_moves = len(moves) // 2 + (len(moves) % 2 if color == chess.WHITE else 0)
        own_diffs = [sign * d for d in diffs]
        player = {
            "n_moves": n_moves,
            "result_score": _result_score(result, color),
            "delivered_mate": int(is_mate and board.turn != color),
            "was_mated": int(is_mate and board.turn == color),
            "captures": s["captures"],
            "capture_rate": s["captures"] / n_moves if n_moves else math.nan,
            "checks": s["checks"],
            "check_rate": s["checks"] / n_moves if n_moves else math.nan,
            "promotions": s["promotions"],
            "castled": int(s["castle_side"] != "none"),
            "castle_side": s["castle_side"],
            "castle_move": s["castle_move"],
            "queen_moves_opening": s["queen_moves_opening"],
            "distinct_pieces_opening": len(opening_pieces[color]),
            "material_end": float(own_diffs[-1]) if own_diffs else math.nan,
            "max_material_deficit": float(max(0, -min(own_diffs))) if own_diffs else math.nan,
            "recovered_deficit": _recovered_deficit(own_diffs),
        }
        for m in MATERIAL_CHECKPOINTS:
            player[f"material_m{m}"] = float(sign * checkpoints[m]) if m in checkpoints else math.nan
        player.update(_clock_features(s["clocks"], base, inc))
        out[name] = player
    return out


def _castling_rook_squares(board: chess.Board, move: chess.Move) -> tuple[int, int]:
    """Return (from, to) of the rook for a standard castling move."""
    rank = chess.square_rank(move.from_square)
    if board.is_kingside_castling(move):
        return chess.square(7, rank), chess.square(5, rank)
    return chess.square(0, rank), chess.square(3, rank)


def player_rows(features: dict) -> list[dict]:
    """Turn `extract_features` output into two model rows (white first, then black).

    Each row has the game-level features, the player's own features (``own_*``), the opponent's
    features (``opp_*``) and a ``color`` flag (1 = white). The same function is used in training and the app.
    """
    rows = []
    for name, opp in (("white", "black"), ("black", "white")):
        row = {"color": int(name == "white"), **features["game"]}
        row.update({f"own_{k}": v for k, v in features[name].items()})
        row.update({f"opp_{k}": v for k, v in features[opp].items()})
        rows.append(row)
    return rows


CATEGORICAL_FEATURES = ["termination", "eco", "eco_group", "own_castle_side", "opp_castle_side"]


# ---------------------------------------------------------------------------------------------------
# Feature set B: engine evaluations from %eval comments (only ~9% of Lichess games have them).
# Not used by the deployed model; compared against set A in src/evaluate.py.
# ---------------------------------------------------------------------------------------------------

EVAL_CAP = 1000          # centipawns; mate scores and huge advantages are capped so one move cannot dominate
EVAL_EARLY_MOVES = 15    # "opening accuracy" = this player's first 15 moves
DEFAULT_THRESHOLDS = {"inaccuracy": 50, "mistake": 100, "blunder": 300}


def extract_eval_features(pgn: str, thresholds: dict[str, int] | None = None,
                          min_coverage: float = 0.9) -> dict | None:
    """Centipawn-loss statistics per player, or None if too few moves carry an engine evaluation.

    Loss of a move = (eval before the move - eval after it) from the mover's point of view, floored at 0.
    Moves are classified by loss: >= blunder, >= mistake, >= inaccuracy (mutually exclusive).
    """
    th = thresholds or DEFAULT_THRESHOLDS
    game = parse_game(pgn)
    nodes = list(game.mainline())
    evals = []
    for node in nodes:
        score = node.eval()
        evals.append(None if score is None else
                     max(-EVAL_CAP, min(EVAL_CAP, score.white().score(mate_score=EVAL_CAP))))
    if not nodes or sum(e is not None for e in evals) / len(nodes) < min_coverage:
        return None

    losses: dict[chess.Color, list[float]] = {chess.WHITE: [], chess.BLACK: []}
    before = 20  # typical engine eval of the starting position
    for ply, after in enumerate(evals):
        color = chess.WHITE if ply % 2 == 0 else chess.BLACK
        if before is not None and after is not None:
            sign = 1 if color == chess.WHITE else -1
            losses[color].append(max(0.0, sign * (before - after)))
        else:
            losses[color].append(math.nan)
        before = after

    out = {}
    for name, color in COLORS.items():
        lst = losses[color]
        valid = [x for x in lst if not math.isnan(x)]
        early = [x for x in lst[:EVAL_EARLY_MOVES] if not math.isnan(x)]
        late = [x for x in lst[EVAL_EARLY_MOVES:] if not math.isnan(x)]
        n = len(valid) or 1
        blunders = sum(x >= th["blunder"] for x in valid)
        mistakes = sum(th["mistake"] <= x < th["blunder"] for x in valid)
        inaccuracies = sum(th["inaccuracy"] <= x < th["mistake"] for x in valid)
        out[name] = {
            "eval_acpl": mean(valid) if valid else math.nan,
            "eval_acpl_first15": mean(early) if early else math.nan,
            "eval_acpl_after15": mean(late) if late else math.nan,
            "eval_inaccuracies": inaccuracies,
            "eval_mistakes": mistakes,
            "eval_blunders": blunders,
            "eval_blunder_rate": blunders / n,
            "eval_mistake_rate": mistakes / n,
        }
    return out


def eval_player_rows(eval_features: dict) -> list[dict]:
    """Same own_/opp_ layout as `player_rows`, for set B features (white row first)."""
    rows = []
    for name, opp in (("white", "black"), ("black", "white")):
        row = {f"own_{k}": v for k, v in eval_features[name].items()}
        row.update({f"opp_{k}": v for k, v in eval_features[opp].items()})
        rows.append(row)
    return rows
