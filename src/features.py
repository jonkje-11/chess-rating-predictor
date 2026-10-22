"""Feature extraction: turns one game into numbers. Used by both training and the app."""

import io
import math
import statistics
from functools import cache
from pathlib import Path


import chess
import chess.pgn

from src.pgn_utils import strip_rating_headers


# The only headers the feature code may read. Everything else (ratings, names,
# titles, links) is off limits, even if it is still in the text.
ALLOWED_HEADERS = {"Result", "TimeControl", "ECO", "Termination"}

# Lichess chess-openings list (CC0), one TSV file per ECO letter.
OPENINGS_DIR = Path(__file__).parent.parent / "data" / "openings"



def read_game(pgn):
    """Parse the PGN text of one game into a python-chess game object."""
    return chess.pgn.read_game(io.StringIO(pgn))


def header(game, name):
    """Read a header, but only if it is on the allowed list."""
    if name not in ALLOWED_HEADERS:
        raise ValueError(f"Feature code may not read the header {name!r}")
    return game.headers.get(name, "")


def clock_times(game):
    """Return the seconds left after each move, as two lists: (white, black)."""
    white = []
    black = []
    for node in game.mainline():
        seconds = node.clock()
        if seconds is None:
            continue
        # node.turn() is the side to move AFTER this move,
        # so if Black is to move, White just moved.
        if node.turn() == chess.BLACK:
            white.append(seconds)
        else:
            black.append(seconds)
    return white, black

def parse_time_control(game):
    """Return (base, increment) in seconds, e.g. "180+2" gives (180, 2)."""
    base, increment = header(game, "TimeControl").split("+")
    return int(base), int(increment)


def move_times(clocks, base, increment):
    """Seconds spent on each move, from the clock left before and after it."""
    times = []
    previous = base
    for current in clocks:
        spent = previous - current + increment
        times.append(max(spent, 0))
        previous = current
    return times


def time_features(clocks, base, increment):
    """Time-use features for one player. NaN when the game has no clock data."""
    if len(clocks) < 2:
        names = ["avg_move_time", "std_move_time", "opening_time", "first_long_think",
                 "longest_think", "instant_move_share", "low_time_share", "time_left_share"]
        return {name: math.nan for name in names}

    times = move_times(clocks, base, increment)

    # First move (1, 2, 3 ...) that took more than 10 seconds.
    # A player who never thinks that long gets "one more than their number of moves".
    first_long_think = len(times) + 1
    for move_number, spent in enumerate(times, start=1):
        if spent > 10:
            first_long_think = move_number
            break

    return {
        "avg_move_time": statistics.mean(times),
        "std_move_time": statistics.pstdev(times),
        "opening_time": sum(times[:10]),
        "first_long_think": first_long_think,
        "longest_think": max(times),
        "instant_move_share": sum(t <= 1 for t in times) / len(times),
        "low_time_share": sum(c < 10 for c in clocks) / len(clocks),
        "time_left_share": clocks[-1] / (base + 40 * increment),

    }

def game_features(game):
    """Features about how the game went, from each player's point of view: (white, black)."""
    final_board = game.end().board()
    plies = game.end().ply()
    result = header(game, "Result")
    on_time = header(game, "Termination") == "Time forfeit"
    checkmate = final_board.is_checkmate()

    # Score from White's point of view: 1 = win, 0.5 = draw, 0 = loss.
    white_score = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}.get(result, math.nan)

    features = {}
    for color, score in (("white", white_score), ("black", 1 - white_score)):
        features[color] = {
            "plies": plies,
            "score": score,
            "game_ended_in_mate": int(checkmate),
            "was_mated": int(checkmate and score == 0),
            "lost_on_time": int(on_time and score == 0),
        }
    return features["white"], features["black"]

# Starting squares of each side's knights and bishops, to measure development.
MINOR_PIECE_START = {
    chess.WHITE: [chess.B1, chess.G1, chess.C1, chess.F1],
    chess.BLACK: [chess.B8, chess.G8, chess.C8, chess.F8],
}


def opening_features(game):
    """Opening habits for each player: (white, black)."""
    board = game.board()
    stats = {
        color: {"castle_move": None, "castled_kingside": 0, "castled_queenside": 0,
                "early_queen_moves": 0, "knight_edge_moves": 0, "moves": 0}
        for color in (chess.WHITE, chess.BLACK)
    }
    developed = {}

    for move in game.mainline_moves():
        color = board.turn
        s = stats[color]
        s["moves"] += 1
        piece = board.piece_type_at(move.from_square)

        if board.is_castling(move) and s["castle_move"] is None:
            s["castle_move"] = s["moves"]
            if board.is_kingside_castling(move):
                s["castled_kingside"] = 1
            else:
                s["castled_queenside"] = 1
        if piece == chess.QUEEN and s["moves"] <= 10:
            s["early_queen_moves"] += 1
        if piece == chess.KNIGHT and chess.square_file(move.to_square) in (0, 7):
            s["knight_edge_moves"] += 1

        board.push(move)

        # After both players' 10th move, count knights and bishops that have left home.
        if board.ply() == 20:
            for side in (chess.WHITE, chess.BLACK):
                home = MINOR_PIECE_START[side]
                developed[side] = sum(
                    board.piece_at(square) is None or board.piece_at(square).color != side
                    for square in home
                )

    result = []
    for color in (chess.WHITE, chess.BLACK):
        s = stats[color]
        result.append({
            "castled": int(s["castle_move"] is not None),
            "castle_move": s["castle_move"] if s["castle_move"] is not None else s["moves"] + 1,
            "castled_kingside": s["castled_kingside"],
            "castled_queenside": s["castled_queenside"],
            "early_queen_moves": s["early_queen_moves"],
            "knight_edge_moves": s["knight_edge_moves"],
            "minor_pieces_developed": developed.get(color, math.nan),
        })
    return result[0], result[1]

@cache
def load_opening_book():
    """Read the openings list once. Returns {position: ECO code} for every book position."""
    book = {}
    for tsv in sorted(OPENINGS_DIR.glob("*.tsv")):
        lines = tsv.read_text(encoding="utf-8").splitlines()[1:]  # skip the header row
        for line in lines:
            eco, name, moves = line.split("\t")
            board = chess.Board()
            for token in moves.split():
                if token[0].isdigit():  # skip move numbers like "1."
                    continue
                board.push_san(token)
                book.setdefault(board.epd(), eco)
            book[board.epd()] = eco  # the named end position decides its own code
    if not book:
        raise FileNotFoundError(f"No opening files found in {OPENINGS_DIR}")
    return book


def book_features(game):
    """How long each player stayed in known opening theory, and the game's ECO code."""
    book = load_opening_book()
    board = game.board()
    in_book = {chess.WHITE: 0, chess.BLACK: 0}
    left_book_first = None
    derived_eco = "unknown"

    for move in game.mainline_moves():
        color = board.turn
        board.push(move)
        position = board.epd()
        if position not in book:
            left_book_first = color
            break
        in_book[color] += 1
        derived_eco = book[position]

    eco = header(game, "ECO") or derived_eco
    result = []
    for color in (chess.WHITE, chess.BLACK):
        result.append({
            "book_moves": in_book[color],
            "left_book_first": int(left_book_first == color),
            "eco": eco,
        })
    return result[0], result[1]

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(board, color):
    """Total piece value for one side (the king does not count)."""
    return sum(len(board.pieces(piece, color)) * value for piece, value in PIECE_VALUES.items())


def material_features(game):
    """Material balance over the game, from each player's point of view: (white, black)."""
    board = game.board()
    diffs = [0]  # White's material minus Black's, after every ply
    for move in game.mainline_moves():
        board.push(move)
        diffs.append(material(board, chess.WHITE) - material(board, chess.BLACK))

    white_score = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}.get(header(game, "Result"), math.nan)

    result = []
    for sign, score in ((1, white_score), (-1, 1 - white_score)):
        own = [sign * d for d in diffs]  # flip the sign for Black
        # A lead or deficit only counts if it lasts two more plies, so the
        # middle of an exchange (I take, you take back) is ignored.
        lasting_low = [max(own[i:i + 3]) for i in range(len(own))]
        lasting_high = [min(own[i:i + 3]) for i in range(len(own))]

        # Largest lasting deficit that the player later got back from (balance back to 0 or better).
        recovered = 0
        worst = 0
        for value in lasting_low:
            worst = min(worst, value)
            if value >= 0:
                recovered = max(recovered, -worst)
                worst = 0

        result.append({
            "material_move20": own[40] if len(own) > 40 else math.nan,
            "material_move30": own[60] if len(own) > 60 else math.nan,
            "material_move40": own[80] if len(own) > 80 else math.nan,
            "material_end": own[-1],
            "max_deficit_recovered": recovered,
            "failed_conversion": int(max(lasting_high) >= 3 and score < 1),
        })
    return result[0], result[1]

def valuable_pieces(board, color):
    """Squares of one side's knights, bishops, rooks and queens."""
    squares = chess.SquareSet()
    for piece in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        squares |= board.pieces(piece, color)
    return squares


def is_hanging(board, square, owner):
    """A piece is hanging if the other side attacks it and its own side does not defend it."""
    return board.is_attacked_by(not owner, square) and not board.is_attacked_by(owner, square)


def mistake_features(game):
    """Mistakes we can spot without an engine, counted per player: (white, black)."""
    board = game.board()
    counts = {
        color: {"hung_pieces": 0, "missed_free_captures": 0, "missed_mate_in_1": 0, "moves": 0}
        for color in (chess.WHITE, chess.BLACK)
    }

    for move in game.mainline_moves():
        color = board.turn
        c = counts[color]
        c["moves"] += 1

        # 1. Was there a mate in one that the player did not play?
        mate_moves = []
        for candidate in board.legal_moves:
            if board.gives_check(candidate):
                board.push(candidate)
                if board.is_checkmate():
                    mate_moves.append(candidate)
                board.pop()
        if mate_moves and move not in mate_moves:
            c["missed_mate_in_1"] += 1

        # 2. Did the opponent leave a piece hanging that the player did not take?
        #    (Not counted when the player is in check and may have no choice.)
        free = [sq for sq in valuable_pieces(board, not color) if is_hanging(board, sq, not color)]
        if free and not board.is_check() and move.to_square not in free:
            c["missed_free_captures"] += 1

        quiet_move = not board.is_capture(move)
        board.push(move)

        # 3. After a quiet (non-capturing) move: does the player leave a piece hanging?
        if quiet_move and any(is_hanging(board, sq, color) for sq in valuable_pieces(board, color)):
            c["hung_pieces"] += 1

    result = []
    for color in (chess.WHITE, chess.BLACK):
        c = counts[color]
        moves = max(c["moves"], 1)
        result.append({
            "hung_piece_rate": c["hung_pieces"] / moves,
            "missed_capture_rate": c["missed_free_captures"] / moves,
            "missed_mate_in_1": c["missed_mate_in_1"],
        })
    return result[0], result[1]

def extract_features(pgn):
    """All features for one game, as two dicts: (white, black).

    Leaky headers are removed first, so this is safe to call on any game,
    in training and in the app.
    """
    cleaned, removed = strip_rating_headers(pgn)
    game = read_game(cleaned)
    base, increment = parse_time_control(game)
    white_clocks, black_clocks = clock_times(game)

    parts = [
        (time_features(white_clocks, base, increment), time_features(black_clocks, base, increment)),
        game_features(game),
        opening_features(game),
        book_features(game),
        material_features(game),
        mistake_features(game),
    ]

    white = {"base_time": base, "increment": increment}
    black = {"base_time": base, "increment": increment}
    for white_part, black_part in parts:
        white.update(white_part)
        black.update(black_part)
    return white, black
