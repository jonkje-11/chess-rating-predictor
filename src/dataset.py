"""Build the training table: two rows per game (one per player), features plus true rating."""

import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from src.config import load_config, players_table_file, raw_games_file

import pandas as pd

from src.features import extract_features
from src.pgn_utils import strip_rating_headers

CONFIG = load_config()
RAW_FILE = raw_games_file(CONFIG)
OUTPUT_FILE = players_table_file(CONFIG)
SEED = CONFIG["seed"]
TEST_SHARE = CONFIG["split"]["test_share"]


# Features that describe the whole game, so they are the same for both players.
# They are stored once per row, without an own_/opp_ prefix.
GAME_LEVEL = {"base_time", "increment", "plies", "game_ended_in_mate", "eco"}


def split_games(text):
    """Split the text of a PGN file into one string per game."""
    return ["[Event " + part for part in text.split("[Event ") if part.strip()]


def scramble(name):
    """Turn a username into a short code. The same name always gives the same code,
    but the name cannot be read back from it."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]

def build_rows(white, black):
    """Turn one game's two feature dictionaries into two model rows (one per player).

    Used both for the training table and in the app, so the columns always match.
    """
    rows = []
    for is_white, own, opponent in ((1, white, black), (0, black, white)):
        row = {"is_white": is_white}
        row.update({name: value for name, value in own.items() if name in GAME_LEVEL})
        row.update({"own_" + name: value for name, value in own.items() if name not in GAME_LEVEL})
        row.update({"opp_" + name: value for name, value in opponent.items() if name not in GAME_LEVEL})
        rows.append(row)
    return rows


def rows_for_game(game_number, pgn):
    """Two rows for one game: one for White and one for Black.
    Returns an empty list if the game cannot be processed."""
    try:
        cleaned, removed = strip_rating_headers(pgn)
        white, black = extract_features(cleaned)

        rows = []
        rows = build_rows(white, black)
        for row, color in zip(rows, ("White", "Black")):
            row["game_id"] = game_number
            row["player_id"] = scramble(removed.get(color, ""))
            row["rating"] = int(removed[color + "Elo"])
        return rows

    except Exception as error:
        print(f"Skipped game {game_number}: {error!r}")
        return []



def main():
    max_games = int(sys.argv[1]) if len(sys.argv) > 1 else None
    games = split_games(RAW_FILE.read_text(encoding="utf-8"))[:max_games]
    print("Games:", len(games))

    with ProcessPoolExecutor() as pool:
        results = pool.map(rows_for_game, range(len(games)), games, chunksize=200)
        rows = [row for pair in results for row in pair]

    print("Skipped games:", len(games) - len(rows) // 2)
    table = pd.DataFrame(rows)

    # Both rows of a game must land on the same side, so split by game, not by row.
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SHARE, random_state=SEED)
    train_index, test_index = next(splitter.split(table, groups=table["game_id"]))
    table["split"] = "train"
    table.loc[test_index, "split"] = "test"
    print("Train rows:", len(train_index), "Test rows:", len(test_index))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTPUT_FILE, index=False)
    print("Rows:", len(table), "Columns:", table.shape[1])
    print("Saved to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
