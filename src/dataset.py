"""Build the per-player modelling table from the raw games and assign a train/test split.

Each game gives two rows (white, black). Target: that player's rating. The split is grouped by game ID
so both rows of a game land on the same side (otherwise the opponent features of a test row would
have been seen as own features in training).

Usage:
    python -m src.dataset               # uses data.n_games from config.yaml
    python -m src.dataset --n 20000
"""

from __future__ import annotations

import argparse
import logging
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import ROOT, load_config
from src.features import CATEGORICAL_FEATURES, extract_features, player_rows

log = logging.getLogger(__name__)

# Columns that are NOT features: target, split bookkeeping, and the username, which is kept ONLY
# for an optional grouped-by-player evaluation and must never be fed to a model.
TARGET = "rating"
META_COLUMNS = ["game_id", "player", "split", TARGET]


def raw_path(cfg: dict, n: int) -> Path:
    return ROOT / cfg["data"]["raw_dir"] / f"games_{cfg['data']['month']}_n{n}.parquet"


def processed_path(cfg: dict, n: int) -> Path:
    return ROOT / cfg["data"]["processed_dir"] / f"features_{cfg['data']['month']}_n{n}.parquet"


def _rows_for_game(args: tuple[str, str, str, str, str, str]) -> list[dict] | None:
    """Worker: features for one game plus labels. Labels are attached here, outside features.py."""
    game_id, pgn, white, black, white_elo, black_elo = args
    try:
        rows = player_rows(extract_features(pgn))
    except Exception:  # noqa: BLE001 - a malformed game is counted and skipped, not fatal
        return None
    for row, player, elo in zip(rows, (white, black), (white_elo, black_elo)):
        row.update({"game_id": game_id, "player": player, TARGET: int(elo)})
    return rows


def build_table(games: pd.DataFrame, workers: int | None = None) -> tuple[pd.DataFrame, int]:
    """Extract features for all games in parallel. Returns (table, number of failed games)."""
    jobs = zip(games.game_id, games.pgn, games.White, games.Black, games.WhiteElo, games.BlackElo)
    rows: list[dict] = []
    failed = 0
    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap(_rows_for_game, jobs, chunksize=500), start=1):
            if result is None:
                failed += 1
            else:
                rows.extend(result)
            if i % 50_000 == 0:
                log.info("processed %d / %d games", i, len(games))
    table = pd.DataFrame(rows)
    for col in CATEGORICAL_FEATURES:
        table[col] = table[col].astype("category")
    return table, failed


def assign_split(table: pd.DataFrame, test_size: float, seed: int) -> pd.Series:
    """'train' / 'test' per row, grouped by game so a game's two rows never straddle the split."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, test_idx = next(splitter.split(table, groups=table["game_id"]))
    split = np.full(len(table), "train", dtype=object)
    split[test_idx] = "test"
    return pd.Series(split, index=table.index, dtype="category")


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in META_COLUMNS]


def load_table(cfg: dict | None = None, n: int | None = None) -> pd.DataFrame:
    """Load the processed table written by `main`."""
    cfg = cfg or load_config()
    return pd.read_parquet(processed_path(cfg, n or cfg["data"]["n_games"]))


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=cfg["data"]["n_games"], help="which raw file (games kept)")
    p.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    games = pd.read_parquet(raw_path(cfg, args.n))
    log.info("Extracting features for %d games", len(games))
    start = time.time()
    table, failed = build_table(games, args.workers)
    table["split"] = assign_split(table, cfg["split"]["test_size"], cfg["seed"])
    log.info("Done in %.0f s: %d rows, %d feature columns, %d games failed to parse",
             time.time() - start, len(table), len(feature_columns(table)), failed)
    log.info("Rows per split: %s", table["split"].value_counts().to_dict())
    log.info("Rating: mean %.0f, std %.0f, min %d, max %d",
             table[TARGET].mean(), table[TARGET].std(), table[TARGET].min(), table[TARGET].max())

    out = processed_path(cfg, args.n)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    log.info("Saved %s", out)


if __name__ == "__main__":
    main()
