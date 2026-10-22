"""Read the project settings from config.yaml."""

from pathlib import Path

import yaml

CONFIG_FILE = Path(__file__).parent.parent / "config.yaml"


def load_config():
    """Return all settings as a dictionary."""
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def raw_games_file(config):
    """Where the downloaded games for the configured month are stored."""
    return Path("data/raw") / f"lichess_blitz_{config['data']['month']}.pgn"


def players_table_file(config):
    """Where the finished table for the configured month is stored."""
    return Path("data/processed") / f"players_{config['data']['month']}.parquet"
