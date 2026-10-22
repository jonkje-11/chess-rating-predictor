import io
import json
import sys
from pathlib import Path
from src.config import load_config, raw_games_file

import chess.pgn
import requests
import zstandard

CONFIG = load_config()
MONTH = CONFIG["data"]["month"]
FILTERS = CONFIG["filters"]

URL = CONFIG["data"]["url_template"].format(month=MONTH)
OUTPUT_FILE = raw_games_file(CONFIG)
COUNTS_FILE = Path(f"docs/report/tables/filter_counts_{MONTH}.json")


def estimated_duration(time_control):
    """Lichess formula: base time + 40 x increment, in seconds."""
    base, increment = time_control.split("+")
    return int(base) + 40 * int(increment)


def skip_reason(game):
    """Return why a game is filtered out, or None if we keep it."""
    headers = game.headers
    time_control = headers.get("TimeControl", "-")
    if time_control == "-" or not FILTERS["blitz_min_seconds"] <= estimated_duration(time_control) <= FILTERS["blitz_max_seconds"]:
        return "not blitz"
    if not headers.get("WhiteElo", "").isdigit() or not headers.get("BlackElo", "").isdigit():
        return "missing rating"
    if headers.get("Termination") in FILTERS["excluded_terminations"]:
        return "abandoned or rules infraction"
    if game.end().ply() < FILTERS["min_plies"]:
        return "too short"
    return None


response = requests.get(URL, stream=True)
decompressor = zstandard.ZstdDecompressor()
binary_stream = decompressor.stream_reader(response.raw)
text_stream = io.TextIOWrapper(binary_stream, encoding="utf-8")

MAX_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else CONFIG["data"]["n_games"]  # number of games to KEEP


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
COUNTS_FILE.parent.mkdir(parents=True, exist_ok=True)

games_read = 0
games_kept = 0
skipped = {}

with open(OUTPUT_FILE, "w", encoding="utf-8") as output:
    while games_kept < MAX_GAMES:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            break
        games_read += 1
        reason = skip_reason(game)
        if reason is None:
            games_kept += 1
            output.write(str(game) + "\n\n")
        else:
            skipped[reason] = skipped.get(reason, 0) + 1

counts = {"games_read": games_read, "games_kept": games_kept, "skipped": skipped}
with open(COUNTS_FILE, "w") as f:
    json.dump(counts, f, indent=2)

print(counts)
print("Saved games to", OUTPUT_FILE)
