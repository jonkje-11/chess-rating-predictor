"""Stream rated blitz games from the Lichess open database and keep the first N that pass the filters.

The monthly file is ~29 GB compressed, so it is never downloaded: the HTTP response is decompressed
on the fly and the connection is closed as soon as N games have been kept.

Games are split and filtered on header text only (no full move parsing), which keeps streaming fast.
Move-level features are computed later by src/features.py.

Usage:
    python -m src.download                 # values from config.yaml
    python -m src.download --n 20000       # smaller dev run
    python -m src.download --sample 40     # also write the first 40 kept games to the sample PGN
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
import zstandard

from src.config import ROOT, load_config

log = logging.getLogger(__name__)

HEADER_RE = re.compile(r'^\[(\w+) "(.*)"\]\s*$')
COMMENT_RE = re.compile(r"\{[^}]*\}")
MOVE_NUMBER_RE = re.compile(r"\d+\.(\.\.)?")
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}

# Columns kept in the raw Parquet. Rating headers and names are stored ONLY as labels / grouping keys;
# features.py never reads them (see AGENTS.md section 4).
HEADER_COLUMNS = [
    "Event", "UTCDate", "UTCTime", "Result", "TimeControl", "Termination", "ECO", "Opening",
    "WhiteElo", "BlackElo", "White", "Black",
]

SCHEMA = pa.schema(
    [("game_id", pa.string())]
    + [(h, pa.string()) for h in HEADER_COLUMNS]
    + [("plies", pa.int32()), ("pgn", pa.string())]
)


def iter_pgn_texts(lines: Iterator[str]) -> Iterator[str]:
    """Split a stream of PGN lines into one string per game.

    A new game starts at an `[Event ` header that follows movetext; this is how Lichess dumps are laid out.
    """
    buf: list[str] = []
    seen_moves = False
    for line in lines:
        if line.startswith("[Event ") and seen_moves:
            yield "".join(buf)
            buf, seen_moves = [], False
        buf.append(line)
        if line.strip() and not line.startswith("["):
            seen_moves = True
    if buf and seen_moves:
        yield "".join(buf)


def parse_headers(pgn: str) -> tuple[dict[str, str], str]:
    """Return (headers, movetext) for a single game's PGN text."""
    headers: dict[str, str] = {}
    moves: list[str] = []
    for line in pgn.splitlines():
        m = HEADER_RE.match(line)
        if m:
            headers[m.group(1)] = m.group(2)
        elif line.strip():
            moves.append(line)
    return headers, " ".join(moves)


def count_plies(movetext: str) -> int:
    """Count half-moves in movetext without a full chess parse (comments, numbers, result removed)."""
    text = MOVE_NUMBER_RE.sub(" ", COMMENT_RE.sub(" ", movetext))
    return sum(1 for tok in text.split() if tok not in RESULT_TOKENS)


def parse_base_time(time_control: str) -> int | None:
    """'180+2' -> 180. Returns None for '-' (correspondence) or malformed values."""
    base = time_control.split("+", 1)[0]
    return int(base) if base.isdigit() else None


def rejection_reason(headers: dict[str, str], plies: int, cfg: dict) -> str | None:
    """Return the name of the first filter the game fails, or None if it is kept.

    Order matters for the logged counts: each game is attributed to the first rule that removes it.
    """
    event = headers.get("Event", "")
    # The file only contains rated games, but Swiss events are labelled e.g. "Blitz swiss" without
    # the word "Rated", so only explicitly casual games are rejected here.
    if "Casual" in event:
        return "not_rated"
    if headers.get("Variant", "Standard") != "Standard" or "FEN" in headers:
        return "not_standard"
    if cfg["event_keyword"] not in event:
        return "not_blitz_event"
    base = parse_base_time(headers.get("TimeControl", "-"))
    if base is None or not cfg["base_time_min"] <= base <= cfg["base_time_max"]:
        return "base_time_out_of_range"
    if not (headers.get("WhiteElo", "").isdigit() and headers.get("BlackElo", "").isdigit()):
        return "missing_elo"
    if headers.get("Termination", "") in cfg["excluded_terminations"]:
        return "excluded_termination"
    if plies < cfg["min_plies"]:
        return "too_short"
    return None


def game_id_from_site(site: str) -> str:
    """'https://lichess.org/AbCdEfGh' -> 'AbCdEfGh'. Used only as a row/group identifier, never as a feature."""
    return site.rstrip("/").rsplit("/", 1)[-1]


def stream_lines(url: str, timeout: float = 60.0) -> Iterator[str]:
    """Yield decompressed text lines from a remote .zst file without storing it on disk."""
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        reader = zstandard.ZstdDecompressor().stream_reader(resp.raw)
        yield from io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def download(month: str, n_games: int, out_path: Path, cfg: dict, sample_size: int = 0,
             sample_path: Path | None = None, chunk_size: int = 10_000) -> Counter:
    """Stream, filter and save the first `n_games` kept games. Returns the filter counts."""
    url = cfg["data"]["url_template"].format(month=month)
    log.info("Streaming %s (keeping %d games)", url, n_games)
    counts: Counter = Counter()
    rows: list[dict] = []
    samples: list[str] = []
    start = time.time()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(out_path, SCHEMA) as writer:
        for pgn in iter_pgn_texts(stream_lines(url)):
            counts["seen"] += 1
            headers, movetext = parse_headers(pgn)
            plies = count_plies(movetext)
            reason = rejection_reason(headers, plies, cfg["filters"])
            if reason:
                counts[reason] += 1
                continue

            counts["kept"] += 1
            row = {"game_id": game_id_from_site(headers.get("Site", "")), "plies": plies, "pgn": pgn.strip()}
            row.update({h: headers.get(h) for h in HEADER_COLUMNS})
            rows.append(row)
            if len(samples) < sample_size:
                samples.append(pgn.strip())

            if len(rows) >= chunk_size:
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                rows.clear()
                log.info("kept %d / %d (seen %d, %.0f games/s)", counts["kept"], n_games,
                         counts["seen"], counts["seen"] / (time.time() - start))
            if counts["kept"] >= n_games:
                break
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))

    if sample_path and samples:
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text("\n\n".join(samples) + "\n", encoding="utf-8")
        log.info("Wrote %d sample games to %s", len(samples), sample_path)

    log.info("Done in %.0f s. Filter counts: %s", time.time() - start, dict(counts))
    return counts


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--month", default=cfg["data"]["month"])
    p.add_argument("--n", type=int, default=cfg["data"]["n_games"], help="games to keep after filtering")
    p.add_argument("--sample", type=int, default=0, help="also write this many kept games to the sample PGN")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    out_path = raw_dir / f"games_{args.month}_n{args.n}.parquet"
    counts = download(args.month, args.n, out_path, cfg, args.sample, ROOT / cfg["data"]["sample_pgn"])

    # Filter counts go into the report, so they are saved next to it (small, committed).
    counts_path = ROOT / "report" / f"filter_counts_{args.month}_n{args.n}.json"
    counts_path.write_text(json.dumps({"month": args.month, "n_games": args.n, **counts}, indent=2))
    log.info("Saved %s and %s", out_path, counts_path)


if __name__ == "__main__":
    main()
