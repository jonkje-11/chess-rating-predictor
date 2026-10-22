"""Helpers for PGN text. Most important: removing headers that leak player strength."""

# Header lines that reveal who played or how strong they are.
# The model must never see these, in training or in the app.
LEAKY_HEADERS = {
    "WhiteElo", "BlackElo",
    "WhiteRatingDiff", "BlackRatingDiff",
    "WhiteTitle", "BlackTitle",
    "White", "Black",
    "Site", "GameId",
}


def strip_rating_headers(pgn):
    """Remove leaky headers from one game.

    Returns the cleaned PGN text and a dict with the removed values,
    so the app can show the true ratings after the prediction.
    """
    kept_lines = []
    removed = {}
    for line in pgn.splitlines():
        if line.startswith("["):
            name = line[1:].split(" ", 1)[0]
            if name in LEAKY_HEADERS:
                removed[name] = line.split('"')[1]
                continue
        kept_lines.append(line)
    return "\n".join(kept_lines), removed
