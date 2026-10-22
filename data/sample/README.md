# Sample data (committed)

Both files contain real games from the Lichess open database, September 2026 (CC0), in their original form.

| File | Games | Used by | Note |
|---|---:|---|---|
| `sample_games.pgn` | 40 | The tests | Mostly from the training split. The tests check values against these exact games, so don't change them. |
| `example_games.pgn` | 12 | The app's "Try an example" button | **All from the test split**, never seen in training. 3 per rating band (<1200, 1200–1599, 1600–1999, 2000+), so the demo shows honest results, including the weaker guesses at the extremes. |
