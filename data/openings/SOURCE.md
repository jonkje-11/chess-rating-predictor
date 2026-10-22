# Source of the opening files

The files `a.tsv` to `e.tsv` are downloaded unchanged from the Lichess chess-openings project:

- Repository: https://github.com/lichess-org/chess-openings
- Version: commit `a6189a30dc273ccb21fc2536a9a2fefd5592a67a`, downloaded 2026-10-21
- Licence: Creative Commons Zero v1.0 Universal (CC0)
- Contents: 3,865 named opening lines, one file per ECO volume (A–E), with the columns `eco`, `name` and `pgn`

Used by `src/features.py` as a reference list: to measure how many moves each player stays in known opening lines, and to derive the ECO code when a game has none.

To download the same version again:

```
for f in a b c d e; do
  curl -sfL -o data/openings/$f.tsv https://raw.githubusercontent.com/lichess-org/chess-openings/a6189a30dc273ccb21fc2536a9a2fefd5592a67a/$f.tsv
done
```
