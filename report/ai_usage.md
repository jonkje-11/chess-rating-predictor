# AI usage log

The project uses Claude Code (Anthropic, model Claude Opus 5.5) as a coding assistant. This log records what the AI produced, so the report can document it honestly. All design decisions were reviewed by the student.

| Date | What the AI did |
|---|---|
| 2026-10-06 | Read the course assignment and report template; proposed the project plan and aligned AGENTS.md with the course requirements. |
| 2026-10-06 | Set up git repo, `.gitignore`/`.gitattributes`, folder structure, pinned `requirements.txt`, `config.yaml`, `Makefile`, README skeleton, report skeleton (following the course template) and this log. |
| 2026-10-06 | Wrote `src/download.py` (streams the Lichess month over HTTP, decompresses on the fly, filters on headers, logs per-rule counts, writes Parquet in chunks) and `src/config.py`. Verified the fast ply counter against python-chess (0 mismatches on 2,000 games) and fixed a filter bug where Swiss games were wrongly treated as unrated. |
| 2026-10-06 | Wrote `src/pgn_utils.py` (`strip_rating_headers`, parsing/validation with user-facing errors, Lichess game-ID extraction) and `tests/test_pgn_utils.py` (16 tests, incl. checking that no leaky header survives stripping on all sample games). Recorded the frontend decision (Gradio + custom CSS, prediction logic isolated in `src/predict.py` for an easy pivot) in AGENTS.md. |
