"""Gradio web app: paste a PGN, upload a .pgn file or give a Lichess link -> estimated ratings.

This file is UI only. All ML, feature and leakage logic lives in src/predict.py (`predict_pgn`),
so the frontend can be replaced later without touching the model code.

Run locally:  python app/app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import chess
import chess.svg
import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow `python app/app.py` and the HF Space root app.py

from src.pgn_utils import PGNError  # noqa: E402
from src.predict import fetch_lichess_pgn, load_bundle, predict_pgn  # noqa: E402

SAMPLE_PGN = ROOT / "data" / "sample" / "sample_games.pgn"
BOARD_COLORS = {"square light": "#f0d9b5", "square dark": "#b58863",
                "square light lastmove": "#cdd26a", "square dark lastmove": "#aaa23a"}

CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto; }
#title h1 { font-size: 2rem; margin-bottom: 0.2rem; }
#title p { color: var(--body-text-color-subdued); margin-top: 0; }
.rp-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.rp-card { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 16px;
           background: var(--block-background-fill); }
.rp-card .side { display: flex; align-items: center; gap: 8px; font-weight: 600;
                 color: var(--body-text-color-subdued); text-transform: uppercase; font-size: 0.8rem;
                 letter-spacing: 0.04em; }
.rp-card .dot { width: 14px; height: 14px; border-radius: 50%; border: 1px solid #888; }
.rp-card .dot.white { background: #f5f5f5; } .rp-card .dot.black { background: #222; }
.rp-card .est { font-size: 2.4rem; font-weight: 700; line-height: 1.1; margin: 8px 0 2px;
                color: var(--body-text-color); }
.rp-card .range { color: var(--body-text-color-subdued); font-size: 0.9rem; }
.rp-card .actual { margin-top: 10px; padding-top: 10px; border-top: 1px dashed var(--border-color-primary);
                   font-size: 0.95rem; color: var(--body-text-color); }
.rp-badge { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
            margin-left: 6px; }
.rp-badge.good { background: #d8f0e3; color: #0f5c35; }
.rp-badge.ok   { background: #fdf0d2; color: #6b4a00; }
.rp-badge.off  { background: #fbdcdc; color: #7a1d1d; }
.rp-meta { margin-top: 12px; color: var(--body-text-color-subdued); font-size: 0.85rem; }
.rp-note { margin-top: 12px; padding: 10px 12px; border-radius: 10px; font-size: 0.85rem;
           background: var(--background-fill-secondary); color: var(--body-text-color-subdued); }
.rp-error { border: 1px solid #e3a0a0; background: #fdf2f2; color: #7a1d1d; border-radius: 12px; padding: 14px; }
.rp-empty { color: var(--body-text-color-subdued); padding: 24px; text-align: center;
            border: 1px dashed var(--border-color-primary); border-radius: 12px; }
#board svg { width: 100%; height: auto; max-width: 340px; display: block; margin: 0 auto; border-radius: 6px; }
@media (max-width: 640px) { .rp-cards { grid-template-columns: 1fr; } }
"""

THEME = gr.themes.Soft(primary_hue="amber", neutral_hue="stone",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

EMPTY_RESULT = '<div class="rp-empty">Paste a game, upload a file or enter a Lichess link, then press <b>Estimate</b>.</div>'


def _badge(error: int) -> str:
    size = abs(error)
    kind = "good" if size <= 100 else "ok" if size <= 250 else "off"
    return f'<span class="rp-badge {kind}">{"+" if error > 0 else ""}{error}</span>'


def _card(side: str, p: dict) -> str:
    actual = ""
    if p["actual"] is not None:
        actual = (f'<div class="actual">Actual rating: <b>{p["actual"]}</b>{_badge(p["error"])}'
                  f'<div class="range">model was off by {abs(p["error"])} points</div></div>')
    return (f'<div class="rp-card"><div class="side"><span class="dot {side}"></span>{side}</div>'
            f'<div class="est">~{p["estimate"]}</div>'
            f'<div class="range">likely range {p["low"]}–{p["high"]}</div>{actual}</div>')


def render_result(res: dict) -> str:
    g = res["game"]
    meta = [f'{g["plies"]} half-moves', f'opening {html.escape(str(g["eco"]))}',
            f'result {html.escape(str(g["result"]))}']
    if g["time_control"]:
        meta.append(f'time control {g["time_control"]}')
    if not g["has_clocks"]:
        meta.append("no clock data (less accurate)")
    reveal = ("The real ratings were hidden from the model and are only shown for comparison."
              if res["had_true_ratings"] else "No ratings were found in the input, so there is nothing to compare with.")
    return (f'<div class="rp-cards">{_card("white", res["white"])}{_card("black", res["black"])}</div>'
            f'<div class="rp-meta">± {res["margin"]} is the model\'s average error on unseen test games '
            f'(about {round(res["margin_coverage"] * 100)}% of estimates fall within it). {reveal}<br>'
            f'{" · ".join(meta)} · computed in {res["latency_ms"]:.0f} ms</div>'
            f'<div class="rp-note"><b>Limitations.</b> {html.escape(res["limitations"])}</div>')


def render_board(res: dict) -> str:
    g = res["game"]
    board = chess.Board(g["final_fen"])
    last = chess.Move.from_uci(g["last_move"]) if g["last_move"] else None
    return chess.svg.board(board, lastmove=last, size=340, colors=BOARD_COLORS, coordinates=True)


def _run(pgn: str) -> tuple[str, str]:
    try:
        res = predict_pgn(pgn)
    except PGNError as exc:
        return f'<div class="rp-error"><b>Could not use this game.</b> {html.escape(str(exc))}</div>', ""
    return render_result(res), render_board(res)


def from_text(pgn: str) -> tuple[str, str]:
    return _run(pgn or "")


def from_file(path: str | None) -> tuple[str, str]:
    if not path:
        return '<div class="rp-error">Please choose a .pgn file first.</div>', ""
    return _run(Path(path).read_text(encoding="utf-8", errors="replace"))


def from_lichess(url: str) -> tuple[str, str]:
    try:
        pgn = fetch_lichess_pgn(url or "")
    except PGNError as exc:
        return f'<div class="rp-error"><b>Could not fetch the game.</b> {html.escape(str(exc))}</div>', ""
    return _run(pgn)


def example_games(k: int = 3) -> list[str]:
    """Pick a low-, mid- and high-rated game from the committed sample file."""
    text = SAMPLE_PGN.read_text(encoding="utf-8").strip()
    games = [g if i == 0 else "[Event " + g for i, g in enumerate(text.split("\n\n[Event "))]

    def avg_elo(g: str) -> float:
        vals = [int(line.split('"')[1]) for line in g.splitlines()
                if line.startswith(("[WhiteElo", "[BlackElo")) and line.split('"')[1].isdigit()]
        return sum(vals) / len(vals) if vals else 0

    ranked = sorted(games, key=avg_elo)
    return [ranked[0], ranked[len(ranked) // 2], ranked[-1]][:k]


def model_footer() -> str:
    try:
        b = load_bundle()
    except FileNotFoundError:
        return "No trained model found. Run `python -m src.train`."
    t = b["trained_on"]
    names = {"hgb": "gradient boosting (HistGradientBoostingRegressor)", "ridge": "Ridge regression",
             "random_forest": "random forest"}
    return (f"Model: {names.get(b['model_name'], b['model_name'])}, scikit-learn, trained on {t['n_games']:,} Lichess blitz games from "
            f"{t['month']} · test MAE {b['test_metrics']['MAE']:.0f} rating points · "
            "Data: Lichess open database (CC0) · "
            "[Source code and report](https://github.com/jonkje-11/chess-rating-predictor)")


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Chess Rating Predictor") as demo:
        gr.Markdown("# ♞ Chess Rating Predictor\nEstimate how strong both players are from the moves of one "
                    "game. The model never sees the players' real ratings.", elem_id="title")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("Paste PGN"):
                        pgn_box = gr.Textbox(lines=12, max_lines=20, show_label=False,
                                             placeholder='[Event "Rated Blitz game"]\n...\n\n1. e4 e5 2. Nf3 ...')
                        text_btn = gr.Button("Estimate", variant="primary")
                        gr.Examples(example_games(), inputs=pgn_box, label="Example games (click one, then Estimate)",
                                    example_labels=["Example: lower-rated game", "Example: mid-rated game",
                                                    "Example: higher-rated game"])
                    with gr.Tab("Upload file"):
                        file_in = gr.File(file_types=[".pgn", ".txt"], type="filepath", label="PGN file (first game is used)")
                        file_btn = gr.Button("Estimate", variant="primary")
                    with gr.Tab("Lichess link"):
                        url_in = gr.Textbox(label="Lichess game URL or ID",
                                            placeholder="https://lichess.org/AbCdEfGh")
                        url_btn = gr.Button("Fetch and estimate", variant="primary")
            with gr.Column(scale=6):
                result = gr.HTML(EMPTY_RESULT)
                board = gr.HTML("", elem_id="board")
        gr.Markdown(model_footer())

        text_btn.click(from_text, pgn_box, [result, board])
        file_btn.click(from_file, file_in, [result, board])
        url_btn.click(from_lichess, url_in, [result, board])
        url_in.submit(from_lichess, url_in, [result, board])
    return demo


def main() -> None:
    build_demo().launch(theme=THEME, css=CSS)


if __name__ == "__main__":
    main()
