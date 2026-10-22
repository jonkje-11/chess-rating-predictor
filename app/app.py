"""Streamlit website: estimate both players' ratings from one chess game.

UI only. All ML (header stripping, features, model) lives in src/predict.py.
"""

import base64
import random
import sys
from pathlib import Path

import chess
import chess.svg
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))  # lets this file import from src/ when Streamlit runs it

from src.predict import InvalidGame, fetch_lichess_pgn, predict_pgn  # noqa: E402

EXAMPLES_FILE = ROOT / "data" / "sample" / "example_games.pgn"  # test-split games, never seen in training

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, button, textarea, input {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {
  display: none !important;
}
.stApp { background: #fbfbfd; }
.block-container { max-width: 720px; padding-top: 4.5rem; padding-bottom: 3rem; }

.eyebrow { font-size: .78rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: #2a78d6; }
.hero { font-size: clamp(2.5rem, 7vw, 3.8rem); font-weight: 700; letter-spacing: -0.04em; line-height: 1.03;
        color: #0b0b0b; margin: .55rem 0 0; padding: 0; }
.lead { font-size: 1.18rem; line-height: 1.55; color: #6e6e73; margin: 1.1rem 0 2.4rem; max-width: 34rem; }

.stTextArea textarea { border-radius: 18px !important; border: 1px solid #dcdce1 !important; background: #ffffff !important;
                       padding: 1rem 1.15rem !important; font-size: .95rem; line-height: 1.5;
                       box-shadow: 0 1px 2px rgba(0,0,0,.04); }
.stTextArea textarea:focus { border-color: #2a78d6 !important; box-shadow: 0 0 0 4px rgba(42,120,214,.14) !important; }
.stTextArea [data-baseweb="textarea"], .stTextArea [data-baseweb="base-input"] { border: none !important; background: transparent !important; }

.stButton > button { border-radius: 999px; padding: .62rem 1.45rem; font-weight: 600; font-size: .95rem;
                     border: 1px solid #dcdce1; background: #ffffff; color: #0b0b0b; transition: all .15s ease; }
.stButton > button:hover { border-color: #0b0b0b; color: #0b0b0b; background: #ffffff; }
.stButton > button:focus:not(:active) { border-color: #0b0b0b; color: #0b0b0b; box-shadow: none; }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] { background: #0b0b0b; border-color: #0b0b0b; color: #ffffff; }
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { background: #2a78d6; border-color: #2a78d6; color: #ffffff; }

.cards { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 2.6rem 0 1.2rem; }
.card { background: #ffffff; border-radius: 24px; padding: 1.6rem 1.7rem 1.5rem;
        box-shadow: 0 12px 32px rgba(0,0,0,.06), 0 1px 3px rgba(0,0,0,.05); }
.who { display: flex; align-items: center; gap: .5rem; font-size: .78rem; font-weight: 600; color: #6e6e73;
       text-transform: uppercase; letter-spacing: .12em; }
.dot { width: .8rem; height: .8rem; border-radius: 50%; border: 1px solid #b9b9c0; display: inline-block; }
.dot.white { background: #ffffff; } .dot.black { background: #1d1d1f; border-color: #1d1d1f; }
.big { font-size: 3.6rem; font-weight: 700; letter-spacing: -0.045em; line-height: 1.05; margin: .55rem 0 .25rem;
       font-variant-numeric: tabular-nums; color: #0b0b0b; }
.range { color: #6e6e73; font-size: .95rem; }
.actual { margin-top: 1.1rem; padding-top: 1rem; border-top: 1px solid #ececf0; font-size: .92rem; color: #3a3a3c; }
.actual b { color: #0b0b0b; }
.pill { display: inline-block; margin-left: .35rem; padding: .1rem .55rem; border-radius: 999px; font-size: .75rem;
        font-weight: 600; background: #eef4fc; color: #1c5cab; }
.pill.out { background: #f3f3f5; color: #52514e; }

.board { display: flex; justify-content: center; margin: 1.6rem 0 .6rem; }
.board img { border-radius: 18px; box-shadow: 0 12px 32px rgba(0,0,0,.08); max-width: 100%; height: auto; }
.note { text-align: center; color: #6e6e73; font-size: .88rem; margin-bottom: 2rem; }

[data-testid="stExpander"] details { border: none !important; border-top: 1px solid #ececf0 !important; border-radius: 0 !important; background: transparent; }
[data-testid="stExpander"] summary { padding-left: 0 !important; font-weight: 600; color: #0b0b0b; }
.stAlert { border-radius: 16px; }
.footer { margin-top: 3.5rem; text-align: center; color: #a1a1a6; font-size: .78rem; line-height: 1.6; }
@media (max-width: 640px) { .cards { grid-template-columns: 1fr; } .block-container { padding-top: 2.5rem; } }
</style>
"""


def load_example():
    """Put a random unseen test game in the input box (a callback runs before the page is redrawn)."""
    text = EXAMPLES_FILE.read_text(encoding="utf-8")
    games = ["[Event " + part.strip() for part in text.split("[Event ") if part.strip()]
    st.session_state["game_text"] = random.choice(games)


def looks_like_link(text):
    """One line containing lichess.org, or a bare 8- or 12-character game ID."""
    text = text.strip()
    return "\n" not in text and ("lichess.org" in text or (len(text) in (8, 12) and text.isalnum()))


def card(color, entry, margin):
    """One player's result as an HTML card. Only our own numbers go into the HTML."""
    actual = ""
    if entry["true_rating"] is not None:
        inside = abs(entry["error"]) <= margin
        pill = "<span class='pill'>within range</span>" if inside else "<span class='pill out'>outside range</span>"
        actual = (f"<div class='actual'>Actual rating <b>{entry['true_rating']}</b> · "
                  f"off by {entry['error']:+d}{pill}</div>")
    return (f"<div class='card'><div class='who'><span class='dot {color}'></span>{color}</div>"
            f"<div class='big'>{entry['estimate']}</div>"
            f"<div class='range'>Likely {entry['estimate'] - margin}–{entry['estimate'] + margin}</div>"
            f"{actual}</div>")


def board_image(fen):
    """The final position as a soft, rounded board image drawn by python-chess."""
    colors = {"square light": "#f2f2f5", "square dark": "#b9c6d8", "margin": "#ffffff"}
    svg = chess.svg.board(chess.Board(fen), size=380, coordinates=False, colors=colors)
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"<div class='board'><img src='data:image/svg+xml;base64,{data}' width='380' alt='Final position'></div>"


st.set_page_config(page_title="Chess Rating Predictor", page_icon="♞", layout="centered")
st.markdown(STYLE, unsafe_allow_html=True)
st.markdown(
    "<div class='eyebrow'>Machine learning · Lichess blitz</div>"
    "<h1 class='hero'>How strong were they?</h1>"
    "<p class='lead'>Paste one game. The model reads the moves and the clock times and estimates both "
    "players' rating, without ever seeing the real one.</p>",
    unsafe_allow_html=True,
)

text = st.text_area("Game", key="game_text", height=150, label_visibility="collapsed",
                    placeholder="Paste a Lichess link (https://lichess.org/…) or the game's PGN")
go_column, example_column, _ = st.columns([1.25, 1.25, 2])
go = go_column.button("Estimate ratings", type="primary")
example_column.button("Try an example", on_click=load_example)

if go:
    try:
        if looks_like_link(text):
            with st.spinner("Fetching the game from Lichess…"):
                pgn = fetch_lichess_pgn(text)
        else:
            pgn = text
        result = predict_pgn(pgn)
    except InvalidGame as error:
        st.error(str(error))
    else:
        margin = result["margin"]
        st.markdown(f"<div class='cards'>{card('white', result['white'], margin)}"
                    f"{card('black', result['black'], margin)}</div>", unsafe_allow_html=True)
        st.markdown(board_image(result["final_fen"]), unsafe_allow_html=True)
        st.markdown(f"<p class='note'>Final position · about half of all estimates land within ±{margin}</p>",
                    unsafe_allow_html=True)

with st.expander("How accurate is this?"):
    st.write(
        "The model was trained on 250,000 Lichess blitz games from September 2026 and tested on 50,000 other "
        "games. On average it misses by about 219 rating points. It works best for typical players (1200–2000). "
        "When a game gives little evidence it pulls the guess toward the middle (about 1600), so beginners are "
        "often guessed too high and strong players too low. It only knows Lichess blitz: ratings on other sites "
        "or in other time controls are not comparable. The example games were never seen during training."
    )

st.markdown("<div class='footer'>Trained on 250,000 Lichess blitz games (September 2026) · "
            "average error 219 rating points<br>Data: Lichess open database (CC0)</div>", unsafe_allow_html=True)
