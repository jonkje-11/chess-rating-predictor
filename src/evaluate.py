"""Evaluate the saved model on the test set and make the report's tables and figures."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # draw straight to image files, no window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.base import clone
from sklearn.inspection import permutation_importance

import joblib
import numpy as np
import pandas as pd

from src.config import load_config, players_table_file
from src.train import MODEL_FILE, TABLES_DIR, scores

CONFIG = load_config()
SEED = CONFIG["seed"]
METRICS_FILE = Path("models/metrics.json")
FIGURES_DIR = Path("docs/report/figures")

# Chart colours from a validated colour-blind-safe palette.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"  # text
MUTED = "#52514e"  # axis labels, reference lines
GRID = "#e4e3df"
SURFACE = "#fcfcfb"  # background
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])


RATING_BINS = [0, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 4000]
RATING_LABELS = ["<1000", "1000-1199", "1200-1399", "1400-1599", "1600-1799", "1800-1999", "2000-2199", "2200+"]
LENGTH_BINS = [0, 40, 60, 80, 100, 1000]
LENGTH_LABELS = ["20-39", "40-59", "60-79", "80-99", "100+"]


def load_data():
    """The training and test rows, plus the saved model and its feature list."""
    table = pd.read_parquet(players_table_file(CONFIG))
    train = table[table["split"] == "train"]
    test = table[table["split"] == "test"]
    saved = joblib.load(MODEL_FILE)
    return train, test, saved["model"], saved["features"]


def headline(train, test, predictions):
    """The official result: test-set scores, the lazy guesser, and the pass/fail check."""
    pass_mark = CONFIG["evaluation"]["pass_mark"]
    lazy = np.full(len(test), train["rating"].mean())
    model_scores = scores(test["rating"], predictions)
    lazy_scores = scores(test["rating"], lazy)
    errors = np.abs(predictions - test["rating"].to_numpy())
    metrics = {
        "test_rows": len(test),
        "mae": model_scores["MAE"],
        "rmse": model_scores["RMSE"],
        "r2": model_scores["R2"],
        "lazy_mae": lazy_scores["MAE"],
        "share_within_mae": float(np.mean(errors <= model_scores["MAE"])),
        "pass_mark": pass_mark,
        "passed": bool(model_scores["MAE"] <= pass_mark and model_scores["MAE"] < lazy_scores["MAE"]),
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    return metrics


def error_table(test, predictions, column, bins, labels):
    """Errors split into groups, e.g. by true rating or by game length.

    mean_error > 0 means the model guesses too HIGH for that group, < 0 too LOW.
    """
    errors = predictions - test["rating"].to_numpy()
    groups = pd.cut(test[column], bins=bins, labels=labels, right=False)
    grouped = pd.DataFrame({"group": groups, "error": errors}).groupby("group", observed=True)["error"]
    return pd.DataFrame({
        "rows": grouped.size(),
        "MAE": grouped.apply(lambda e: e.abs().mean()),
        "mean_error": grouped.mean(),
    }).reset_index()

def new_figure(title, xlabel, ylabel, size=(7, 4.5)):
    """An empty chart with the project's shared style: light grid, no box, muted labels."""
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    return fig, ax


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_predicted_vs_actual(test, predictions):
    """Every test player as a dot, binned into hexagons; darker = more players."""
    true = test["rating"].to_numpy()
    fig, ax = new_figure("Predicted vs. true rating (test set)", "True rating", "Predicted rating", size=(6.5, 6))
    ax.grid(False)
    cells = ax.hexbin(true, predictions, gridsize=60, cmap=BLUES, mincnt=1, bins="log", linewidths=0)
    fig.colorbar(cells, ax=ax, label="Players (log scale)")

    ax.plot([400, 3000], [400, 3000], color=MUTED, linestyle="--", linewidth=1.5)
    ax.text(2950, 2850, "perfect guess", color=MUTED, fontsize=9, ha="right")

    centers = np.arange(600, 2700, 100)
    means = [predictions[(true >= c - 50) & (true < c + 50)].mean() for c in centers]
    ax.plot(centers, means, color=ORANGE, linewidth=2)
    ax.text(centers[-1] - 40, means[-1] + 90, "average guess", color=INK, fontsize=9, ha="right",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))

    ax.set_xlim(400, 3000)
    ax.set_ylim(400, 3000)
    save(fig, "predicted_vs_actual.png")


def plot_error_by_rating(by_rating):
    """Average signed error per rating group: shows regression toward the mean."""
    fig, ax = new_figure("Average error by true rating (test set)", "True rating", "Predicted minus true rating")
    colors = [ORANGE if error > 0 else BLUE for error in by_rating["mean_error"]]
    bars = ax.bar(by_rating["group"].astype(str), by_rating["mean_error"], color=colors, width=0.7)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.bar_label(bars, fmt="%+.0f", color=INK, fontsize=9, padding=3)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(handles=[Patch(color=ORANGE, label="guessed too high"), Patch(color=BLUE, label="guessed too low")],
              frameon=False)
    save(fig, "error_by_rating.png")


def feature_importance(model, test, features):
    """How much worse the model gets when one feature is shuffled (permutation importance)."""
    sample = test.sample(n=CONFIG["evaluation"]["permutation_rows"], random_state=SEED)
    result = permutation_importance(
        model, sample[features], sample["rating"],
        n_repeats=CONFIG["evaluation"]["permutation_repeats"],
        scoring="neg_mean_absolute_error", random_state=SEED,
    )
    importance = pd.DataFrame({
        "feature": features,
        "mae_increase": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("mae_increase", ascending=False)
    importance.to_csv(TABLES_DIR / "feature_importance.csv", index=False)
    return importance


def plot_importance(importance, top=15):
    rows = importance.head(top).iloc[::-1]  # reversed, so the most important ends up on top
    fig, ax = new_figure(f"Top {top} features (permutation importance, test set)",
                         "Increase in MAE when the feature is shuffled (rating points)", "", size=(7, 6))
    ax.barh(rows["feature"], rows["mae_increase"], color=BLUE, height=0.6)
    ax.grid(axis="y", visible=False)
    save(fig, "feature_importance.png")


def learning_curve(model, train, test, features):
    """Train the final model's settings on more and more games and measure the error."""
    game_ids = np.random.default_rng(SEED).permutation(train["game_id"].unique())
    rows = []
    for n_games in CONFIG["evaluation"]["learning_curve_games"]:
        subset = train[train["game_id"].isin(game_ids[:n_games])]
        fitted = clone(model).fit(subset[features], subset["rating"])
        rows.append({
            "games": n_games,
            "train_MAE": scores(subset["rating"], fitted.predict(subset[features]))["MAE"],
            "test_MAE": scores(test["rating"], fitted.predict(test[features]))["MAE"],
        })
        print(rows[-1])
    curve = pd.DataFrame(rows)
    curve.to_csv(TABLES_DIR / "learning_curve.csv", index=False)
    return curve


def plot_learning_curve(curve, pass_mark):
    fig, ax = new_figure("Learning curve", "Training games (log scale)", "MAE (rating points)")
    for column, label, color in (("test_MAE", "Test set", BLUE), ("train_MAE", "Training set", ORANGE)):
        ax.plot(curve["games"], curve[column], color=color, linewidth=2, marker="o", markersize=6, label=label)
        ax.text(curve["games"].iloc[-1] * 1.12, curve[column].iloc[-1], label, color=INK, fontsize=9, va="center")
    ax.axhline(pass_mark, color=MUTED, linestyle="--", linewidth=1)
    ax.text(curve["games"].iloc[0], pass_mark + 3, f"pass mark ({pass_mark})", color=MUTED, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(curve["games"], labels=[f"{games // 1000}k" for games in curve["games"]])
    ax.minorticks_off()
    ax.set_xlim(right=curve["games"].iloc[-1] * 2.2)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    save(fig, "learning_curve.png")


def main():
    train, test, model, features = load_data()
    predictions = model.predict(test[features])

    metrics = headline(train, test, predictions)
    print(json.dumps(metrics, indent=2))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    by_rating = error_table(test, predictions, "rating", RATING_BINS, RATING_LABELS)
    by_length = error_table(test, predictions, "plies", LENGTH_BINS, LENGTH_LABELS)
    by_rating.to_csv(TABLES_DIR / "error_by_rating.csv", index=False)
    by_length.to_csv(TABLES_DIR / "error_by_length.csv", index=False)
    print(by_rating.round(1).to_string(index=False))
    print(by_length.round(1).to_string(index=False))

    plot_predicted_vs_actual(test, predictions)
    plot_error_by_rating(by_rating)

    print("Computing feature importance (about a minute)...")
    importance = feature_importance(model, test, features)
    print(importance.head(15).round(2).to_string(index=False))
    plot_importance(importance)

    print("Computing learning curve (a few minutes)...")
    curve = learning_curve(model, train, test, features)
    plot_learning_curve(curve, CONFIG["evaluation"]["pass_mark"])
    print("Figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
