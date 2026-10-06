"""Evaluate the saved model and produce every figure and table used in the report.

Outputs (report/figures/*.png, report/tables/*.csv|md, report/tables/summary.json):
  - prediction vs. actual, error by rating bucket and by game length (regression toward the mean)
  - permutation importance on the test set
  - learning curve (tuned model and Ridge, by number of training games)
  - evaluation on unseen players (split grouped by username)
  - a non-ML heuristic baseline (mean rating per time control)
  - feature set A vs. A+B (engine evals) on the games that have %eval
  - app latency on the sample games

Usage:
    python -m src.evaluate                 # everything
    python -m src.evaluate --skip learning_curve,set_b
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from multiprocessing import Pool
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from src.config import ROOT, load_config  # noqa: E402
from src.dataset import TARGET, load_table, raw_path  # noqa: E402
from src.features import eval_player_rows, extract_eval_features  # noqa: E402
from src.train import build_model, regression_metrics, save_table  # noqa: E402

log = logging.getLogger(__name__)

# Reference palette (dataviz skill): categorical slots 1-2 and the single-hue blue ramp.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

RATING_BUCKETS = [0, 1000, 1500, 2000, 4000]
RATING_LABELS = ["<1000", "1000-1499", "1500-1999", "2000+"]
PLY_BUCKETS = [0, 40, 60, 80, 100, 140, 1000]
PLY_LABELS = ["20-39", "40-59", "60-79", "80-99", "100-139", "140+"]


def style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Recessive grid and axes, text in ink tokens (never series colours)."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def save_fig(fig: plt.Figure, name: str, cfg: dict) -> None:
    out = ROOT / cfg["paths"]["figures_dir"] / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out.relative_to(ROOT))


def split_frames(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return table[table["split"] == "train"], table[table["split"] == "test"]


def bucket_errors(y: np.ndarray, pred: np.ndarray, key: pd.Series, labels: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"bucket": key, "y": y, "pred": pred, "abs_err": np.abs(y - pred)})
    out = df.groupby("bucket", observed=False).agg(
        rows=("y", "size"), MAE=("abs_err", "mean"), mean_actual=("y", "mean"), mean_predicted=("pred", "mean"))
    out["bias"] = out["mean_predicted"] - out["mean_actual"]
    return out.reindex(labels).reset_index()


# --------------------------------------------------------------------------------------- experiments

def predictions_and_errors(bundle: dict, test: pd.DataFrame, cfg: dict) -> dict:
    cols = bundle["feature_columns"]
    y = test[TARGET].to_numpy()
    pred = bundle["pipeline"].predict(test[cols])

    # Prediction vs. actual: density + identity line + mean prediction per actual-rating bin.
    fig, ax = plt.subplots(figsize=(6, 5.2))
    hb = ax.hexbin(y, pred, gridsize=60, cmap=BLUE_RAMP, mincnt=5, bins="log", linewidths=0)
    lims = [400, 3000]
    ax.plot(lims, lims, color=INK_2, linewidth=1, linestyle="--", label="Perfect prediction")
    bins = np.arange(400, 3100, 100)
    centers, means = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y >= lo) & (y < hi)
        if m.sum() >= 50:
            centers.append((lo + hi) / 2)
            means.append(pred[m].mean())
    ax.plot(centers, means, color=ORANGE, linewidth=2, label="Mean prediction per 100-point bin")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    style_axes(ax, "Predicted vs. actual rating (test set)", "Actual rating", "Predicted rating")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Rows (log scale)", color=INK_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    save_fig(fig, "pred_vs_actual", cfg)

    by_rating = bucket_errors(y, pred, pd.cut(test[TARGET], RATING_BUCKETS, labels=RATING_LABELS, right=False),
                              RATING_LABELS)
    by_length = bucket_errors(y, pred, pd.cut(test["plies"], PLY_BUCKETS, labels=PLY_LABELS, right=False),
                              PLY_LABELS)
    save_table(by_rating.round(1), "error_by_rating", cfg)
    save_table(by_length.round(1), "error_by_length", cfg)

    for df, name, xlabel in ((by_rating, "error_by_rating", "Actual rating"),
                             (by_length, "error_by_length", "Game length (plies)")):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        axes[0].bar(df["bucket"], df["MAE"], color=BLUE, width=0.6)
        for x, v in zip(df["bucket"], df["MAE"]):
            axes[0].annotate(f"{v:.0f}", (x, v), ha="center", va="bottom", fontsize=8, color=INK_2,
                             xytext=(0, 2), textcoords="offset points")
        style_axes(axes[0], "Mean absolute error", xlabel, "MAE (rating points)")
        axes[1].bar(df["bucket"], df["bias"], color=BLUE, width=0.6)
        axes[1].axhline(0, color=INK_2, linewidth=0.8)
        style_axes(axes[1], "Bias (mean predicted - mean actual)", xlabel, "Rating points")
        for ax in axes:
            ax.tick_params(axis="x", labelsize=8)
        save_fig(fig, name, cfg)

    abs_err = np.abs(y - pred)
    return {"test_metrics": regression_metrics(y, pred), "pred": pred,
            "coverage_within_mae": float(np.mean(abs_err <= abs_err.mean())),
            "by_rating": by_rating, "by_length": by_length}


def feature_importance(bundle: dict, test: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ecfg = cfg["evaluation"]
    sample = test.sample(min(ecfg["permutation_rows"], len(test)), random_state=cfg["seed"])
    cols = bundle["feature_columns"]
    log.info("Permutation importance on %d test rows x %d repeats", len(sample), ecfg["permutation_repeats"])
    res = permutation_importance(bundle["pipeline"], sample[cols], sample[TARGET],
                                 scoring="neg_mean_absolute_error", n_repeats=ecfg["permutation_repeats"],
                                 random_state=cfg["seed"], n_jobs=1)
    imp = pd.DataFrame({"feature": cols, "MAE_increase": res.importances_mean, "std": res.importances_std})
    imp = imp.sort_values("MAE_increase", ascending=False).reset_index(drop=True)
    save_table(imp.round(2), "permutation_importance", cfg)

    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["MAE_increase"], xerr=top["std"], color=BLUE, height=0.6,
            error_kw={"ecolor": INK_2, "elinewidth": 0.8})
    style_axes(ax, "Permutation importance (top 20, test set)", "Increase in MAE when shuffled (rating points)", "")
    ax.tick_params(axis="y", labelsize=8)
    save_fig(fig, "permutation_importance", cfg)
    return imp


def learning_curve(bundle: dict, table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Test MAE vs. number of training games, for the tuned model and Ridge (bias/variance view)."""
    train, test = split_frames(table)
    cols, seed = bundle["feature_columns"], cfg["seed"]
    games = train["game_id"].drop_duplicates().sample(frac=1.0, random_state=seed).to_list()
    rows = []
    for n in cfg["evaluation"]["learning_curve_games"]:
        n = min(n, len(games))
        sub = train[train["game_id"].isin(set(games[:n]))]
        for name, params in ((bundle["model_name"], bundle["params"]), ("ridge", None)):
            model = build_model(name, cols, seed, params)
            start = time.time()
            model.fit(sub[cols], sub[TARGET])
            rows.append({"model": name if name != bundle["model_name"] else f"{name} (tuned)", "games": n,
                         "train_MAE": regression_metrics(sub[TARGET], model.predict(sub[cols]))["MAE"],
                         "test_MAE": regression_metrics(test[TARGET], model.predict(test[cols]))["MAE"],
                         "fit_seconds": round(time.time() - start, 1)})
            log.info("learning curve %s n=%d test MAE %.1f", rows[-1]["model"], n, rows[-1]["test_MAE"])
    lc = pd.DataFrame(rows)
    save_table(lc.round(1), "learning_curve", cfg)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (name, grp), color in zip(lc.groupby("model", sort=False), (BLUE, ORANGE)):
        ax.plot(grp["games"], grp["test_MAE"], color=color, linewidth=2, marker="o", markersize=5,
                label=f"{name}: test")
        ax.plot(grp["games"], grp["train_MAE"], color=color, linewidth=1.2, linestyle="--", marker="o",
                markersize=4, label=f"{name}: train")
    ax.set_xscale("log")
    ax.set_xticks(lc["games"].unique())
    ax.set_xticklabels([f"{g // 1000}k" for g in lc["games"].unique()])
    style_axes(ax, "Learning curve", "Training games (log scale; 2 rows per game)", "MAE (rating points)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    save_fig(fig, "learning_curve", cfg)
    return lc


def unseen_players(bundle: dict, table: pd.DataFrame, cfg: dict) -> dict:
    """Re-split so test players never appear in training (usernames used ONLY for grouping).

    A random share of usernames is held out; every game involving one of them goes to test, and
    only rows of held-out players are scored, so the model has never seen those players.
    """
    rng = np.random.default_rng(cfg["seed"])
    players = table["player"].unique()
    held_out = set(rng.choice(players, int(len(players) * cfg["evaluation"]["player_test_share"]), replace=False))
    game_has_held = table.assign(h=table["player"].isin(held_out)).groupby("game_id")["h"].transform("any")
    train = table[~game_has_held]
    test = table[game_has_held & table["player"].isin(held_out)]
    cols = bundle["feature_columns"]
    model = build_model(bundle["model_name"], cols, cfg["seed"], bundle["params"])
    model.fit(train[cols], train[TARGET])
    m = regression_metrics(test[TARGET], model.predict(test[cols]))

    _, std_test = split_frames(table)
    train_players = set(table.loc[table["split"] == "train", "player"])
    return {"unseen_players_MAE": m["MAE"], "unseen_players_RMSE": m["RMSE"], "unseen_players_R2": m["R2"],
            "unseen_players_train_rows": len(train), "unseen_players_test_rows": len(test),
            "unique_players": int(len(players)),
            "share_test_rows_player_in_train_standard_split": float(std_test["player"].isin(train_players).mean())}


def heuristic_baseline(table: pd.DataFrame) -> dict:
    """Non-ML baseline: predict the mean training rating for the game's time control."""
    train, test = split_frames(table)
    tc = lambda df: df["base_time"].astype(str) + "+" + df["increment"].astype(str)  # noqa: E731
    means = train.groupby(tc(train))[TARGET].mean()
    pred = tc(test).map(means).fillna(train[TARGET].mean())
    return {f"heuristic_time_control_{k}": v for k, v in regression_metrics(test[TARGET], pred).items()}


def _eval_rows(args: tuple[str, str]) -> list[dict] | None:
    game_id, pgn = args
    try:
        f = extract_eval_features(pgn)
    except Exception:  # noqa: BLE001
        return None
    if f is None:
        return None
    rows = eval_player_rows(f)
    for row, color in zip(rows, (1, 0)):
        row.update({"game_id": game_id, "color": color})
    return rows


def set_b_experiment(bundle: dict, table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Feature set A vs. A+B on the games that carry engine evaluations (same train/test split)."""
    raw = pd.read_parquet(raw_path(cfg, cfg["data"]["n_games"]), columns=["game_id", "pgn"])
    raw = raw[raw["pgn"].str.contains("%eval", regex=False)]
    log.info("Set B: extracting eval features for %d games with %%eval", len(raw))
    rows: list[dict] = []
    with Pool() as pool:
        for r in pool.imap(_eval_rows, zip(raw["game_id"], raw["pgn"]), chunksize=200):
            if r:
                rows.extend(r)
    b = pd.DataFrame(rows)
    sub = table.merge(b, on=["game_id", "color"], how="inner")
    train, test = split_frames(sub)
    cols_a = bundle["feature_columns"]
    cols_b = cols_a + [c for c in b.columns if c not in ("game_id", "color")]
    seed = cfg["seed"]

    results = []
    deployed = bundle["pipeline"].predict(test[cols_a])
    results.append({"model": "Deployed model (set A, trained on all games)", "train_rows": bundle["trained_on"]["train_rows"],
                    **regression_metrics(test[TARGET], deployed)})
    for label, cols in (("Set A, trained on eval subset", cols_a), ("Set A+B, trained on eval subset", cols_b)):
        model = build_model(bundle["model_name"], cols, seed, bundle["params"])
        model.fit(train[cols], train[TARGET])
        results.append({"model": label, "train_rows": len(train),
                        **regression_metrics(test[TARGET], model.predict(test[cols]))})
    res = pd.DataFrame(results)
    res.insert(1, "test_rows", len(test))
    save_table(res.round(3), "set_a_vs_b", cfg)
    log.info("Set B results:\n%s", res.round(1).to_string(index=False))
    return res


def app_latency(cfg: dict) -> dict:
    from src.predict import predict_pgn  # imported here so evaluate does not depend on the app at import time
    text = (ROOT / cfg["data"]["sample_pgn"]).read_text(encoding="utf-8").strip()
    games = [g if i == 0 else "[Event " + g for i, g in enumerate(text.split("\n\n[Event "))]
    predict_pgn(games[0])  # warm-up: loads the model
    times = []
    for g in games:
        start = time.perf_counter()
        predict_pgn(g)
        times.append((time.perf_counter() - start) * 1000)
    return {"latency_ms_median": float(np.median(times)), "latency_ms_max": float(np.max(times)),
            "latency_games": len(games)}


# --------------------------------------------------------------------------------------------- main

def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=cfg["data"]["n_games"], help="which dataset (games kept), as in train.py")
    p.add_argument("--skip", default="", help="comma list: importance,learning_curve,unseen_players,set_b,latency")
    args = p.parse_args()
    cfg["data"]["n_games"] = args.n
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    bundle = joblib.load(ROOT / cfg["paths"]["models_dir"] / "model.joblib")
    table = load_table(cfg)
    train, test = split_frames(table)
    summary: dict = {"model": bundle["model_name"], "params": {k: (float(v) if isinstance(v, (int, float, np.number)) and not isinstance(v, bool) else v)
                                                                for k, v in bundle["params"].items()},
                     "rows": len(table), "games": int(table["game_id"].nunique()),
                     "train_rows": len(train), "test_rows": len(test),
                     "rating_mean": float(table[TARGET].mean()), "rating_std": float(table[TARGET].std())}

    res = predictions_and_errors(bundle, test, cfg)
    summary.update({f"test_{k}": v for k, v in res["test_metrics"].items()})
    summary["coverage_within_mae"] = res["coverage_within_mae"]
    summary.update(heuristic_baseline(table))
    if "importance" not in skip:
        summary["top_features"] = feature_importance(bundle, test, cfg).head(10)["feature"].tolist()
    if "learning_curve" not in skip:
        learning_curve(bundle, table, cfg)
    if "unseen_players" not in skip:
        summary.update(unseen_players(bundle, table, cfg))
    if "set_b" not in skip:
        set_b_experiment(bundle, table, cfg)
    if "latency" not in skip:
        summary.update(app_latency(cfg))

    out = ROOT / cfg["paths"]["tables_dir"] / "summary.json"
    out.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    log.info("Saved %s", out.relative_to(ROOT))
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, (list, dict))}, indent=2, default=float))


if __name__ == "__main__":
    main()
