"""Train and compare models, tune the best one, refit it on the full training set and save it.

Model selection uses a validation split carved out of the training set (grouped by game), and tuning
uses grouped cross-validation on the training set. The test set is only used to report final numbers.

Usage:
    python -m src.train                          # all models + tuning, settings from config.yaml
    python -m src.train --n 20000 --models dummy,hgb --no-tune
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import ROOT, load_config
from src.dataset import TARGET, feature_columns, load_table
from src.features import CATEGORICAL_FEATURES

log = logging.getLogger(__name__)

MODEL_ORDER = ["dummy", "ridge", "random_forest", "hgb"]

# Search space for HistGradientBoosting. Early stopping is off during the search because its internal
# validation split is random by row, which would put the two rows of a game on both sides.
HGB_SEARCH_SPACE = {
    "model__learning_rate": loguniform(0.03, 0.3),
    "model__max_iter": randint(200, 800),
    "model__max_leaf_nodes": randint(15, 128),
    "model__min_samples_leaf": randint(20, 400),
    "model__l2_regularization": loguniform(1e-3, 10),
}


def column_groups(cols: list[str]) -> tuple[list[str], list[str]]:
    """Split feature columns into (categorical, numeric), categorical first as the pipelines expect."""
    cat = [c for c in cols if c in CATEGORICAL_FEATURES]
    return cat, [c for c in cols if c not in CATEGORICAL_FEATURES]


def _ordinal_encoder() -> OrdinalEncoder:
    # Rare ECO codes are grouped so every categorical fits HistGradientBoosting's 255-bin limit.
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                          encoded_missing_value=np.nan, max_categories=250)


def make_models(cat_cols: list[str], num_cols: list[str], seed: int) -> dict[str, Pipeline]:
    """All candidate models as full pipelines, so preprocessing is saved together with the model."""
    linear_prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), cat_cols),
        # Linear models cannot take NaN: impute and keep an is_missing indicator per column.
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num_cols),
    ])
    tree_prep = ColumnTransformer([
        ("cat", _ordinal_encoder(), cat_cols),
        ("num", "passthrough", num_cols),  # trees handle NaN natively
    ])
    is_cat = [True] * len(cat_cols) + [False] * len(num_cols)
    return {
        "dummy": Pipeline([("model", DummyRegressor(strategy="mean"))]),
        "ridge": Pipeline([("prep", linear_prep), ("model", Ridge(alpha=1.0))]),
        # Leaf size and row subsampling keep the forest small enough to save and fast enough to train.
        "random_forest": Pipeline([("prep", tree_prep), ("model", RandomForestRegressor(
            n_estimators=100, min_samples_leaf=20, max_features=0.33, max_samples=0.5,
            n_jobs=-1, random_state=seed))]),
        "hgb": Pipeline([("prep", tree_prep), ("model", HistGradientBoostingRegressor(
            categorical_features=is_cat, max_iter=500, learning_rate=0.1, early_stopping=True,
            random_state=seed))]),
    }


def build_model(name: str, cols: list[str], seed: int, params: dict | None = None) -> Pipeline:
    """A fresh, unfitted pipeline for `name`, optionally with tuned parameters applied."""
    model = make_models(*column_groups(cols), seed)[name]
    return model.set_params(**params) if params else model


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def ordered_columns(table: pd.DataFrame) -> list[str]:
    cat, num = column_groups(feature_columns(table))
    return cat + num


def _evaluate(name: str, model: Pipeline, fit_s: float, sets: dict) -> dict:
    row = {"model": name, "fit_seconds": round(fit_s, 1), "size_MB": round(len(pickle.dumps(model)) / 1e6, 2)}
    for split, (X, y) in sets.items():
        row.update({f"{split}_{k}": v for k, v in regression_metrics(y, model.predict(X)).items()})
    return row


def compare_models(table: pd.DataFrame, names: list[str], cfg: dict) -> pd.DataFrame:
    """Fit each model on train-minus-validation; report train, validation (for selection) and test metrics."""
    seed, cols = cfg["seed"], ordered_columns(table)
    train = table[table["split"] == "train"].reset_index(drop=True)
    test = table[table["split"] == "test"]
    fit_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=cfg["split"]["validation_size"],
                                              random_state=seed).split(train, groups=train["game_id"]))
    fit, val = train.iloc[fit_idx], train.iloc[val_idx]
    sets = {"train": (fit[cols], fit[TARGET]), "val": (val[cols], val[TARGET]), "test": (test[cols], test[TARGET])}

    results = []
    for name in names:
        model = build_model(name, cols, seed)
        start = time.time()
        model.fit(*sets["train"])
        row = _evaluate(name, model, time.time() - start, sets)
        results.append(row)
        log.info("%-14s val MAE %.1f | test MAE %.1f | fit %.0fs", name, row["val_MAE"], row["test_MAE"],
                 row["fit_seconds"])
    return pd.DataFrame(results)


def tune(table: pd.DataFrame, name: str, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Random search with grouped K-fold CV on (a subset of) the training games. Returns best params + log."""
    if name != "hgb":
        log.info("No search space defined for %s; skipping tuning.", name)
        return {}, pd.DataFrame()
    tcfg, seed, cols = cfg["tuning"], cfg["seed"], ordered_columns(table)
    train = table[table["split"] == "train"]
    games = train["game_id"].drop_duplicates()
    if len(games) > tcfg["max_games"]:
        games = games.sample(tcfg["max_games"], random_state=seed)
        train = train[train["game_id"].isin(set(games))]

    model = build_model(name, cols, seed, {"model__early_stopping": False})
    search = RandomizedSearchCV(
        model, HGB_SEARCH_SPACE, n_iter=tcfg["n_iter"], cv=GroupKFold(n_splits=tcfg["cv_folds"]),
        scoring="neg_mean_absolute_error", random_state=seed, n_jobs=1, refit=False, verbose=1,
    )
    log.info("Tuning %s on %d rows (%d games), %d candidates x %d folds", name, len(train), len(games),
             tcfg["n_iter"], tcfg["cv_folds"])
    start = time.time()
    search.fit(train[cols], train[TARGET], groups=train["game_id"])
    log.info("Tuning took %.0f s. Best CV MAE %.1f with %s", time.time() - start, -search.best_score_,
             search.best_params_)

    res = pd.DataFrame(search.cv_results_)
    keep = [c for c in res.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score", "mean_fit_time"]
    res = res[keep].rename(columns=lambda c: c.replace("param_model__", ""))
    res["cv_MAE"] = -res.pop("mean_test_score")
    res["cv_MAE_std"] = res.pop("std_test_score")
    params = {**search.best_params_, "model__early_stopping": False}
    return params, res.sort_values("cv_MAE").reset_index(drop=True)


def save_table(df: pd.DataFrame, stem: str, cfg: dict) -> None:
    out_dir = ROOT / cfg["paths"]["tables_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(df.to_markdown(index=False, floatfmt=".4g") + "\n", encoding="utf-8")


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=cfg["data"]["n_games"])
    p.add_argument("--models", default=",".join(MODEL_ORDER), help=f"comma list from {MODEL_ORDER}")
    p.add_argument("--no-tune", action="store_true", help="skip the hyperparameter search")
    p.add_argument("--no-save", action="store_true", help="compare only, do not save the best model")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed = cfg["seed"]

    table = load_table(cfg, args.n)
    log.info("Loaded %d rows (%d games)", len(table), table["game_id"].nunique())
    results = compare_models(table, [m.strip() for m in args.models.split(",")], cfg)
    pd.set_option("display.width", 200)
    print(results[["model", "val_MAE", "test_MAE", "test_RMSE", "test_R2", "train_MAE", "fit_seconds", "size_MB"]]
          .round(3).to_string(index=False))
    if args.no_save:
        save_table(results.round(4), f"model_comparison_n{args.n}", cfg)
        return

    candidates = results[results["model"] != "dummy"]
    best = (candidates if len(candidates) else results).sort_values("val_MAE").iloc[0]["model"]
    log.info("Best model by validation MAE: %s", best)
    params: dict = {}
    if cfg["tuning"]["enabled"] and not args.no_tune:
        params, search_log = tune(table, best, cfg)
        if len(search_log):
            save_table(search_log.round(4), f"tuning_{best}_n{args.n}", cfg)

    cols = ordered_columns(table)
    train, test = table[table["split"] == "train"], table[table["split"] == "test"]
    model = build_model(best, cols, seed, params)
    start = time.time()
    model.fit(train[cols], train[TARGET])
    fit_s = time.time() - start
    pred = model.predict(test[cols])
    metrics = regression_metrics(test[TARGET], pred)
    abs_err = np.abs(test[TARGET].to_numpy() - pred)

    # Add the final (tuned, full-train) model to the comparison table; its val metrics are not comparable.
    final_row = _evaluate(f"{best}_final", model, fit_s, {"train": (train[cols], train[TARGET]),
                                                          "test": (test[cols], test[TARGET])})
    results = pd.concat([results, pd.DataFrame([final_row])], ignore_index=True)
    save_table(results.round(4), f"model_comparison_n{args.n}", cfg)

    bundle = {
        "pipeline": model,
        "model_name": best,
        "params": params,
        "feature_columns": cols,
        "categorical_columns": column_groups(cols)[0],
        "test_metrics": metrics,
        # The app's +/- comes from held-out error, not a made-up number (AGENTS.md section 7).
        "interval": {"mae": float(metrics["MAE"]),
                     "coverage_of_mae": float(np.mean(abs_err <= metrics["MAE"])),
                     "abs_err_q80": float(np.quantile(abs_err, 0.80))},
        "trained_on": {"month": cfg["data"]["month"], "n_games": args.n, "train_rows": len(train)},
        "versions": {"sklearn": sklearn.__version__},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = ROOT / cfg["paths"]["models_dir"] / "model.joblib"
    joblib.dump(bundle, out, compress=3)
    log.info("Saved %s (%.1f MB). Test MAE %.1f, RMSE %.1f, R2 %.3f", out, out.stat().st_size / 1e6,
             metrics["MAE"], metrics["RMSE"], metrics["R2"])


if __name__ == "__main__":
    main()
