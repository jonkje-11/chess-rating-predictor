"""Train and compare models, then refit the best one on the full training set and save it.

Model selection uses a validation split carved out of the training set (grouped by game), so the
test set is only used to report final numbers, never to pick a model.

Usage:
    python -m src.train                          # all models, data.n_games from config.yaml
    python -m src.train --n 20000 --models dummy,hgb
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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import ROOT, load_config
from src.dataset import TARGET, feature_columns, load_table
from src.features import CATEGORICAL_FEATURES

log = logging.getLogger(__name__)

MODEL_ORDER = ["dummy", "ridge", "random_forest", "hgb"]


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


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def split_xy(table: pd.DataFrame, cols: list[str]):
    return table[cols], table[TARGET].to_numpy()


def compare_models(table: pd.DataFrame, names: list[str], seed: int) -> tuple[pd.DataFrame, dict]:
    """Fit each model on train-minus-validation; report validation (for selection) and test metrics."""
    cols = feature_columns(table)
    cat_cols = [c for c in cols if c in CATEGORICAL_FEATURES]
    num_cols = [c for c in cols if c not in CATEGORICAL_FEATURES]
    ordered = cat_cols + num_cols

    train = table[table["split"] == "train"].reset_index(drop=True)
    test = table[table["split"] == "test"]
    fit_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
                            .split(train, groups=train["game_id"]))
    X_fit, y_fit = split_xy(train.iloc[fit_idx], ordered)
    X_val, y_val = split_xy(train.iloc[val_idx], ordered)
    X_test, y_test = split_xy(test, ordered)

    models = make_models(cat_cols, num_cols, seed)
    results, fitted = [], {}
    for name in names:
        model = models[name]
        start = time.time()
        model.fit(X_fit, y_fit)
        fit_s = time.time() - start
        row = {"model": name, "fit_seconds": round(fit_s, 1),
               "size_MB": round(len(pickle.dumps(model)) / 1e6, 2)}
        row.update({f"train_{k}": v for k, v in regression_metrics(y_fit, model.predict(X_fit)).items()})
        row.update({f"val_{k}": v for k, v in regression_metrics(y_val, model.predict(X_val)).items()})
        row.update({f"test_{k}": v for k, v in regression_metrics(y_test, model.predict(X_test)).items()})
        results.append(row)
        fitted[name] = model
        log.info("%-14s val MAE %.1f | test MAE %.1f | fit %.0fs", name, row["val_MAE"], row["test_MAE"], fit_s)
    return pd.DataFrame(results), {"models": models, "ordered": ordered, "cat_cols": cat_cols,
                                   "num_cols": num_cols}


def save_tables(df: pd.DataFrame, stem: str) -> None:
    out_dir = ROOT / "report" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(df.to_markdown(index=False, floatfmt=".3g") + "\n", encoding="utf-8")


def main() -> None:
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=cfg["data"]["n_games"])
    p.add_argument("--models", default=",".join(MODEL_ORDER), help=f"comma list from {MODEL_ORDER}")
    p.add_argument("--no-save", action="store_true", help="compare only, do not save the best model")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed = cfg["seed"]

    table = load_table(cfg, args.n)
    log.info("Loaded %d rows (%d games)", len(table), table["game_id"].nunique())
    names = [m.strip() for m in args.models.split(",")]
    results, ctx = compare_models(table, names, seed)

    pd.set_option("display.width", 200)
    print(results[["model", "val_MAE", "test_MAE", "test_RMSE", "test_R2", "train_MAE", "fit_seconds", "size_MB"]]
          .round(3).to_string(index=False))
    save_tables(results.round(4), f"model_comparison_n{args.n}")

    if args.no_save:
        return
    candidates = results[results["model"] != "dummy"]
    best = (candidates if len(candidates) else results).sort_values("val_MAE").iloc[0]["model"]
    log.info("Best model by validation MAE: %s. Refitting on the full training set.", best)

    train = table[table["split"] == "train"]
    test = table[table["split"] == "test"]
    model = make_models(ctx["cat_cols"], ctx["num_cols"], seed)[best]
    model.fit(*split_xy(train, ctx["ordered"]))
    X_test, y_test = split_xy(test, ctx["ordered"])
    pred = model.predict(X_test)
    abs_err = np.abs(y_test - pred)
    metrics = regression_metrics(y_test, pred)

    bundle = {
        "pipeline": model,
        "model_name": best,
        "feature_columns": ctx["ordered"],
        "categorical_columns": ctx["cat_cols"],
        "test_metrics": metrics,
        # The app's +/- comes from held-out error, not a made-up number (AGENTS.md section 7).
        "interval": {"mae": float(metrics["MAE"]), "abs_err_q68": float(np.quantile(abs_err, 0.68)),
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
