"""Train and compare models on a validation set (the test set is not used here)."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import joblib
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from src.config import load_config, players_table_file

CONFIG = load_config()
SEED = CONFIG["seed"]
TABLES_DIR = Path("docs/report/tables")
MODEL_FILE = Path("models/model.joblib")


# Columns that are not inputs to the model: IDs, the answer, and the split label.
NOT_FEATURES = ["game_id", "player_id", "rating", "split"]
CATEGORICAL = ["eco"]


def load_training_data():
    """Training rows only, split again (by game) into a fit part and a validation part."""
    table = pd.read_parquet(players_table_file(CONFIG))
    train = table[table["split"] == "train"].reset_index(drop=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=CONFIG["split"]["validation_share"], random_state=SEED)
    fit_index, val_index = next(splitter.split(train, groups=train["game_id"]))
    return train.iloc[fit_index], train.iloc[val_index]


def feature_columns(table):
    return [column for column in table.columns if column not in NOT_FEATURES]


def scores(true, predicted):
    """MAE (main metric), RMSE and R², all on the same predictions."""
    return {
        "MAE": mean_absolute_error(true, predicted),
        "RMSE": np.sqrt(mean_squared_error(true, predicted)),
        "R2": r2_score(true, predicted),
    }


def time_control_baseline(fit, val):
    """Non-ML rule: guess the average rating of players at that time control (e.g. 3+0)."""
    means = fit.groupby(["base_time", "increment"])["rating"].mean()
    keys = pd.MultiIndex.from_frame(val[["base_time", "increment"]])
    return means.reindex(keys).fillna(fit["rating"].mean()).to_numpy()


def make_models(numeric):
    """The three ML models, each with its own preprocessing in one pipeline."""
    ridge = make_pipeline(
        ColumnTransformer([
            ("numbers", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
            ("opening", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
        ]),
        Ridge(alpha=1.0),
    )
    forest = make_pipeline(
        ColumnTransformer([
            ("numbers", SimpleImputer(strategy="median"), numeric),
            ("opening", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
        ]),
        RandomForestRegressor(n_estimators=100, min_samples_leaf=20, n_jobs=-1, random_state=SEED),
    )
    boosting = make_pipeline(
        ColumnTransformer([
            ("opening", OrdinalEncoder(max_categories=250, handle_unknown="use_encoded_value",
                                       unknown_value=-1, encoded_missing_value=-1), CATEGORICAL),
            ("numbers", "passthrough", numeric),
        ]),
        HistGradientBoostingRegressor(max_iter=300, categorical_features=[0], early_stopping=False,
                                      random_state=SEED),
    )
    return {"Ridge regression": ridge, "Random forest": forest, "Gradient boosting": boosting}

def tune_boosting(fit, features, numeric):
    """Random search over gradient boosting settings, with cross-validation grouped by game."""
    step = "histgradientboostingregressor__"  # name of the model step inside the pipeline
    search = RandomizedSearchCV(
        make_models(numeric)["Gradient boosting"],
        param_distributions={
            step + "learning_rate": loguniform(0.03, 0.3),
            step + "max_iter": randint(200, 800),
            step + "max_leaf_nodes": randint(15, 128),
            step + "min_samples_leaf": randint(20, 300),
            step + "l2_regularization": uniform(0, 1),
        },
        n_iter=CONFIG["tuning"]["n_iter"],
        cv=GroupKFold(n_splits=CONFIG["tuning"]["cv_folds"]),
        scoring="neg_mean_absolute_error",
        random_state=SEED,
    )
    search.fit(fit[features], fit["rating"], groups=fit["game_id"])
    return search


def main():
    fit, val = load_training_data()
    features = feature_columns(fit)
    numeric = [column for column in features if column not in CATEGORICAL]
    print("Fit rows:", len(fit), "Validation rows:", len(val), "Features:", len(features))

    results = []
    dummy = DummyRegressor(strategy="mean").fit(fit[features], fit["rating"])
    results.append({"model": "Lazy guesser (mean)", **scores(val["rating"], dummy.predict(val[features]))})
    results.append({"model": "Mean per time control", **scores(val["rating"], time_control_baseline(fit, val))})

    for name, model in make_models(numeric).items():
        start = time.time()
        model.fit(fit[features], fit["rating"])
        result = scores(val["rating"], model.predict(val[features]))
        seconds = round(time.time() - start)
        results.append({"model": name, **result, "seconds": seconds})
        print(f"{name}: MAE {result['MAE']:.0f} ({seconds} s)")

        print("Tuning gradient boosting (this takes a few minutes)...")
    search = tune_boosting(fit, features, numeric)
    tuned = scores(val["rating"], search.best_estimator_.predict(val[features]))
    print("Best settings:", search.best_params_)
    print(f"Gradient boosting, tuned: MAE {tuned['MAE']:.0f}")
    results.append({"model": "Gradient boosting, tuned", **tuned})

    table = pd.DataFrame(results)
    print(table.round(3).to_string(index=False))
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "model_comparison_validation.csv", index=False)
    pd.DataFrame(search.cv_results_).to_csv(TABLES_DIR / "tuning_results.csv", index=False)

    # Pick tuned or untuned by validation MAE, then train it on ALL training rows.
    untuned_mae = table.loc[table["model"] == "Gradient boosting", "MAE"].iloc[0]
    if tuned["MAE"] < untuned_mae:
        chosen, chosen_name = search.best_estimator_, "Gradient boosting, tuned"
    else:
        chosen, chosen_name = make_models(numeric)["Gradient boosting"], "Gradient boosting"
    all_train = pd.concat([fit, val])
    final_model = clone(chosen).fit(all_train[features], all_train["rating"])

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "features": features, "name": chosen_name}, MODEL_FILE)
    print("Saved", chosen_name, "to", MODEL_FILE)



if __name__ == "__main__":
    main()
