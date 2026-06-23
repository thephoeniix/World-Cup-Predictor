import pandas as pd

import elo
import features
import poisson_model as pm

# The World Cups we evaluate, per the project blueprint.
BACKTEST_YEARS = [2010, 2014, 2018, 2022, 2026]

# Columns that get z-scored before going into the Poisson model.
STANDARDIZE_COLS = ["elo_diff", "host_diff"]

# Named model variants, all sharing the same Poisson engine (only the
# formula changes) -- this is the V1 floor the blueprint asks for:
# "tasa base" (no team-strength signal at all) and "Elo puro" (Elo only,
# no host advantage) are what the real model has to beat.
FORMULAS = {
    "base_rate": "goals ~ is_home",
    "elo_only": "goals ~ is_home + elo_diff",
    "elo_host": "goals ~ is_home + elo_diff + host_diff",
}


def get_train_test(df, wc_year):
    """
    Split into "everything before this World Cup" (train) and "this World
    Cup's matches" (test).

    Why this matters: train is NOT "previous World Cups only" — it's every
    match (friendlies, qualifiers, other tournaments) before the cutoff
    date. That's what feeds the Elo history and the Poisson fit. Nothing
    dated on/after the cutoff is allowed into train, otherwise the model
    would be "predicting" a tournament it already partly saw.
    """
    wc_matches = df[
        (df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == wc_year)
    ]
    cutoff = wc_matches["date"].min()

    train = df[df["date"] < cutoff].copy()
    test = wc_matches.copy()

    return train, test


def standardize(train, test, cols=STANDARDIZE_COLS):
    """
    Z-score `cols` using mean/std computed from TRAIN ONLY, then apply
    those same numbers to test.

    Why: if we fit mean/std on train+test (or on test alone), information
    from the tournament we're trying to predict leaks into the scaling —
    the backtest would look better than the model actually is. Computing
    the stats from train and reusing them on test is what keeps this a
    fair, "as if we were there before the tournament" evaluation.
    """
    train = train.copy()
    test = test.copy()

    mean = train[cols].mean()
    std = train[cols].std().replace(0, 1)  # constant column (e.g. no host data yet for 2010) -> leave as-is, don't divide by zero

    train[cols] = (train[cols] - mean) / std
    test[cols] = (test[cols] - mean) / std

    return train, test


def run_backtest(df, wc_year, formula=FORMULAS["elo_host"]):
    """
    Train the shared Poisson engine on everything before `wc_year`, then
    predict that World Cup's matches.

    `formula` picks the model variant (see FORMULAS) -- everything else
    (the train/test split, the standardization) is identical, so any
    difference in the resulting predictions is attributable to the
    feature list, not to the evaluation setup.

    Returns the test rows with two extra columns, lambda_home/lambda_away
    (the model's predicted expected goals for each side) — metrics.py will
    turn those into win/draw/loss probabilities and score them (RPS, etc.)
    """
    train, test = get_train_test(df, wc_year)
    train, test = standardize(train, test)

    long_train = pm.to_long(train)
    model = pm.fit_poisson(long_train, formula=formula)

    lambda_home, lambda_away = pm.predict_lambdas(model, test)

    test = test.copy()
    test["lambda_home"] = lambda_home.values
    test["lambda_away"] = lambda_away.values

    return test, model


def run_all_backtests(df, years=BACKTEST_YEARS, formula=FORMULAS["elo_host"]):
    """Run run_backtest for every year in `years`, stacked into one table."""
    all_results = []

    for year in years:
        test, _ = run_backtest(df, year, formula=formula)
        test["wc_year"] = year
        all_results.append(test)

    return pd.concat(all_results, ignore_index=True)


def compare_models(df, formulas=FORMULAS, years=BACKTEST_YEARS):
    """Run every named model variant over the same years, tagged with a `model` column."""
    all_results = []

    for name, formula in formulas.items():
        results = run_all_backtests(df, years=years, formula=formula)
        results["model"] = name
        all_results.append(results)

    return pd.concat(all_results, ignore_index=True)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/results_cleaned.csv")
    df["date"] = pd.to_datetime(df["date"])

    df = elo.add_elo_features(df)
    df = features.build_features(df)

    results = run_all_backtests(df)

    print(results.groupby("wc_year")[["lambda_home", "lambda_away"]].mean())
