import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson


def to_long(df):
    """
    Reshape one row-per-match into two rows-per-match (attacker/defender).

    Why: we want ONE Poisson model that works for both home and away goals,
    instead of training two separate models. Each match becomes:
      - one row from the home team's point of view (is_home=1)
      - one row from the away team's point of view (is_home=0)
    The away row flips the sign of elo_diff/host_diff because those columns
    are defined as "home minus away" — from the away team's perspective it's
    the mirror image ("me minus my opponent").
    """
    home = pd.DataFrame({
        "team": df["home_team"],
        "opponent": df["away_team"],
        "goals": df["home_score"],
        "is_home": 1,
        "elo_diff": df["elo_diff"],
        "host_diff": df["host_diff"],
    })

    away = pd.DataFrame({
        "team": df["away_team"],
        "opponent": df["home_team"],
        "goals": df["away_score"],
        "is_home": 0,
        "elo_diff": -df["elo_diff"],
        "host_diff": -df["host_diff"],
    })

    long_df = pd.concat([home, away], ignore_index=True)

    # Matches with no score yet (2026 fixtures) have nothing to train on.
    long_df = long_df.dropna(subset=["goals"])

    return long_df


def fit_poisson(long_df, formula="goals ~ is_home + elo_diff + host_diff"):
    """
    Fit the shared Poisson GLM: log(lambda) = b0 + b_home*is_home + sum(bi * feature_i).

    `formula` is the only thing that should change between Model A and Model B
    (e.g. adding "+ gdp_diff + population_diff" later) — the engine itself
    (this function) stays the same for both.
    """
    model = smf.glm(
        formula=formula,
        data=long_df,
        family=sm.families.Poisson(),
    ).fit()

    return model


def predict_lambdas(model, df):
    """
    Given match-level (wide) rows with elo_diff/host_diff, return the
    predicted expected goals (lambda) for the home team and the away team.

    We build the same two perspectives used in to_long() — one row per
    side, is_home flipped and diffs sign-flipped for the away side — and
    ask the fitted model to score each one.
    """
    home_view = pd.DataFrame({
        "is_home": 1,
        "elo_diff": df["elo_diff"],
        "host_diff": df["host_diff"],
    })
    away_view = pd.DataFrame({
        "is_home": 0,
        "elo_diff": -df["elo_diff"],
        "host_diff": -df["host_diff"],
    })

    lambda_home = model.predict(home_view)
    lambda_away = model.predict(away_view)

    return lambda_home, lambda_away


def match_outcome_probs(lambda_home, lambda_away, max_goals=10):
    """
    Turn each match's (lambda_home, lambda_away) into P(home win), P(draw),
    P(away win), by treating home/away goals as independent Poisson draws
    and summing the joint grid over the three regions h>a, h==a, h<a.

    `max_goals` just bounds the grid we sum over (10 already covers
    ~99.99% of the Poisson mass for any realistic lambda in football).
    """
    lambda_home = np.asarray(lambda_home, dtype=float)
    lambda_away = np.asarray(lambda_away, dtype=float)
    goals = np.arange(max_goals + 1)

    # (n_matches, max_goals+1) pmf tables for each side.
    pmf_home = poisson.pmf(goals[None, :], lambda_home[:, None])
    pmf_away = poisson.pmf(goals[None, :], lambda_away[:, None])

    # Joint pmf grid per match: (n_matches, n_home_goals, n_away_goals).
    joint = pmf_home[:, :, None] * pmf_away[:, None, :]

    home_win = np.triu(np.ones((goals.size, goals.size)), k=1).T
    draw = np.eye(goals.size)
    away_win = np.triu(np.ones((goals.size, goals.size)), k=1)

    p_home = (joint * home_win).sum(axis=(1, 2))
    p_draw = (joint * draw).sum(axis=(1, 2))
    p_away = (joint * away_win).sum(axis=(1, 2))

    return pd.DataFrame({"p_home": p_home, "p_draw": p_draw, "p_away": p_away})
