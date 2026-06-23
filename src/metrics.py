import numpy as np
import pandas as pd

import poisson_model as pm

EPS = 1e-15


def actual_outcome(home_score, away_score):
    """0 = home win, 1 = draw, 2 = away win. NaN where the match hasn't been played yet."""
    home_score = np.asarray(home_score, dtype=float)
    away_score = np.asarray(away_score, dtype=float)

    outcome = np.where(
        home_score > away_score, 0, np.where(home_score == away_score, 1, 2)
    ).astype(float)
    outcome[np.isnan(home_score) | np.isnan(away_score)] = np.nan
    return outcome


def _onehot(outcome, n_classes=3):
    n = len(outcome)
    onehot = np.zeros((n, n_classes))
    valid = ~np.isnan(outcome)
    onehot[np.arange(n)[valid], outcome[valid].astype(int)] = 1
    return onehot, valid


def rps(probs, outcome):
    """
    Ranked Probability Score per match (lower is better). Penalizes being
    wrong by a lot (predicting home when away wins) more than being wrong
    by a little (predicting draw when away wins), because it works on
    CUMULATIVE probabilities over the ordered categories home < draw < away.
    """
    probs = np.asarray(probs, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    onehot, valid = _onehot(outcome, probs.shape[1])

    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(onehot, axis=1)

    score = ((cum_p - cum_o) ** 2).sum(axis=1) / (probs.shape[1] - 1)
    score[~valid] = np.nan
    return score


def brier(probs, outcome):
    """Multi-class Brier score per match: sum of squared errors over all 3 outcome probabilities."""
    probs = np.asarray(probs, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    onehot, valid = _onehot(outcome, probs.shape[1])

    score = ((probs - onehot) ** 2).sum(axis=1)
    score[~valid] = np.nan
    return score


def log_loss(probs, outcome):
    """Per-match log loss: -log(probability assigned to the outcome that actually happened)."""
    probs = np.asarray(probs, dtype=float)
    outcome = np.asarray(outcome, dtype=float)

    n = probs.shape[0]
    score = np.full(n, np.nan)
    valid = ~np.isnan(outcome)
    idx = np.arange(n)[valid]
    classes = outcome[valid].astype(int)
    p_correct = np.clip(probs[idx, classes], EPS, 1 - EPS)
    score[idx] = -np.log(p_correct)
    return score


def calibration_error(probs, outcome, n_bins=10):
    """
    Expected Calibration Error: bins matches by the model's confidence in
    its predicted (argmax) class, then compares confidence to actual
    accuracy within each bin. ECE close to 0 means "when the model says
    70%, it's right about 70% of the time."
    """
    probs = np.asarray(probs, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    valid = ~np.isnan(outcome)

    probs, outcome = probs[valid], outcome[valid].astype(int)
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == outcome).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidence, bins) - 1, 0, n_bins - 1)

    n = len(confidence)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        ece += mask.sum() / n * abs(correct[mask].mean() - confidence[mask].mean())

    return ece


def evaluate(df):
    """
    Score a backtest's predictions match-by-match.

    `df` needs lambda_home, lambda_away (from backtest.run_backtest) and
    home_score, away_score (the real result). Matches with no result yet
    (future 2026 fixtures) get NaN scores instead of being dropped, so the
    caller can see how many matches were actually scoreable.
    """
    df = df.copy()
    probs = pm.match_outcome_probs(df["lambda_home"], df["lambda_away"])
    outcome = actual_outcome(df["home_score"], df["away_score"])

    df["p_home"] = probs["p_home"].values
    df["p_draw"] = probs["p_draw"].values
    df["p_away"] = probs["p_away"].values
    df["outcome"] = outcome

    prob_matrix = probs[["p_home", "p_draw", "p_away"]].values
    df["rps"] = rps(prob_matrix, outcome)
    df["brier"] = brier(prob_matrix, outcome)
    df["log_loss"] = log_loss(prob_matrix, outcome)

    return df


def summarize(scored_df, group_col="wc_year"):
    """Mean RPS / Brier / Log Loss per group (e.g. per World Cup), over scoreable matches only."""
    scoreable = scored_df.dropna(subset=["outcome"])
    return scoreable.groupby(group_col)[["rps", "brier", "log_loss"]].mean()


def summarize_models(scored_df):
    """Mean RPS / Brier / Log Loss per (World Cup, model) -- for backtest.compare_models() output."""
    scoreable = scored_df.dropna(subset=["outcome"])
    return scoreable.groupby(["wc_year", "model"])[["rps", "brier", "log_loss"]].mean()


def bootstrap_paired_diff(score_a, score_b, n_boot=10000, seed=None):
    """
    Paired bootstrap on a per-match metric (e.g. RPS) to test whether
    Model A's average score is significantly different from Model B's, on
    the SAME matches. Resampling match indices (not separately for A and
    B) keeps the pairing intact -- this is the "bootstrap pareado" the
    blueprint requires before trusting any RPS difference as signal.
    """
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    valid = ~(np.isnan(score_a) | np.isnan(score_b))
    score_a, score_b = score_a[valid], score_b[valid]

    rng = np.random.default_rng(seed)
    n = len(score_a)
    diffs = score_a - score_b
    observed = diffs.mean()

    boot_idx = rng.integers(0, n, size=(n_boot, n))
    boot_diffs = diffs[boot_idx].mean(axis=1)

    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    p_value = 2 * min((boot_diffs <= 0).mean(), (boot_diffs >= 0).mean())

    return {
        "observed_diff": observed,
        "ci95": (ci_low, ci_high),
        "p_value": min(p_value, 1.0),
        "significant": not (ci_low <= 0 <= ci_high),
    }


def bootstrap_compare_models(scored_df, model_a, model_b, metric="rps", by_year=True, n_boot=10000, seed=None):
    """
    Paired bootstrap of `metric` between two named models from a
    backtest.compare_models() output, matched match-by-match on
    (wc_year, date, home_team, away_team) so the pairing survives the
    merge regardless of row order.
    """
    merge_keys = ["wc_year", "date", "home_team", "away_team"]
    a = scored_df[scored_df["model"] == model_a][merge_keys + [metric]]
    b = scored_df[scored_df["model"] == model_b][merge_keys + [metric]]
    merged = a.merge(b, on=merge_keys, suffixes=(f"_{model_a}", f"_{model_b}"))
    merged = merged.dropna(subset=[f"{metric}_{model_a}", f"{metric}_{model_b}"])

    groups = merged.groupby("wc_year") if by_year else [("all", merged)]

    results = []
    for wc_year, group in groups:
        diff = bootstrap_paired_diff(
            group[f"{metric}_{model_a}"], group[f"{metric}_{model_b}"], n_boot=n_boot, seed=seed
        )
        diff["wc_year"] = wc_year
        diff["n_matches"] = len(group)
        results.append(diff)

    return pd.DataFrame(results)


if __name__ == "__main__":
    import elo
    import features
    import backtest as bt

    df = pd.read_csv("data/processed/results_cleaned.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = elo.add_elo_features(df)
    df = features.build_features(df)

    compared = bt.compare_models(df)
    scored = evaluate(compared)

    print("=== Mean RPS / Brier / Log Loss, by World Cup and model ===")
    print(summarize_models(scored))

    for challenger, baseline, label in [
        ("elo_host", "elo_only", "elo_host vs elo_only (Elo puro)"),
        ("elo_host", "base_rate", "elo_host vs base_rate (tasa base)"),
    ]:
        print(f"\n=== {label} -- bootstrap on RPS, per World Cup ===")
        print(bootstrap_compare_models(scored, challenger, baseline))

        print(f"\n=== {label} -- bootstrap on RPS, pooled across all World Cups ===")
        print(bootstrap_compare_models(scored, challenger, baseline, by_year=False))
