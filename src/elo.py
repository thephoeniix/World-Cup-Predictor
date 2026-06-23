import pandas as pd

INITIAL_ELO = 1500
HOME_ADVANTAGE = 100
K = 30


def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def actual_score(home_score, away_score):
    if home_score > away_score:
        return 1.0
    if home_score == away_score:
        return 0.5
    return 0.0


def add_elo_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    ratings = {}
    elo_home_pre = []
    elo_away_pre = []

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_elo = ratings.get(home_team, INITIAL_ELO)
        away_elo = ratings.get(away_team, INITIAL_ELO)

        elo_home_pre.append(home_elo)
        elo_away_pre.append(away_elo)

        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            continue

        home_advantage = 0 if row.get("neutral", False) else HOME_ADVANTAGE

        expected_home = expected_score(
            home_elo + home_advantage,
            away_elo
        )

        actual_home = actual_score(
            row["home_score"],
            row["away_score"]
        )

        delta_home = K * (actual_home - expected_home)

        ratings[home_team] = home_elo + delta_home
        ratings[away_team] = away_elo - delta_home

    df["elo_home_pre"] = elo_home_pre
    df["elo_away_pre"] = elo_away_pre
    df["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]

    return df