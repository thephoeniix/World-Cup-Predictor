import pandas as pd

HOSTS_BY_YEAR = {
    2010: {"South Africa"},
    2014: {"Brazil"},
    2018: {"Russia"},
    2022: {"Qatar"},
    2026: {"Mexico", "USA", "Canada"},
}


def get_host_diff(row):
    if row["tournament"] != "FIFA World Cup":
        return 0
    
    year = int(row["date"].year)
    hosts = HOSTS_BY_YEAR.get(year, set())

    if row["home_team"] in hosts:
        return 1
    if row["away_team"] in hosts:
        return -1
    return 0


def build_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["elo_diff"] = df["elo_home_pre"] - df["elo_away_pre"]
    df["host_diff"] = df.apply(get_host_diff, axis=1)

    return df