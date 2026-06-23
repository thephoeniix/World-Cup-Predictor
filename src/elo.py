import pandas as pd

INITIAL_ELO = 1500
HOME_ADVANTAGE = 100
K = 30 

def expected_score(elo_a, elo_b):
    """Calculate the expected score for team A against team B"""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def actual_score(home_score, away_score):
    """Calculate the actual score for team A against team B"""
    if home_score > away_score:
        return 1  # Team A wins
    elif home_score == away_score:
        return 0.5  # Draw
    else:
        return 0  # Team A loses
    
def add_elo_features(df):
    """Add ELO features to the DataFrame"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    rating = {}
    elo_home_prev = []
    elo_away_prev = []
    
    for _, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        
        home_elo = rating.get(home_team, INITIIAL_ELO)
        away_elo = rating.get(away_team, INITIIAL_ELO)
        
        elo_home_prev.append(home_elo)
        elo_away_prev.append(away_elo)
        
        if pd.isna(row['home_score']) or pd.isna(row['away_score']):
            continue  # Skip if scores are missing
        
        home_advantage = 0 if row.get('neutral', False) else HOME_ADVANTAGE
        
        expected_home = expected_score(home_elo + home_advantage, away_elo)
        
        actual_home = actual_score(row['home_score'], row['away_score'])
        
        delta_home = K * (actual_home - expected_home)
        
        rating[home_team] = home_elo + delta_home
        rating[away_team] = away_elo - delta_home
        
    df['elo_home_prev'] = elo_home_prev
    df['elo_away_prev'] = elo_away_prev
    
    return df