import pandas as pd

def load_raw():
    """Load the raw data from the CSV file."""
    df = pd.read_csv("data/raw/international_results/results.csv")
    return df

def filter_a_team_matches(df):
    """Filter Team B/sub23/friendly matches is datasets includes them"""
    patron = r'\b(?:Olympic|U-?23)\b|\sB$'
    mask = ~(df['home_team'].str.contains(patron, na=False) | df['away_team'].str.contains(patron, na=False))
    return df[mask]

"""TABLE ALIASES"""

ALIASES = {
    # América
    "United States": "USA",
    "United States of America": "USA",
    "Mexico": "Mexico",
    "Costa Rica": "Costa Rica",
    "Panama": "Panama",
    "Canada": "Canada",

    # Europa
    "England": "England",
    "Scotland": "Scotland",
    "Wales": "Wales",
    "Northern Ireland": "Northern Ireland",
    "Czech Republic": "Czechia",
    "Republic of Ireland": "Ireland",
    "Russia": "Russia",

    # África
    "Ivory Coast": "Côte d'Ivoire",
    "DR Congo": "Congo DR",
    "Congo": "Congo",
    "Cape Verde": "Cape Verde",
    "South Africa": "South Africa",

    # Asia
    "South Korea": "Korea Republic",
    "Korea, South": "Korea Republic",
    "North Korea": "Korea DPR",
    "Iran": "IR Iran",
    "Saudi Arabia": "Saudi Arabia",
    "United Arab Emirates": "UAE",

    # Oceanía
    "New Zealand": "New Zealand",

    # Balcanes
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Serbia and Montenegro": "Serbia and Montenegro",

    # Otros
    "Curacao": "Curaçao",
}

def normalize_team_names(df,col):
    """Normalize team names using the ALIASES dictionary."""
    df[col] = df[col].replace(ALIASES)
    return df

""""RELEVANT COMPETITIONS"""

def clean(df):
    df = filter_a_team_matches(df)
    df = normalize_team_names(df, 'home_team')
    df = normalize_team_names(df, 'away_team')
    df = df.sort_values("date").reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = load_raw()
    df_cleaned = clean(df)
    df_cleaned.to_csv("data/processed/results_cleaned.csv", index=False)
    print(df_cleaned.shape, df_cleaned["date"].min(), df_cleaned["date"].max())