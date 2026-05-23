import pandas as pd
import requests
from io import StringIO

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

# Valid seasons only — one per year, format XXYY
SEASONS = [
    "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2425",
]

LEAGUES = [
    "E0", "E1",       # England Premier, Championship
    "D1", "D2",       # Germany Bundesliga 1 & 2
    "I1", "I2",       # Italy Serie A & B
    "SP1", "SP2",     # Spain La Liga & Segunda
    "F1", "F2",       # France Ligue 1 & 2
    "N1",             # Netherlands Eredivisie
    "B1",             # Belgium Pro League
    "P1",             # Portugal Primeira Liga
    "T1",             # Turkey Süper Lig
    "G1",             # Greece Super League
    "SC0",            # Scotland Premiership
]

# Only keep columns relevant to corners and cards
KEEP_COLS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG",           # full-time goals (for context)
    "HC", "AC",               # corners home/away
    "HY", "AY",               # yellow cards home/away
    "HR", "AR",               # red cards home/away
]

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

all_data = []

for season in SEASONS:
    for league in LEAGUES:
        url = BASE_URL.format(season=season, league=league)
        try:
            r = requests.get(url, timeout=15, headers=headers)
            if r.status_code != 200:
                print(f"SKIP  {league} {season} — HTTP {r.status_code}")
                continue

            df = pd.read_csv(StringIO(r.text), on_bad_lines="skip")

            # Keep only columns that exist in this file
            cols = [c for c in KEEP_COLS if c in df.columns]
            if not {"HC", "AC", "HY", "AY"}.issubset(df.columns):
                print(f"SKIP  {league} {season} — missing corner/card columns")
                continue

            df = df[cols].copy()
            df["Season"] = season
            df["League"] = league

            # Drop rows where all stat columns are NaN
            stat_cols = [c for c in ["HC", "AC", "HY", "AY", "HR", "AR"] if c in df.columns]
            df = df.dropna(subset=stat_cols, how="all")

            all_data.append(df)
            print(f"OK    {league} {season} — {len(df)} rows")

        except Exception as e:
            print(f"ERROR {league} {season} — {e}")

if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("club_corners_cards_2016_2025.csv", index=False)
    print(f"\nSaved: club_corners_cards_2016_2025.csv ({len(final_df):,} rows)")
else:
    print("No data downloaded.")