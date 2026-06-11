from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "data/club_corners_cards_2016_2025.csv"
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

SEASONS = (
    "1617",
    "1718",
    "1819",
    "1920",
    "2021",
    "2122",
    "2223",
    "2324",
    "2425",
)

LEAGUES = (
    "E0",
    "E1",
    "D1",
    "D2",
    "I1",
    "I2",
    "SP1",
    "SP2",
    "F1",
    "F2",
    "N1",
    "B1",
    "P1",
    "T1",
    "G1",
    "SC0",
)

REQUIRED_STAT_COLUMNS = {"HC", "AC", "HY", "AY"}
OUTPUT_COLUMNS = (
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
)


def download_competition(
    session: requests.Session,
    season: str,
    league: str,
) -> pd.DataFrame | None:
    url = BASE_URL.format(season=season, league=league)
    response = session.get(url, timeout=15)
    if response.status_code != requests.codes.ok:
        print(f"SKIP  {league} {season}: HTTP {response.status_code}")
        return None

    frame = pd.read_csv(StringIO(response.text), on_bad_lines="skip")
    if not REQUIRED_STAT_COLUMNS.issubset(frame.columns):
        print(f"SKIP  {league} {season}: missing corner/card columns")
        return None

    available_columns = [
        column for column in OUTPUT_COLUMNS if column in frame.columns
    ]
    frame = frame[available_columns].copy()
    stat_columns = [
        column
        for column in ("HC", "AC", "HY", "AY", "HR", "AR")
        if column in frame.columns
    ]
    frame = frame.dropna(subset=stat_columns, how="all")
    frame["Season"] = season
    frame["League"] = league
    print(f"OK    {league} {season}: {len(frame)} rows")
    return frame


def main() -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "wc2026-prediction-model/1.0"

    frames = [
        frame
        for season in SEASONS
        for league in LEAGUES
        if (frame := download_competition(session, season, league)) is not None
    ]
    if not frames:
        raise RuntimeError("No club corner/card data was downloaded")

    club_data = pd.concat(frames, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    club_data.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(club_data):,} rows to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
