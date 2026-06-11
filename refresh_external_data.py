from __future__ import annotations

import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
ELO_URL = "https://eloratings.net/World.tsv"
SQUAD_URL = "https://fdp.fifa.org/assetspublic/ce281/pdf/SquadLists-English.pdf"
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/"
    "fifa.world/scoreboard?dates=20260611-20260628&limit=200"
)
FOOTYSTATS_URL = "https://footystats.org/world-cup"

TEAM_ALIASES = {
    "United States": "USA",
    "Czechia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Türkiye": "Turkey",
    "Ivory Coast": "Côte d'Ivoire",
    "Cape Verde": "Cabo Verde",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Congo DR": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
}


def download(url: str, destination: Path) -> None:
    response = requests.get(
        url,
        headers={"User-Agent": "wc2026-prediction-model/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    destination.write_bytes(response.content)


def report_csv_coverage(path: Path) -> None:
    frame = pd.read_csv(path)
    populated = int(frame.dropna(how="all").shape[0])
    print(f"{path.relative_to(ROOT)}: {populated} populated rows")


def fetch_market_odds() -> None:
    fixtures = pd.read_csv(ROOT / "comp-notebook/data/group_fixtures.csv")
    fixture_pairs = {
        (row.home_team, row.away_team): int(row.match_id)
        for row in fixtures.itertuples()
    }
    response = requests.get(
        ESPN_SCOREBOARD_URL,
        headers={"User-Agent": "wc2026-prediction-model/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    rows = []
    for event in response.json().get("events", []):
        competition = event["competitions"][0]
        competitors = {
            competitor["homeAway"]: TEAM_ALIASES.get(
                competitor["team"]["displayName"],
                competitor["team"]["displayName"],
            )
            for competitor in competition["competitors"]
        }
        key = (competitors["home"], competitors["away"])
        reversed_fixture = False
        if key not in fixture_pairs and (key[1], key[0]) in fixture_pairs:
            key = (key[1], key[0])
            reversed_fixture = True
        if key not in fixture_pairs or not competition.get("odds"):
            continue
        market = competition["odds"][0]
        moneyline = market.get("moneyline", {})
        try:
            event_home_odds = moneyline["home"]["close"]["odds"]
            draw_odds = moneyline["draw"]["close"]["odds"]
            event_away_odds = moneyline["away"]["close"]["odds"]
        except KeyError:
            continue
        home_odds = event_away_odds if reversed_fixture else event_home_odds
        away_odds = event_home_odds if reversed_fixture else event_away_odds
        total = market.get("total", {})
        rows.append(
            {
                "match_id": fixture_pairs[key],
                "home_team": key[0],
                "away_team": key[1],
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "total_line": market.get("overUnder"),
                "over_odds": total.get("over", {}).get("close", {}).get("odds"),
                "under_odds": total.get("under", {}).get("close", {}).get("odds"),
                "source": "ESPN API / DraftKings",
                "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            }
        )
    odds = pd.DataFrame(rows).sort_values("match_id")
    odds.to_csv(ROOT / "data/market_odds_2026.csv", index=False)
    print(f"Fetched market odds for {len(odds)}/72 group fixtures.")


def fetch_international_corner_stats() -> None:
    response = requests.get(
        FOOTYSTATS_URL,
        headers={"User-Agent": "Mozilla/5.0 wc2026-prediction-model/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    corner_table = next(
        table
        for table in tables
        if list(table.columns)
        == ["Country", "AVG", "7.5+", "8.5+", "9.5+", "10.5+", "11.5+", "12.5+", "13.5+"]
    )
    corner_average = {
        TEAM_ALIASES.get(row.Country, row.Country): float(row.AVG)
        for row in corner_table.itertuples()
    }
    fixtures = pd.read_csv(ROOT / "comp-notebook/data/group_fixtures.csv")
    rows = []
    for row in fixtures.itertuples():
        home_average = corner_average.get(row.home_team)
        away_average = corner_average.get(row.away_team)
        if home_average is None or away_average is None:
            continue
        expected_total = (home_average + away_average) / 2.0
        rows.append(
            {
                "match_id": row.match_id,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_corners": expected_total / 2.0,
                "away_corners": expected_total / 2.0,
                "home_yellow": None,
                "away_yellow": None,
                "home_red": None,
                "away_red": None,
                "source": "FootyStats World Cup last-10 national team averages",
                "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            }
        )
    stats = pd.DataFrame(rows).sort_values("match_id")
    stats.to_csv(ROOT / "data/international_match_stats_2026.csv", index=False)
    print(f"Fetched international corner estimates for {len(stats)}/72 fixtures.")


def main() -> None:
    download(ELO_URL, ROOT / "data/world_elo_2026-06-10.tsv")
    squad_pdf = ROOT / "data/worldcup/squads_2026.pdf"
    squad_text = ROOT / "data/worldcup/squads_2026.txt"
    download(SQUAD_URL, squad_pdf)
    subprocess.run(
        ["pdftotext", "-layout", str(squad_pdf), str(squad_text)],
        check=True,
    )
    fetch_market_odds()
    fetch_international_corner_stats()

    print("Refreshed World Football Elo and the official FIFA squad list.")
    for relative_path in (
        "data/market_odds_2026.csv",
        "data/international_match_stats_2026.csv",
        "data/injuries_2026.csv",
        "data/referee_assignments_2026.csv",
    ):
        report_csv_coverage(ROOT / relative_path)
    print("Run `python build_submission.py` to retrain and regenerate the notebook.")


if __name__ == "__main__":
    main()
