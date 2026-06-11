from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import poisson


BIG_FIVE_COUNTRIES = {"ENG", "ESP", "GER", "ITA", "FRA"}
SECOND_TIER_COUNTRIES = {
    "NED",
    "POR",
    "BEL",
    "TUR",
    "BRA",
    "ARG",
    "USA",
    "KSA",
}


def american_to_decimal(value: float) -> float:
    return 1.0 + (100.0 / abs(value) if value < 0 else value / 100.0)


def normalized_market_probabilities(
    home_odds: float, draw_odds: float, away_odds: float
) -> tuple[float, float, float]:
    decimal = []
    for value in (home_odds, draw_odds, away_odds):
        decimal.append(
            american_to_decimal(value)
            if value <= 0 or value >= 20
            else float(value)
        )
    inverse = 1.0 / np.asarray(decimal)
    normalized = inverse / inverse.sum()
    return tuple(float(value) for value in normalized)


@dataclass
class MarketOdds:
    probabilities: dict[tuple[str, str], tuple[float, float, float]]
    expected_total_goals: dict[tuple[str, str], float]

    @classmethod
    def load(cls, path: Path, canonical_function) -> "MarketOdds":
        if not path.exists():
            return cls({}, {})
        odds = pd.read_csv(path)
        required = {
            "home_team",
            "away_team",
            "home_odds",
            "draw_odds",
            "away_odds",
        }
        missing = required - set(odds.columns)
        if missing:
            raise ValueError(f"Market odds file is missing columns: {sorted(missing)}")
        probabilities = {}
        expected_totals = {}
        for row in odds.itertuples():
            if pd.isna(row.home_odds) or pd.isna(row.draw_odds) or pd.isna(row.away_odds):
                continue
            key = (
                canonical_function(row.home_team),
                canonical_function(row.away_team),
            )
            probabilities[key] = normalized_market_probabilities(
                float(row.home_odds),
                float(row.draw_odds),
                float(row.away_odds),
            )
            if all(
                column in odds.columns
                for column in ("total_line", "over_odds", "under_odds")
            ):
                total_values = (
                    row.total_line,
                    row.over_odds,
                    row.under_odds,
                )
                if not any(pd.isna(value) for value in total_values):
                    over_decimal = (
                        american_to_decimal(float(row.over_odds))
                        if float(row.over_odds) <= 0 or float(row.over_odds) >= 20
                        else float(row.over_odds)
                    )
                    under_decimal = (
                        american_to_decimal(float(row.under_odds))
                        if float(row.under_odds) <= 0 or float(row.under_odds) >= 20
                        else float(row.under_odds)
                    )
                    inverse = np.array([1 / over_decimal, 1 / under_decimal])
                    over_probability = float(inverse[0] / inverse.sum())
                    threshold = math.floor(float(row.total_line))

                    def equation(rate: float) -> float:
                        return 1.0 - poisson.cdf(threshold, rate) - over_probability

                    expected_totals[key] = float(brentq(equation, 0.05, 8.0))
        return cls(probabilities, expected_totals)

    def get(self, home_team: str, away_team: str):
        return self.probabilities.get((home_team, away_team))


def parse_official_squads(text_path: Path, canonical_function) -> pd.DataFrame:
    text = text_path.read_text(encoding="utf-8", errors="replace")
    page_pattern = re.compile(
        r"SQUAD LIST\s*\n.*?\n\s*([A-Za-zÀ-ÿ' .-]+) \([A-Z]{3}\)(.*?)(?=\f|\Z)",
        re.S,
    )
    player_pattern = re.compile(
        r"^\s*(\d{1,2})\s+(GK|DF|MF|FW)\s+(.+?)\s+(\d{2}/\d{2}/\d{4})\s+(.+?)\s+\(([A-Z]{3})\)\s+(\d{3})\s*$"
    )
    rows = []
    for page_match in page_pattern.finditer(text):
        team = canonical_function(page_match.group(1).strip())
        for line in page_match.group(2).splitlines():
            player_match = player_pattern.match(line)
            if not player_match:
                continue
            number, position, player, birth_date, club, club_country, height = (
                player_match.groups()
            )
            rows.append(
                {
                    "team": team,
                    "number": int(number),
                    "position": position,
                    "player": " ".join(player.split()),
                    "birth_date": pd.to_datetime(birth_date, format="%d/%m/%Y"),
                    "club": " ".join(club.split()),
                    "club_country": club_country,
                    "height_cm": int(height),
                }
            )
    squad = pd.DataFrame(rows)
    if squad.empty:
        raise RuntimeError("No squad players were parsed from the official FIFA PDF")
    return squad


def build_squad_strength(
    squad: pd.DataFrame,
    injury_path: Path,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    squad = squad.copy()
    squad["age"] = (reference_date - squad["birth_date"]).dt.days / 365.25
    squad["club_tier"] = np.select(
        [
            squad["club_country"].isin(BIG_FIVE_COUNTRIES),
            squad["club_country"].isin(SECOND_TIER_COUNTRIES),
        ],
        [2.0, 1.0],
        default=0.0,
    )
    summary = squad.groupby("team").agg(
        squad_size=("player", "size"),
        average_age=("age", "mean"),
        big_five_share=("club_country", lambda values: values.isin(BIG_FIVE_COUNTRIES).mean()),
        average_club_tier=("club_tier", "mean"),
        goalkeeper_height=("height_cm", lambda values: values.mean()),
        forward_count=("position", lambda values: int((values == "FW").sum())),
    )

    summary["injury_penalty"] = 0.0
    if injury_path.exists():
        injuries = pd.read_csv(injury_path)
        required = {"team", "player", "status", "impact"}
        missing = required - set(injuries.columns)
        if missing:
            raise ValueError(f"Injury file is missing columns: {sorted(missing)}")
        active = injuries[injuries["status"].str.lower().isin({"out", "doubtful"})]
        penalties = active.groupby("team")["impact"].sum()
        summary["injury_penalty"] = summary.index.to_series().map(penalties).fillna(0)

    tier_z = (summary["average_club_tier"] - summary["average_club_tier"].mean()) / (
        summary["average_club_tier"].std() or 1
    )
    big_five_z = (summary["big_five_share"] - summary["big_five_share"].mean()) / (
        summary["big_five_share"].std() or 1
    )
    age_penalty = abs(summary["average_age"] - 27.5) * 1.5
    summary["elo_adjustment"] = (
        18.0 * tier_z + 12.0 * big_five_z - age_penalty - summary["injury_penalty"]
    ).clip(-55, 55)
    return summary.reset_index()


def load_team_match_overrides(path: Path, canonical_function) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    overrides = pd.read_csv(path)
    for column in ("home_team", "away_team"):
        if column in overrides:
            overrides[column] = overrides[column].map(canonical_function)
    return overrides
