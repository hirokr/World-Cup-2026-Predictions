from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ensemble_models import MatchEnsemble, backtest_goal_ensemble

ROOT = Path(__file__).resolve().parent
COMP_DIR = ROOT / "comp-notebook"
REFERENCE_DATE = pd.Timestamp("2026-06-10")
RNG_SEED = 20260610
ELO_URL = "https://eloratings.net/World.tsv"

QUALIFIED_TEAMS = {
    "UEFA Playoff A": "Bosnia and Herzegovina",
    "UEFA Playoff B": "Sweden",
    "UEFA Playoff C": "Turkey",
    "UEFA Playoff D": "Czech Republic",
    "FIFA Playoff 1": "DR Congo",
    "FIFA Playoff 2": "Iraq",
}

# Competition display names -> names used by the historical results dataset.
MODEL_NAME = {
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
}

ELO_CODE = {
    "Mexico": "MX",
    "South Africa": "ZA",
    "South Korea": "KR",
    "Czech Republic": "CZ",
    "Canada": "CA",
    "Bosnia and Herzegovina": "BA",
    "Qatar": "QA",
    "Switzerland": "CH",
    "Brazil": "BR",
    "Morocco": "MA",
    "Haiti": "HT",
    "Scotland": "SQ",
    "United States": "US",
    "Paraguay": "PY",
    "Australia": "AU",
    "Turkey": "TR",
    "Germany": "DE",
    "Curaçao": "CW",
    "Ivory Coast": "CI",
    "Ecuador": "EC",
    "Netherlands": "NL",
    "Japan": "JP",
    "Sweden": "SE",
    "Tunisia": "TN",
    "Belgium": "BE",
    "Egypt": "EG",
    "Iran": "IR",
    "New Zealand": "NZ",
    "Spain": "ES",
    "Cape Verde": "CV",
    "Saudi Arabia": "SA",
    "Uruguay": "UY",
    "France": "FR",
    "Senegal": "SN",
    "Iraq": "IQ",
    "Norway": "NO",
    "Argentina": "AR",
    "Algeria": "DZ",
    "Austria": "AT",
    "Jordan": "JO",
    "Portugal": "PT",
    "DR Congo": "CD",
    "Uzbekistan": "UZ",
    "Colombia": "CO",
    "England": "EN",
    "Croatia": "HR",
    "Ghana": "GH",
    "Panama": "PA",
}

TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 3.0,
    "UEFA Euro": 2.5,
    "Copa América": 2.5,
    "African Cup of Nations": 2.2,
    "AFC Asian Cup": 2.0,
    "Gold Cup": 1.8,
    "FIFA World Cup qualification": 1.5,
    "UEFA Euro qualification": 1.3,
    "CONCACAF Nations League": 1.2,
    "UEFA Nations League": 1.2,
    "Friendly": 0.55,
}


def canonical(team: str) -> str:
    return MODEL_NAME.get(team, team)


def tournament_weight(name: str) -> float:
    for key, value in TOURNAMENT_WEIGHT.items():
        if key in str(name):
            return value
    return 1.0


def load_results() -> pd.DataFrame:
    results = pd.read_csv(ROOT / "data/results.csv", parse_dates=["date"])
    results = results.dropna(subset=["home_score", "away_score"]).copy()
    results = results[results["date"] <= REFERENCE_DATE]
    results = results[results["date"] >= "2006-01-01"]
    results["home_team"] = results["home_team"].map(canonical)
    results["away_team"] = results["away_team"].map(canonical)
    return results


def load_fixtures() -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = pd.read_csv(COMP_DIR / "data/group_fixtures.csv")
    knockout = pd.read_csv(COMP_DIR / "data/knockout_slots.csv")
    groups[["home_team", "away_team"]] = groups[["home_team", "away_team"]].replace(
        QUALIFIED_TEAMS
    )
    return groups, knockout


def load_current_elo() -> dict[str, float]:
    import requests

    snapshot = ROOT / "data/world_elo_2026-06-10.tsv"
    if not snapshot.exists():
        response = requests.get(ELO_URL, timeout=20)
        response.raise_for_status()
        snapshot.write_bytes(response.content)
    code_rating = {}
    for line in snapshot.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 4:
            code_rating[fields[2]] = float(fields[3])
    ratings = {
        team: code_rating[code]
        for team, code in ELO_CODE.items()
        if code in code_rating
    }
    missing = set(ELO_CODE) - set(ratings)
    if missing:
        raise RuntimeError(f"Missing current Elo ratings for: {sorted(missing)}")
    return ratings


def long_goal_rows(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "team": matches["home_team"],
            "opponent": matches["away_team"],
            "venue": np.where(matches["neutral"], 0.0, 1.0),
            "goals": matches["home_score"].astype(float),
            "date": matches["date"],
            "tournament": matches["tournament"],
        }
    )
    away = pd.DataFrame(
        {
            "team": matches["away_team"],
            "opponent": matches["home_team"],
            "venue": np.where(matches["neutral"], 0.0, -1.0),
            "goals": matches["away_score"].astype(float),
            "date": matches["date"],
            "tournament": matches["tournament"],
        }
    )
    return pd.concat([home, away], ignore_index=True)


@dataclass
class GoalModel:
    alpha: float
    half_life_years: float
    pipeline: object | None = None
    fallback_goals: float = 1.25
    elo: dict[str, float] | None = None
    current_elo: dict[str, float] | None = None

    @staticmethod
    def _fit_elo(matches: pd.DataFrame) -> dict[str, float]:
        ratings: dict[str, float] = {}
        for row in matches.sort_values("date").itertuples():
            home = canonical(row.home_team)
            away = canonical(row.away_team)
            home_rating = ratings.get(home, 1500.0)
            away_rating = ratings.get(away, 1500.0)
            home_advantage = 0.0 if row.neutral else 65.0
            expected = 1.0 / (
                1.0 + 10 ** (-(home_rating + home_advantage - away_rating) / 400.0)
            )
            if row.home_score > row.away_score:
                actual = 1.0
            elif row.home_score < row.away_score:
                actual = 0.0
            else:
                actual = 0.5
            margin = abs(float(row.home_score) - float(row.away_score))
            margin_multiplier = 1.0 if margin <= 1 else math.log1p(margin)
            k = 18.0 * min(tournament_weight(row.tournament), 2.5)
            change = k * margin_multiplier * (actual - expected)
            ratings[home] = home_rating + change
            ratings[away] = away_rating - change
        return ratings

    def fit(self, matches: pd.DataFrame, reference_date: pd.Timestamp) -> "GoalModel":
        rows = long_goal_rows(matches)
        age_years = (reference_date - rows["date"]).dt.days.clip(lower=0) / 365.25
        recency = np.power(0.5, age_years / self.half_life_years)
        importance = rows["tournament"].map(tournament_weight)
        weights = (recency * importance).clip(lower=0.02)

        preprocessor = ColumnTransformer(
            [
                (
                    "teams",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["team", "opponent"],
                ),
                ("venue", StandardScaler(), ["venue"]),
            ]
        )
        self.pipeline = make_pipeline(
            preprocessor,
            PoissonRegressor(alpha=self.alpha, max_iter=500, tol=1e-7),
        )
        self.pipeline.fit(rows[["team", "opponent", "venue"]], rows["goals"], **{
            "poissonregressor__sample_weight": weights
        })
        self.fallback_goals = float(np.average(rows["goals"], weights=weights))
        self.elo = self._fit_elo(matches)
        return self

    def expected_goals(self, team_a: str, team_b: str) -> tuple[float, float]:
        assert self.pipeline is not None
        frame = pd.DataFrame(
            [
                {"team": canonical(team_a), "opponent": canonical(team_b), "venue": 0.0},
                {"team": canonical(team_b), "opponent": canonical(team_a), "venue": 0.0},
            ]
        )
        values = self.pipeline.predict(frame)
        # Blend the Poisson attack/defense ratio with a slower-moving Elo prior.
        # This prevents one recent friendly from overwhelming long-term strength.
        assert self.elo is not None
        total = float(values.sum())
        glm_log_ratio = math.log(max(values[0], 0.05) / max(values[1], 0.05))
        rating_source = self.current_elo or self.elo
        elo_difference = rating_source.get(canonical(team_a), 1500.0) - rating_source.get(
            canonical(team_b), 1500.0
        )
        elo_log_ratio = elo_difference * math.log(10) / 500.0
        if self.current_elo:
            log_ratio = 0.35 * glm_log_ratio + 0.65 * elo_log_ratio
        else:
            log_ratio = 0.6 * glm_log_ratio + 0.4 * elo_log_ratio
        ratio = math.exp(log_ratio)
        blended = np.array([total * ratio / (1.0 + ratio), total / (1.0 + ratio)])
        return tuple(np.clip(blended, 0.18, 4.2))


def score_probabilities(
    lambda_home: float, lambda_away: float, max_goals: int = 9
) -> tuple[np.ndarray, float, float, float]:
    matrix = np.outer(
        poisson.pmf(np.arange(max_goals + 1), lambda_home),
        poisson.pmf(np.arange(max_goals + 1), lambda_away),
    )
    matrix /= matrix.sum()
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    return matrix, home, draw, away


def expected_points_score(
    matrix: np.ndarray, required_winner: str | None = None
) -> tuple[int, int]:
    best_score = (0, 0)
    best_points = -1.0
    for predicted_home in range(matrix.shape[0]):
        for predicted_away in range(matrix.shape[1]):
            if required_winner == "home" and predicted_home <= predicted_away:
                continue
            if required_winner == "away" and predicted_away <= predicted_home:
                continue
            expected_points = 0.0
            for actual_home in range(matrix.shape[0]):
                for actual_away in range(matrix.shape[1]):
                    probability = matrix[actual_home, actual_away]
                    if (predicted_home, predicted_away) == (actual_home, actual_away):
                        points = 25
                    else:
                        points = 0
                        if predicted_home - predicted_away == actual_home - actual_away:
                            points += 10
                        if predicted_home + predicted_away == actual_home + actual_away:
                            points += 10
                    expected_points += probability * points
            if expected_points > best_points:
                best_points = expected_points
                best_score = (predicted_home, predicted_away)
    return best_score


def knockout_score_matrix(
    regulation_matrix: np.ndarray,
    expected_home_goals: float,
    expected_away_goals: float,
) -> np.ndarray:
    maximum = regulation_matrix.shape[0] + 5
    final_matrix = np.zeros((maximum, maximum))
    extra_home = poisson.pmf(np.arange(6), expected_home_goals / 3.0)
    extra_away = poisson.pmf(np.arange(6), expected_away_goals / 3.0)
    extra_matrix = np.outer(extra_home, extra_away)
    extra_matrix /= extra_matrix.sum()
    for home_goals in range(regulation_matrix.shape[0]):
        for away_goals in range(regulation_matrix.shape[1]):
            probability = regulation_matrix[home_goals, away_goals]
            if home_goals != away_goals:
                final_matrix[home_goals, away_goals] += probability
                continue
            for extra_home_goals in range(extra_matrix.shape[0]):
                for extra_away_goals in range(extra_matrix.shape[1]):
                    final_matrix[
                        home_goals + extra_home_goals,
                        away_goals + extra_away_goals,
                    ] += probability * extra_matrix[
                        extra_home_goals, extra_away_goals
                    ]
    final_matrix /= final_matrix.sum()
    return final_matrix


def expected_points_count(
    rate: float,
    exact_points: int,
    near_points: int,
    near_distance: int,
    maximum: int,
) -> int:
    probabilities = poisson.pmf(np.arange(maximum + 1), rate)
    probabilities /= probabilities.sum()
    best_prediction = 0
    best_points = -1.0
    for prediction in range(maximum + 1):
        expected_points = 0.0
        for actual, probability in enumerate(probabilities):
            if prediction == actual:
                points = exact_points
            elif abs(prediction - actual) <= near_distance:
                points = near_points
            else:
                points = 0
            expected_points += probability * points
        if expected_points > best_points:
            best_points = expected_points
            best_prediction = prediction
    return best_prediction


def competition_points(
    actual_home: int,
    actual_away: int,
    predicted_home: int,
    predicted_away: int,
    predicted_outcome: int,
) -> float:
    points = 0.0
    if (actual_home, actual_away) == (predicted_home, predicted_away):
        points += 25
    else:
        if actual_home - actual_away == predicted_home - predicted_away:
            points += 10
        if actual_home + actual_away == predicted_home + predicted_away:
            points += 10
    actual_outcome = int(np.sign(actual_home - actual_away))
    if predicted_outcome == actual_outcome:
        points += 40
    return points


def evaluate_candidate(
    results: pd.DataFrame, alpha: float, half_life: float
) -> float:
    scores = []
    for year in (2018, 2022):
        cutoff = pd.Timestamp(f"{year}-06-01")
        train = results[results["date"] < cutoff]
        test = results[
            (results["date"].dt.year == year)
            & results["tournament"].str.fullmatch("FIFA World Cup", na=False)
        ]
        if test.empty:
            continue
        model = GoalModel(alpha, half_life).fit(train, cutoff)
        tournament_scores = []
        for row in test.itertuples():
            lh, la = model.expected_goals(row.home_team, row.away_team)
            matrix, ph, pd_, pa = score_probabilities(lh, la)
            score = np.unravel_index(np.argmax(matrix), matrix.shape)
            outcome = [1, 0, -1][int(np.argmax([ph, pd_, pa]))]
            tournament_scores.append(
                competition_points(
                    int(row.home_score),
                    int(row.away_score),
                    int(score[0]),
                    int(score[1]),
                    outcome,
                )
            )
        scores.append(float(np.mean(tournament_scores)))
    return float(np.mean(scores))


def choose_goal_model(results: pd.DataFrame) -> tuple[GoalModel, pd.DataFrame]:
    trials = []
    for alpha in (0.0005, 0.001, 0.003, 0.01):
        for half_life in (2.5, 4.0, 6.0):
            score = evaluate_candidate(results, alpha, half_life)
            trials.append(
                {
                    "alpha": alpha,
                    "half_life_years": half_life,
                    "mean_backtest_points": score,
                }
            )
    trials_df = pd.DataFrame(trials).sort_values(
        "mean_backtest_points", ascending=False
    )
    best = trials_df.iloc[0]
    model = GoalModel(
        float(best["alpha"]), float(best["half_life_years"])
    ).fit(results, REFERENCE_DATE)
    model.current_elo = load_current_elo()
    return model, trials_df


def prediction_for_match(
    model: MatchEnsemble,
    team_a: str,
    team_b: str,
    knockout: bool = False,
    required_winner: str | None = None,
) -> dict:
    ensemble_prediction = model.predict_match(team_a, team_b)
    lh = ensemble_prediction["expected_home_goals"]
    la = ensemble_prediction["expected_away_goals"]
    matrix = ensemble_prediction["score_matrix"]
    if knockout:
        matrix = knockout_score_matrix(matrix, lh, la)
    optimized_score = expected_points_score(matrix, required_winner)
    return {
        "predicted_home_goals": int(optimized_score[0]),
        "predicted_away_goals": int(optimized_score[1]),
        "corners": expected_points_count(
            ensemble_prediction["corner_mean"], 10, 5, 2, 24
        ),
        "yellow_cards": expected_points_count(
            ensemble_prediction["yellow_mean"], 10, 5, 1, 14
        ),
        "red_cards": int(ensemble_prediction["red_probability"] >= 0.5),
        "home_win_probability": ensemble_prediction["home_probability"],
        "draw_probability": ensemble_prediction["draw_probability"],
        "away_win_probability": ensemble_prediction["away_probability"],
        "lambda_home": lh,
        "lambda_away": la,
        "red_probability": ensemble_prediction["red_probability"],
    }


def simulate_group(
    group_fixtures: pd.DataFrame,
    model: MatchEnsemble,
    rng: np.random.Generator,
    simulations: int = 30000,
) -> tuple[list[str], pd.DataFrame]:
    teams = sorted(
        set(group_fixtures["home_team"]) | set(group_fixtures["away_team"])
    )
    index = {team: i for i, team in enumerate(teams)}
    order_counts: dict[tuple[str, ...], int] = {}
    summaries = {
        team: {"points": 0.0, "goal_difference": 0.0, "goals_for": 0.0}
        for team in teams
    }

    rates = []
    for row in group_fixtures.itertuples():
        prediction = model.predict_match(row.home_team, row.away_team)
        rates.append(
            (
                prediction["expected_home_goals"],
                prediction["expected_away_goals"],
                row,
            )
        )

    for _ in range(simulations):
        points = np.zeros(4, dtype=int)
        gf = np.zeros(4, dtype=int)
        ga = np.zeros(4, dtype=int)
        match_results = []
        for lh, la, row in rates:
            home_goals = int(rng.poisson(lh))
            away_goals = int(rng.poisson(la))
            hi, ai = index[row.home_team], index[row.away_team]
            match_results.append((hi, ai, home_goals, away_goals))
            gf[hi] += home_goals
            ga[hi] += away_goals
            gf[ai] += away_goals
            ga[ai] += home_goals
            if home_goals > away_goals:
                points[hi] += 3
            elif home_goals < away_goals:
                points[ai] += 3
            else:
                points[hi] += 1
                points[ai] += 1

        ranked_idx = []
        for point_total in sorted(set(points), reverse=True):
            tied = [i for i in range(4) if points[i] == point_total]
            head_points = {i: 0 for i in tied}
            head_gf = {i: 0 for i in tied}
            head_ga = {i: 0 for i in tied}
            for hi, ai, home_goals, away_goals in match_results:
                if hi not in head_points or ai not in head_points:
                    continue
                head_gf[hi] += home_goals
                head_ga[hi] += away_goals
                head_gf[ai] += away_goals
                head_ga[ai] += home_goals
                if home_goals > away_goals:
                    head_points[hi] += 3
                elif home_goals < away_goals:
                    head_points[ai] += 3
                else:
                    head_points[hi] += 1
                    head_points[ai] += 1
            tied.sort(
                key=lambda i: (
                    head_points[i],
                    head_gf[i] - head_ga[i],
                    head_gf[i],
                    gf[i] - ga[i],
                    gf[i],
                    model.current_elo.get(canonical(teams[i]), 1500.0),
                ),
                reverse=True,
            )
            ranked_idx.extend(tied)
        order = tuple(teams[i] for i in ranked_idx)
        order_counts[order] = order_counts.get(order, 0) + 1
        for team, i in index.items():
            summaries[team]["points"] += points[i]
            summaries[team]["goal_difference"] += gf[i] - ga[i]
            summaries[team]["goals_for"] += gf[i]

    modal_order = max(order_counts, key=order_counts.get)
    summary = pd.DataFrame.from_dict(summaries, orient="index") / simulations
    summary["modal_order_probability"] = order_counts[modal_order] / simulations
    return list(modal_order), summary


@lru_cache(maxsize=1)
def third_place_combination_map() -> dict[str, dict[str, str]]:
    combinations = pd.read_csv(
        COMP_DIR / "data/third_place_combinations.csv", dtype=str
    )
    return {
        row["qualified_groups"]: {
            winner_group: row[winner_group]
            for winner_group in ("1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L")
        }
        for _, row in combinations.iterrows()
    }


def assign_third_place_slots(
    selected_groups: set[str], knockout: pd.DataFrame
) -> dict[int, str]:
    key = "".join(sorted(selected_groups))
    combinations = third_place_combination_map()
    if key not in combinations:
        raise RuntimeError(f"No unique FIFA Annex C mapping for groups {key}")
    option = combinations[key]
    winner_group_to_match = {
        "1A": 79,
        "1B": 85,
        "1D": 81,
        "1E": 74,
        "1G": 82,
        "1I": 77,
        "1K": 87,
        "1L": 80,
    }
    assignment = {
        match_id: str(option[winner_group])
        for winner_group, match_id in winner_group_to_match.items()
    }
    third_place_matches = knockout[
        knockout["slot_away"].str.startswith("Best 3rd", na=False)
    ]["match_id"].astype(int)
    if set(assignment) != set(third_place_matches):
        raise RuntimeError("Knockout slots do not match FIFA Annex C winner slots")
    return assignment


def resolve_slot(
    slot: str,
    group_orders: dict[str, list[str]],
    match_results: dict[int, dict],
    third_assignment: dict[int, str],
    match_id: int,
) -> str:
    if slot.startswith("Winner Group "):
        return group_orders[slot[-1]][0]
    if slot.startswith("Runner-up Group "):
        return group_orders[slot[-1]][1]
    if slot.startswith("Best 3rd"):
        return group_orders[third_assignment[match_id]][2]
    if slot.startswith("Winner Match "):
        source = int(slot.rsplit(" ", 1)[1])
        return match_results[source]["winner"]
    if slot.startswith("Loser Match "):
        source = int(slot.rsplit(" ", 1)[1])
        return match_results[source]["loser"]
    raise ValueError(f"Unknown bracket slot: {slot}")


def rank_simulated_group(
    teams: list[str],
    points: np.ndarray,
    goals_for: np.ndarray,
    goals_against: np.ndarray,
    results: list[tuple[int, int, int, int]],
    current_elo: dict[str, float],
) -> list[str]:
    ranked_indices = []
    for point_total in sorted(set(points), reverse=True):
        tied = [i for i in range(4) if points[i] == point_total]
        head_points = {i: 0 for i in tied}
        head_gf = {i: 0 for i in tied}
        head_ga = {i: 0 for i in tied}
        for home_index, away_index, home_goals, away_goals in results:
            if home_index not in head_points or away_index not in head_points:
                continue
            head_gf[home_index] += home_goals
            head_ga[home_index] += away_goals
            head_gf[away_index] += away_goals
            head_ga[away_index] += home_goals
            if home_goals > away_goals:
                head_points[home_index] += 3
            elif home_goals < away_goals:
                head_points[away_index] += 3
            else:
                head_points[home_index] += 1
                head_points[away_index] += 1
        tied.sort(
            key=lambda i: (
                head_points[i],
                head_gf[i] - head_ga[i],
                head_gf[i],
                goals_for[i] - goals_against[i],
                goals_for[i],
                current_elo.get(canonical(teams[i]), 1500.0),
            ),
            reverse=True,
        )
        ranked_indices.extend(tied)
    return [teams[index] for index in ranked_indices]


def optimize_full_bracket(
    fixtures: pd.DataFrame,
    knockout: pd.DataFrame,
    model: MatchEnsemble,
    rng: np.random.Generator,
    simulations: int = 5000,
) -> dict[int, dict]:
    group_rates = {}
    for group, games in fixtures.groupby("group"):
        teams = sorted(set(games["home_team"]) | set(games["away_team"]))
        index = {team: position for position, team in enumerate(teams)}
        rates = []
        for row in games.itertuples():
            prediction = model.predict_match(row.home_team, row.away_team)
            rates.append(
                (
                    index[row.home_team],
                    index[row.away_team],
                    prediction["expected_home_goals"],
                    prediction["expected_away_goals"],
                )
            )
        group_rates[group] = (teams, rates)

    prediction_cache = {}
    scenarios = []
    for _ in range(simulations):
        group_orders = {}
        third_rankings = []
        for group, (teams, rates) in group_rates.items():
            points = np.zeros(4, dtype=int)
            goals_for = np.zeros(4, dtype=int)
            goals_against = np.zeros(4, dtype=int)
            results = []
            for home_index, away_index, home_rate, away_rate in rates:
                home_goals = int(rng.poisson(home_rate))
                away_goals = int(rng.poisson(away_rate))
                results.append((home_index, away_index, home_goals, away_goals))
                goals_for[home_index] += home_goals
                goals_against[home_index] += away_goals
                goals_for[away_index] += away_goals
                goals_against[away_index] += home_goals
                if home_goals > away_goals:
                    points[home_index] += 3
                elif home_goals < away_goals:
                    points[away_index] += 3
                else:
                    points[home_index] += 1
                    points[away_index] += 1
            order = rank_simulated_group(
                teams,
                points,
                goals_for,
                goals_against,
                results,
                model.current_elo,
            )
            group_orders[group] = order
            third_index = teams.index(order[2])
            third_rankings.append(
                (
                    group,
                    points[third_index],
                    goals_for[third_index] - goals_against[third_index],
                    goals_for[third_index],
                    model.current_elo.get(canonical(order[2]), 1500.0),
                )
            )

        third_rankings.sort(key=lambda item: item[1:], reverse=True)
        third_groups = {item[0] for item in third_rankings[:8]}
        third_assignment = assign_third_place_slots(third_groups, knockout)

        match_results = {}
        for row in knockout.sort_values("match_id").itertuples():
            home = resolve_slot(
                row.slot_home,
                group_orders,
                match_results,
                third_assignment,
                row.match_id,
            )
            away = resolve_slot(
                row.slot_away,
                group_orders,
                match_results,
                third_assignment,
                row.match_id,
            )
            key = (home, away)
            if key not in prediction_cache:
                prediction_cache[key] = model.predict_core(home, away)
            prediction = prediction_cache[key]
            non_draw_total = (
                prediction["home_probability"] + prediction["away_probability"]
            )
            home_advance_probability = (
                prediction["home_probability"] / non_draw_total
            )
            home_advances = rng.random() < home_advance_probability
            winner = home if home_advances else away
            loser = away if home_advances else home
            match_results[int(row.match_id)] = {
                "home": home,
                "away": away,
                "winner": winner,
                "loser": loser,
            }
        scenarios.append(match_results)

    pair_counts = {match_id: Counter() for match_id in range(73, 105)}
    winner_counts = {match_id: Counter() for match_id in range(73, 105)}
    for scenario in scenarios:
        for match_id, result in scenario.items():
            pair_counts[match_id][(result["home"], result["away"])] += 1
            winner_counts[match_id][result["winner"]] += 1

    multipliers = knockout.set_index("match_id")["multiplier"].to_dict()

    def scenario_utility(scenario: dict[int, dict]) -> float:
        utility = 0.0
        for match_id, prediction in scenario.items():
            predicted_pair = (prediction["home"], prediction["away"])
            matchup_points = 0.0
            for actual_pair, count in pair_counts[match_id].items():
                correct_sides = int(predicted_pair[0] == actual_pair[0]) + int(
                    predicted_pair[1] == actual_pair[1]
                )
                matchup_points += count * (20 if correct_sides == 2 else 10 if correct_sides == 1 else 0)
            winner_points = 20 * winner_counts[match_id][prediction["winner"]]
            utility += multipliers[match_id] * (
                matchup_points + winner_points
            ) / simulations
        return utility

    return max(scenarios, key=scenario_utility)


def build_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = load_results()
    fixtures, knockout = load_fixtures()
    poisson_model, trials = choose_goal_model(results)

    def historical_poisson(
        training_matches: pd.DataFrame, cutoff: pd.Timestamp
    ) -> GoalModel:
        return GoalModel(0.003, 2.5).fit(training_matches, cutoff)

    backtest_path = COMP_DIR / "ensemble_backtest.csv"
    if backtest_path.exists():
        cached_backtest = pd.read_csv(backtest_path)
        if "validation_event" in cached_backtest.columns:
            ensemble_backtest = cached_backtest
        else:
            ensemble_backtest = backtest_goal_ensemble(
                results, tournament_weight, historical_poisson
            )
    else:
        ensemble_backtest = backtest_goal_ensemble(
            results, tournament_weight, historical_poisson
        )
    mean_backtest = (
        ensemble_backtest.groupby(["gradient_weight", "outcome_weight"])[
            "mean_total_points"
        ]
        .mean()
        .sort_values(ascending=False)
    )
    best_gradient_weight, best_outcome_weight = mean_backtest.index[0]
    model = MatchEnsemble(
        canonical_function=canonical,
        importance_function=tournament_weight,
        poisson_model=poisson_model,
        current_elo=load_current_elo(),
        gradient_weight=float(best_gradient_weight),
        outcome_weight=float(best_outcome_weight),
    )
    model.fit(results, ROOT, REFERENCE_DATE)

    group_rows = []
    for row in fixtures.itertuples():
        prediction = prediction_for_match(model, row.home_team, row.away_team)
        outcome = ["home", "draw", "away"][
            int(
                np.argmax(
                    [
                        prediction["home_win_probability"],
                        prediction["draw_probability"],
                        prediction["away_win_probability"],
                    ]
                )
            )
        ]
        group_rows.append(
            {
                **row._asdict(),
                **{k: prediction[k] for k in (
                    "predicted_home_goals",
                    "predicted_away_goals",
                    "corners",
                    "yellow_cards",
                    "red_cards",
                )},
                "winning_team": outcome,
            }
        )
    group_predictions = pd.DataFrame(group_rows).drop(columns=["Index"], errors="ignore")

    rng = np.random.default_rng(RNG_SEED)
    optimized_scenario = optimize_full_bracket(
        fixtures, knockout, model, rng, simulations=5000
    )

    match_results: dict[int, dict] = {}
    knockout_rows = []
    for row in knockout.sort_values("match_id").itertuples():
        home = optimized_scenario[int(row.match_id)]["home"]
        away = optimized_scenario[int(row.match_id)]["away"]
        winner = optimized_scenario[int(row.match_id)]["winner"]
        loser = away if winner == home else home
        home_advance = winner == home
        prediction = prediction_for_match(
            model,
            home,
            away,
            knockout=True,
            required_winner="home" if home_advance else "away",
        )
        match_results[int(row.match_id)] = {"winner": winner, "loser": loser}
        round_code = {
            "Round of 32": 1,
            "Round of 16": 2,
            "Quarter-final": 3,
            "Semi-final": 4,
            "Third-place playoff": 4,
            "Final": 5,
        }[row.round]
        penalty_probability = model.penalty_probability(
            home,
            away,
            prediction["lambda_home"],
            prediction["lambda_away"],
            prediction["draw_probability"],
            round_code,
        )
        knockout_rows.append(
            {
                **row._asdict(),
                "predicted_home_team": home,
                "predicted_away_team": away,
                **{k: prediction[k] for k in (
                    "predicted_home_goals",
                    "predicted_away_goals",
                    "corners",
                    "yellow_cards",
                    "red_cards",
                )},
                "match_winner": "home" if home_advance else "away",
                "penalties": bool(penalty_probability >= 0.5),
            }
        )
    knockout_predictions = pd.DataFrame(knockout_rows).drop(
        columns=["Index"], errors="ignore"
    )
    return group_predictions, knockout_predictions, trials, ensemble_backtest


def dataframe_literal(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False)


def write_notebook(
    group_predictions: pd.DataFrame,
    knockout_predictions: pd.DataFrame,
    trials: pd.DataFrame,
    ensemble_backtest: pd.DataFrame,
) -> None:
    group_csv = dataframe_literal(group_predictions)
    knockout_csv = dataframe_literal(knockout_predictions)
    best = trials.iloc[0]
    mean_backtest = (
        ensemble_backtest.groupby(["gradient_weight", "outcome_weight"])[
            "mean_total_points"
        ]
        .mean()
        .sort_values(ascending=False)
    )
    best_gradient_weight, best_outcome_weight = mean_backtest.index[0]
    best_ensemble_points = mean_backtest.iloc[0]
    poisson_points = mean_backtest.loc[(0.0, 0.0)]

    def markdown(source: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}

    def code(source: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(True),
        }

    notebook = {
        "cells": [
            markdown(
                "# FIFA World Cup 2026 predictions\n\n"
                "This submission uses completed international matches through "
                "March 31, 2026. It combines regularized Poisson score distributions, "
                "gradient-boosted home/away goal models, and a gradient-boosted "
                "win/draw/loss classifier. Match importance, rolling form, rolling "
                "attack/defense, Elo strength, and exponential recency weighting "
                "are calculated without future-match leakage. Separate trained "
                "models predict corners, yellow cards, red-card probability, and "
                "penalty-shootout probability. "
                "USA, Canada, and Mexico receive a modest host-strength adjustment. "
                "Predicted integers maximize expected competition points rather than "
                "simply rounding model means, and knockout score distributions include "
                "the extra-time period required by the competition rules. "
                "Five thousand complete tournaments were simulated using FIFA's "
                "head-to-head group tiebreakers and exact 495-row Annex C mapping. "
                "A coherent bracket was selected to maximize expected matchup and "
                "winner points after applying the knockout-round multipliers.\n\n"
                "The six playoff placeholders have been replaced by Bosnia and "
                "Herzegovina, Sweden, Turkey, Czech Republic, DR Congo, and Iraq."
            ),
            code("from io import StringIO\nimport pandas as pd\n"),
            markdown(
                "## Model selection\n\n"
                f"Selected regularization `alpha={best['alpha']}` and a "
                f"`{best['half_life_years']}`-year recency half-life. The historical "
                f"mean competition score was `{best['mean_backtest_points']:.2f}` "
                "points per match for the Poisson tuning stage. Chronological 2018 "
                f"and 2022 backtesting selected goal blend `{best_gradient_weight:.2f}` "
                f"and outcome-classifier blend `{best_outcome_weight:.2f}`, improving "
                f"mean score/outcome points from `{poisson_points:.2f}` to "
                f"`{best_ensemble_points:.2f}` per match."
            ),
            markdown("## Group stage predictions"),
            code(
                "group_predictions = pd.read_csv(StringIO(r'''\n"
                + group_csv
                + "'''))\n"
                "group_predictions\n"
            ),
            markdown("## Knockout stage predictions"),
            code(
                "knockout_predictions = pd.read_csv(StringIO(r'''\n"
                + knockout_csv
                + "'''))\n"
                "knockout_predictions\n"
            ),
            markdown("## Submission validation"),
            code(
                "assert len(group_predictions) == 72\n"
                "assert len(knockout_predictions) == 32\n"
                "assert not group_predictions.isna().any().any()\n"
                "assert not knockout_predictions.isna().any().any()\n"
                "assert set(group_predictions['winning_team']) <= {'home', 'away', 'draw'}\n"
                "assert set(knockout_predictions['match_winner']) <= {'home', 'away'}\n"
                "assert set(knockout_predictions['penalties']) <= {True, False}\n"
                "knockout_score_winner = knockout_predictions.apply(\n"
                "    lambda row: 'home' if row.predicted_home_goals > row.predicted_away_goals else 'away', axis=1\n"
                ")\n"
                "assert (knockout_score_winner == knockout_predictions['match_winner']).all()\n"
                "print('Validated all 104 predictions with no missing values.')\n"
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (COMP_DIR / "notebook.ipynb").write_text(json.dumps(notebook, indent=1))


def main() -> None:
    (
        group_predictions,
        knockout_predictions,
        trials,
        ensemble_backtest,
    ) = build_predictions()
    group_predictions.to_csv(COMP_DIR / "group_predictions.csv", index=False)
    knockout_predictions.to_csv(COMP_DIR / "knockout_predictions.csv", index=False)
    trials.to_csv(COMP_DIR / "model_backtest.csv", index=False)
    ensemble_backtest.to_csv(COMP_DIR / "ensemble_backtest.csv", index=False)
    write_notebook(
        group_predictions, knockout_predictions, trials, ensemble_backtest
    )
    print(trials.head().to_string(index=False))
    print(
        f"\nWrote {len(group_predictions)} group and "
        f"{len(knockout_predictions)} knockout predictions."
    )


if __name__ == "__main__":
    main()
