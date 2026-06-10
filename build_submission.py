from __future__ import annotations

import json
import math
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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


def card_priors() -> tuple[float, dict[str, float]]:
    bookings = pd.read_csv(ROOT / "data/worldcup/data-csv/bookings.csv")
    recent = bookings[
        bookings["tournament_name"].str.match(r"20(10|14|18|22) FIFA Men's World Cup")
    ].copy()
    match_totals = recent.groupby("match_id")["yellow_card"].sum()
    global_median = float(match_totals.median())
    team_match = recent.groupby(["team_name", "match_id"])["yellow_card"].sum()
    team_mean = team_match.groupby("team_name").mean()
    shrunk = ((team_mean * team_match.groupby("team_name").size()) + global_median * 6) / (
        team_match.groupby("team_name").size() + 6
    )
    return global_median, shrunk.to_dict()


def auxiliary_predictions(
    team_a: str,
    team_b: str,
    lambda_a: float,
    lambda_b: float,
    yellow_base: float,
    yellow_team: dict[str, float],
) -> tuple[int, int, int]:
    expected_total = lambda_a + lambda_b
    corners = int(np.clip(round(9.7 + 0.35 * (expected_total - 2.6)), 7, 13))
    ya = yellow_team.get(canonical(team_a), yellow_base / 2)
    yb = yellow_team.get(canonical(team_b), yellow_base / 2)
    yellow = int(np.clip(round(0.55 * (ya + yb) + 0.45 * yellow_base), 2, 7))
    # Zero is the modal result by a wide margin in both recent World Cups and club data.
    red = 0
    return corners, yellow, red


def prediction_for_match(
    model: GoalModel,
    team_a: str,
    team_b: str,
    yellow_base: float,
    yellow_team: dict[str, float],
) -> dict:
    lh, la = model.expected_goals(team_a, team_b)
    matrix, ph, pd_, pa = score_probabilities(lh, la)
    modal = np.unravel_index(np.argmax(matrix), matrix.shape)
    corners, yellow, red = auxiliary_predictions(
        team_a, team_b, lh, la, yellow_base, yellow_team
    )
    return {
        "predicted_home_goals": int(modal[0]),
        "predicted_away_goals": int(modal[1]),
        "corners": corners,
        "yellow_cards": yellow,
        "red_cards": red,
        "home_win_probability": ph,
        "draw_probability": pd_,
        "away_win_probability": pa,
        "lambda_home": lh,
        "lambda_away": la,
    }


def simulate_group(
    group_fixtures: pd.DataFrame,
    model: GoalModel,
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
        rates.append((*model.expected_goals(row.home_team, row.away_team), row))

    for _ in range(simulations):
        points = np.zeros(4, dtype=int)
        gf = np.zeros(4, dtype=int)
        ga = np.zeros(4, dtype=int)
        for lh, la, row in rates:
            home_goals = int(rng.poisson(lh))
            away_goals = int(rng.poisson(la))
            hi, ai = index[row.home_team], index[row.away_team]
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

        # A fixed random tie-break key prevents alphabetical bias in exact ties.
        tie_break = rng.random(4)
        ranked_idx = sorted(
            range(4),
            key=lambda i: (points[i], gf[i] - ga[i], gf[i], tie_break[i]),
            reverse=True,
        )
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


def parse_allowed_groups(slot: str) -> set[str]:
    inside = slot.split("Groups ", 1)[1].rstrip(")")
    return set(inside.split("/"))


def assign_third_place_slots(
    selected_groups: set[str], knockout: pd.DataFrame
) -> dict[int, str]:
    slots = []
    for row in knockout[knockout["round"] == "Round of 32"].itertuples():
        for side, value in (("home", row.slot_home), ("away", row.slot_away)):
            if str(value).startswith("Best 3rd"):
                slots.append((int(row.match_id), side, parse_allowed_groups(value)))

    slots.sort(key=lambda item: len(item[2] & selected_groups))
    assignment: dict[tuple[int, str], str] = {}

    def search(position: int, remaining: set[str]) -> bool:
        if position == len(slots):
            return not remaining
        match_id, side, allowed = slots[position]
        for group in sorted(allowed & remaining):
            assignment[(match_id, side)] = group
            if search(position + 1, remaining - {group}):
                return True
        assignment.pop((match_id, side), None)
        return False

    if not search(0, set(selected_groups)):
        raise RuntimeError("Could not map the selected third-place teams to bracket slots")
    return {match_id: group for (match_id, _), group in assignment.items()}


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


def build_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = load_results()
    fixtures, knockout = load_fixtures()
    model, trials = choose_goal_model(results)
    yellow_base, yellow_team = card_priors()

    group_rows = []
    for row in fixtures.itertuples():
        prediction = prediction_for_match(
            model, row.home_team, row.away_team, yellow_base, yellow_team
        )
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
    group_orders: dict[str, list[str]] = {}
    third_strength = []
    for group, group_games in fixtures.groupby("group"):
        order, summary = simulate_group(group_games, model, rng)
        group_orders[group] = order
        third = order[2]
        third_strength.append(
            (
                group,
                third,
                float(summary.loc[third, "points"]),
                float(summary.loc[third, "goal_difference"]),
                float(summary.loc[third, "goals_for"]),
            )
        )
    third_strength.sort(key=lambda row: row[2:], reverse=True)
    selected_third_groups = {row[0] for row in third_strength[:8]}
    third_assignment = assign_third_place_slots(selected_third_groups, knockout)

    match_results: dict[int, dict] = {}
    knockout_rows = []
    for row in knockout.sort_values("match_id").itertuples():
        home = resolve_slot(
            row.slot_home, group_orders, match_results, third_assignment, row.match_id
        )
        away = resolve_slot(
            row.slot_away, group_orders, match_results, third_assignment, row.match_id
        )
        prediction = prediction_for_match(
            model, home, away, yellow_base, yellow_team
        )
        home_advance = prediction["home_win_probability"] >= prediction["away_win_probability"]
        winner = home if home_advance else away
        loser = away if home_advance else home
        match_results[int(row.match_id)] = {"winner": winner, "loser": loser}
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
                # A shootout is a low-probability event for every individual fixture.
                "penalties": False,
            }
        )
    knockout_predictions = pd.DataFrame(knockout_rows).drop(
        columns=["Index"], errors="ignore"
    )
    return group_predictions, knockout_predictions, trials


def dataframe_literal(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False)


def write_notebook(
    group_predictions: pd.DataFrame,
    knockout_predictions: pd.DataFrame,
    trials: pd.DataFrame,
) -> None:
    group_csv = dataframe_literal(group_predictions)
    knockout_csv = dataframe_literal(knockout_predictions)
    best = trials.iloc[0]

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
                "March 31, 2026. A regularized Poisson attack/defense model was "
                "selected using 2018 and 2022 World Cup backtests. Match importance "
                "and exponential time decay weight the training observations. The "
                "goal balance is blended with the June 10, 2026 World Football Elo "
                "ratings to stabilize current relative team strength. "
                "Group standings were simulated 30,000 times per group and then "
                "propagated through the official 32-match competition bracket.\n\n"
                "The six playoff placeholders have been replaced by Bosnia and "
                "Herzegovina, Sweden, Turkey, Czech Republic, DR Congo, and Iraq."
            ),
            code("from io import StringIO\nimport pandas as pd\n"),
            markdown(
                "## Model selection\n\n"
                f"Selected regularization `alpha={best['alpha']}` and a "
                f"`{best['half_life_years']}`-year recency half-life. The historical "
                f"mean competition score was `{best['mean_backtest_points']:.2f}` "
                "points per match for the score and outcome categories."
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
    group_predictions, knockout_predictions, trials = build_predictions()
    group_predictions.to_csv(COMP_DIR / "group_predictions.csv", index=False)
    knockout_predictions.to_csv(COMP_DIR / "knockout_predictions.csv", index=False)
    trials.to_csv(COMP_DIR / "model_backtest.csv", index=False)
    write_notebook(group_predictions, knockout_predictions, trials)
    print(trials.head().to_string(index=False))
    print(
        f"\nWrote {len(group_predictions)} group and "
        f"{len(knockout_predictions)} knockout predictions."
    )


if __name__ == "__main__":
    main()
