from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROLLING_RATE = 0.16
INITIAL_GOALS = 1.25
INITIAL_POINTS = 1.3
ELO_HOME_ADVANTAGE = 65.0
TOURNAMENT_HOSTS = {"United States", "Canada", "Mexico"}
HOST_ELO_BONUS = 45.0

MATCH_FEATURES = [
    "home_elo",
    "away_elo",
    "elo_difference",
    "home_attack",
    "away_attack",
    "home_defense",
    "away_defense",
    "home_form",
    "away_form",
    "attack_difference",
    "defense_difference",
    "form_difference",
    "home_experience",
    "away_experience",
    "neutral",
    "importance",
]

AUXILIARY_FEATURES = [
    "expected_home_goals",
    "expected_away_goals",
    "expected_total_goals",
    "expected_goal_difference",
    "absolute_elo_difference",
    "draw_probability",
    "knockout",
    "round_code",
]


@dataclass
class TeamState:
    elo: float = 1500.0
    attack: float = INITIAL_GOALS
    defense: float = INITIAL_GOALS
    form: float = INITIAL_POINTS
    matches: int = 0


def _outcome_points(goals_for: float, goals_against: float) -> float:
    if goals_for > goals_against:
        return 3.0
    if goals_for < goals_against:
        return 0.0
    return 1.0


def _elo_change(
    home_state: TeamState,
    away_state: TeamState,
    home_goals: float,
    away_goals: float,
    neutral: bool,
    importance: float,
) -> float:
    advantage = 0.0 if neutral else ELO_HOME_ADVANTAGE
    expected = 1.0 / (
        1.0 + 10 ** (-(home_state.elo + advantage - away_state.elo) / 400.0)
    )
    actual = _outcome_points(home_goals, away_goals) / 3.0
    if home_goals == away_goals:
        actual = 0.5
    margin = abs(home_goals - away_goals)
    margin_multiplier = 1.0 if margin <= 1 else math.log1p(margin)
    return 18.0 * min(importance, 2.5) * margin_multiplier * (actual - expected)


def _state_features(
    home_state: TeamState,
    away_state: TeamState,
    neutral: bool,
    importance: float,
) -> dict[str, float]:
    return {
        "home_elo": home_state.elo,
        "away_elo": away_state.elo,
        "elo_difference": home_state.elo - away_state.elo,
        "home_attack": home_state.attack,
        "away_attack": away_state.attack,
        "home_defense": home_state.defense,
        "away_defense": away_state.defense,
        "home_form": home_state.form,
        "away_form": away_state.form,
        "attack_difference": home_state.attack - away_state.attack,
        "defense_difference": away_state.defense - home_state.defense,
        "form_difference": home_state.form - away_state.form,
        "home_experience": math.log1p(home_state.matches),
        "away_experience": math.log1p(away_state.matches),
        "neutral": float(neutral),
        "importance": importance,
    }


def _update_state(
    team_state: TeamState,
    goals_for: float,
    goals_against: float,
) -> None:
    rate = ROLLING_RATE
    team_state.attack = (1.0 - rate) * team_state.attack + rate * goals_for
    team_state.defense = (1.0 - rate) * team_state.defense + rate * goals_against
    team_state.form = (1.0 - rate) * team_state.form + rate * _outcome_points(
        goals_for, goals_against
    )
    team_state.matches += 1


def build_rolling_features(
    matches: pd.DataFrame,
    importance_function,
) -> tuple[pd.DataFrame, dict[str, TeamState]]:
    states: dict[str, TeamState] = {}
    rows = []
    for match in matches.sort_values("date").itertuples():
        home_state = states.setdefault(match.home_team, TeamState())
        away_state = states.setdefault(match.away_team, TeamState())
        importance = float(importance_function(match.tournament))
        row = _state_features(home_state, away_state, bool(match.neutral), importance)
        row.update(
            {
                "date": match.date,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "home_score": float(match.home_score),
                "away_score": float(match.away_score),
                "outcome": 0
                if match.home_score > match.away_score
                else 2
                if match.home_score < match.away_score
                else 1,
            }
        )
        rows.append(row)

        change = _elo_change(
            home_state,
            away_state,
            float(match.home_score),
            float(match.away_score),
            bool(match.neutral),
            importance,
        )
        home_state.elo += change
        away_state.elo -= change
        _update_state(home_state, float(match.home_score), float(match.away_score))
        _update_state(away_state, float(match.away_score), float(match.home_score))
    return pd.DataFrame(rows), states


def _sample_weights(
    dates: pd.Series,
    importance: pd.Series,
    reference_date: pd.Timestamp,
    half_life_years: float = 4.0,
) -> np.ndarray:
    age_years = (reference_date - dates).dt.days.clip(lower=0) / 365.25
    return np.power(0.5, age_years / half_life_years) * importance.clip(lower=0.4)


def backtest_goal_ensemble(
    results: pd.DataFrame,
    importance_function,
    poisson_factory,
) -> pd.DataFrame:
    from scipy.stats import poisson

    records = []
    validation_events = [
        ("2016 Euro/Copa", "2016-06-01", "2016-12-31", {"UEFA Euro", "Copa América"}),
        ("2018 World Cup", "2018-06-01", "2018-12-31", {"FIFA World Cup"}),
        (
            "2019 continental",
            "2019-01-01",
            "2019-12-31",
            {"AFC Asian Cup", "African Cup of Nations", "Copa América", "Gold Cup"},
        ),
        ("2021 Euro/Copa", "2021-06-01", "2021-12-31", {"UEFA Euro", "Copa América"}),
        (
            "2022 AFCON",
            "2022-01-01",
            "2022-03-01",
            {"African Cup of Nations"},
        ),
        ("2022 World Cup", "2022-11-01", "2022-12-31", {"FIFA World Cup"}),
        (
            "2024 winter cups",
            "2024-01-01",
            "2024-04-01",
            {"AFC Asian Cup", "African Cup of Nations"},
        ),
        ("2024 Euro/Copa", "2024-06-01", "2024-12-31", {"UEFA Euro", "Copa América"}),
        (
            "2025 AFCON",
            "2025-12-01",
            "2026-03-01",
            {"African Cup of Nations"},
        ),
    ]
    for event_name, cutoff_text, end_text, tournaments in validation_events:
        cutoff = pd.Timestamp(cutoff_text)
        end_date = pd.Timestamp(end_text)
        training_matches = results[results["date"] < cutoff]
        test_matches = results[
            (results["date"] >= cutoff)
            & (results["date"] <= end_date)
            & results["tournament"].isin(tournaments)
        ]
        if test_matches.empty:
            continue
        training_rows, states = build_rolling_features(
            training_matches, importance_function
        )
        weights = _sample_weights(
            training_rows["date"], training_rows["importance"], cutoff
        )
        features = training_rows[MATCH_FEATURES]

        goal_parameters = {
            "loss": "poisson",
            "learning_rate": 0.045,
            "max_iter": 260,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 35,
            "l2_regularization": 1.5,
            "random_state": 20260610,
        }
        home_model = HistGradientBoostingRegressor(**goal_parameters)
        away_model = HistGradientBoostingRegressor(**goal_parameters)
        home_model.fit(features, training_rows["home_score"], sample_weight=weights)
        away_model.fit(features, training_rows["away_score"], sample_weight=weights)
        outcome_model = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=280,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=2.0,
            random_state=20260610,
        )
        outcome_model.fit(features, training_rows["outcome"], sample_weight=weights)
        poisson_model = poisson_factory(training_matches, cutoff)

        for match in test_matches.itertuples():
            home_state = states.get(match.home_team, TeamState())
            away_state = states.get(match.away_team, TeamState())
            row = _state_features(home_state, away_state, True, 3.0)
            test_features = pd.DataFrame([row], columns=MATCH_FEATURES)
            poisson_home, poisson_away = poisson_model.expected_goals(
                match.home_team, match.away_team
            )
            gradient_home = float(home_model.predict(test_features)[0])
            gradient_away = float(away_model.predict(test_features)[0])

            for gradient_blend in (0.0, 0.05, 0.10, 0.15, 0.30, 0.45, 0.60):
                home_rate = (
                    (1.0 - gradient_blend) * poisson_home
                    + gradient_blend * gradient_home
                )
                away_rate = (
                    (1.0 - gradient_blend) * poisson_away
                    + gradient_blend * gradient_away
                )
                matrix = np.outer(
                    poisson.pmf(np.arange(10), home_rate),
                    poisson.pmf(np.arange(10), away_rate),
                )
                matrix /= matrix.sum()
                score_probabilities = np.array(
                    [
                        np.tril(matrix, -1).sum(),
                        np.trace(matrix),
                        np.triu(matrix, 1).sum(),
                    ]
                )
                classifier_probabilities = outcome_model.predict_proba(
                    test_features
                )[0]
                for outcome_blend in (0.0, 0.10, 0.20, 0.30, 0.40, 0.60):
                    probabilities = (
                        (1.0 - outcome_blend) * score_probabilities
                        + outcome_blend * classifier_probabilities
                    )
                    modal_score = np.unravel_index(np.argmax(matrix), matrix.shape)
                    predicted_outcome = int(np.argmax(probabilities))
                    actual_outcome = (
                        0
                        if match.home_score > match.away_score
                        else 2
                        if match.home_score < match.away_score
                        else 1
                    )
                    score_points = 0
                    if modal_score == (int(match.home_score), int(match.away_score)):
                        score_points = 25
                    else:
                        if modal_score[0] - modal_score[1] == match.home_score - match.away_score:
                            score_points += 10
                        if modal_score[0] + modal_score[1] == match.home_score + match.away_score:
                            score_points += 10
                    records.append(
                        {
                            "validation_event": event_name,
                            "gradient_weight": gradient_blend,
                            "outcome_weight": outcome_blend,
                            "score_points": score_points,
                            "outcome_points": 40
                            if predicted_outcome == actual_outcome
                            else 0,
                            "total_points": score_points
                            + (40 if predicted_outcome == actual_outcome else 0),
                        }
                    )
    return (
        pd.DataFrame(records)
        .groupby(
            ["validation_event", "gradient_weight", "outcome_weight"],
            as_index=False,
        )
        .agg(
            matches=("total_points", "size"),
            mean_score_points=("score_points", "mean"),
            mean_outcome_points=("outcome_points", "mean"),
            mean_total_points=("total_points", "mean"),
        )
    )


@dataclass
class MatchEnsemble:
    canonical_function: object
    importance_function: object
    poisson_model: object
    current_elo: dict[str, float]
    market_probabilities: dict[
        tuple[str, str], tuple[float, float, float]
    ] = field(default_factory=dict)
    market_weight: float = 0.35
    gradient_weight: float = 0.10
    outcome_weight: float = 0.20
    states: dict[str, TeamState] = field(default_factory=dict)
    home_goal_model: object | None = None
    away_goal_model: object | None = None
    outcome_model: object | None = None
    corners_model: object | None = None
    yellow_model: object | None = None
    red_model: object | None = None
    penalty_model: object | None = None

    def fit(self, results: pd.DataFrame, root: Path, reference_date: pd.Timestamp) -> None:
        feature_rows, self.states = build_rolling_features(
            results, self.importance_function
        )
        weights = _sample_weights(
            feature_rows["date"], feature_rows["importance"], reference_date
        )
        features = feature_rows[MATCH_FEATURES]

        goal_parameters = {
            "loss": "poisson",
            "learning_rate": 0.045,
            "max_iter": 260,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 35,
            "l2_regularization": 1.5,
            "random_state": 20260610,
        }
        self.home_goal_model = HistGradientBoostingRegressor(**goal_parameters)
        self.away_goal_model = HistGradientBoostingRegressor(**goal_parameters)
        self.home_goal_model.fit(features, feature_rows["home_score"], sample_weight=weights)
        self.away_goal_model.fit(features, feature_rows["away_score"], sample_weight=weights)

        self.outcome_model = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=280,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=2.0,
            random_state=20260610,
        )
        self.outcome_model.fit(features, feature_rows["outcome"], sample_weight=weights)
        self._fit_auxiliary_models(root, feature_rows)

    def _future_features(self, team_a: str, team_b: str) -> pd.DataFrame:
        home = self.canonical_function(team_a)
        away = self.canonical_function(team_b)
        home_state = self.states.get(home, TeamState())
        away_state = self.states.get(away, TeamState())
        home_state = TeamState(**vars(home_state))
        away_state = TeamState(**vars(away_state))
        home_state.elo = self.current_elo.get(home, home_state.elo)
        away_state.elo = self.current_elo.get(away, away_state.elo)
        if home in TOURNAMENT_HOSTS:
            home_state.elo += HOST_ELO_BONUS
        if away in TOURNAMENT_HOSTS:
            away_state.elo += HOST_ELO_BONUS
        row = _state_features(home_state, away_state, True, 3.0)
        return pd.DataFrame([row], columns=MATCH_FEATURES)

    def predict_core(self, team_a: str, team_b: str) -> dict[str, float]:
        features = self._future_features(team_a, team_b)
        poisson_home, poisson_away = self.poisson_model.expected_goals(team_a, team_b)
        gradient_home = float(self.home_goal_model.predict(features)[0])
        gradient_away = float(self.away_goal_model.predict(features)[0])
        expected_home = (
            (1.0 - self.gradient_weight) * poisson_home
            + self.gradient_weight * gradient_home
        )
        expected_away = (
            (1.0 - self.gradient_weight) * poisson_away
            + self.gradient_weight * gradient_away
        )
        expected_home = float(np.clip(expected_home, 0.15, 4.5))
        expected_away = float(np.clip(expected_away, 0.15, 4.5))

        from scipy.stats import poisson

        score_matrix = np.outer(
            poisson.pmf(np.arange(10), expected_home),
            poisson.pmf(np.arange(10), expected_away),
        )
        score_matrix /= score_matrix.sum()
        score_probabilities = np.array(
            [
                np.tril(score_matrix, -1).sum(),
                np.trace(score_matrix),
                np.triu(score_matrix, 1).sum(),
            ]
        )
        classifier_probabilities = self.outcome_model.predict_proba(features)[0]
        outcome_probabilities = (
            (1.0 - self.outcome_weight) * score_probabilities
            + self.outcome_weight * classifier_probabilities
        )
        market = self.market_probabilities.get(
            (
                self.canonical_function(team_a),
                self.canonical_function(team_b),
            )
        )
        if market is not None:
            outcome_probabilities = (
                (1.0 - self.market_weight) * outcome_probabilities
                + self.market_weight * np.asarray(market)
            )
        outcome_probabilities /= outcome_probabilities.sum()

        return {
            "features": features,
            "expected_home_goals": expected_home,
            "expected_away_goals": expected_away,
            "score_matrix": score_matrix,
            "home_probability": float(outcome_probabilities[0]),
            "draw_probability": float(outcome_probabilities[1]),
            "away_probability": float(outcome_probabilities[2]),
        }

    def predict_match(self, team_a: str, team_b: str) -> dict[str, float]:
        core = self.predict_core(team_a, team_b)
        auxiliary = self._auxiliary_features(
            core["features"],
            core["expected_home_goals"],
            core["expected_away_goals"],
            core["draw_probability"],
            knockout=False,
            round_code=0,
        )
        corner_mean = float(self.corners_model.predict(auxiliary)[0])
        yellow_mean = float(self.yellow_model.predict(auxiliary)[0])
        return {
            **{key: value for key, value in core.items() if key != "features"},
            "corner_mean": max(0.1, corner_mean),
            "yellow_mean": max(0.1, yellow_mean),
            "red_probability": float(self.red_model.predict_proba(auxiliary)[0, 1]),
        }

    def penalty_probability(
        self,
        team_a: str,
        team_b: str,
        expected_home: float,
        expected_away: float,
        draw_probability: float,
        round_code: int,
    ) -> float:
        features = self._future_features(team_a, team_b)
        auxiliary = self._auxiliary_features(
            features,
            expected_home,
            expected_away,
            draw_probability,
            knockout=True,
            round_code=round_code,
        )
        return float(self.penalty_model.predict_proba(auxiliary)[0, 1])

    @staticmethod
    def _auxiliary_features(
        match_features: pd.DataFrame,
        expected_home: float,
        expected_away: float,
        draw_probability: float,
        knockout: bool,
        round_code: int,
    ) -> pd.DataFrame:
        row = {
            "expected_home_goals": expected_home,
            "expected_away_goals": expected_away,
            "expected_total_goals": expected_home + expected_away,
            "expected_goal_difference": expected_home - expected_away,
            "absolute_elo_difference": abs(float(match_features["elo_difference"].iloc[0])),
            "draw_probability": draw_probability,
            "knockout": float(knockout),
            "round_code": float(round_code),
        }
        return pd.DataFrame([row], columns=AUXILIARY_FEATURES)

    def _fit_auxiliary_models(
        self, root: Path, international_features: pd.DataFrame
    ) -> None:
        corner_training = self._corner_training_data(root)
        self.corners_model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=220,
            max_leaf_nodes=12,
            min_samples_leaf=45,
            l2_regularization=2.0,
            random_state=20260610,
        )
        self.corners_model.fit(
            corner_training[AUXILIARY_FEATURES], corner_training["target"]
        )

        world_cup_training = self._world_cup_event_training(
            root, international_features
        )
        self.yellow_model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.045,
            max_iter=180,
            max_leaf_nodes=10,
            min_samples_leaf=18,
            l2_regularization=2.5,
            random_state=20260610,
        )
        self.yellow_model.fit(
            world_cup_training[AUXILIARY_FEATURES],
            world_cup_training["yellow_cards"],
        )

        red_base = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                C=0.35,
                max_iter=1000,
                random_state=20260610,
            ),
        )
        self.red_model = CalibratedClassifierCV(red_base, method="sigmoid", cv=4)
        self.red_model.fit(
            world_cup_training[AUXILIARY_FEATURES],
            world_cup_training["has_red_card"],
        )

        knockout_training = world_cup_training[world_cup_training["knockout"] == 1]
        penalty_base = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                C=0.3,
                max_iter=1000,
                random_state=20260610,
            ),
        )
        self.penalty_model = CalibratedClassifierCV(
            penalty_base, method="sigmoid", cv=4
        )
        self.penalty_model.fit(
            knockout_training[AUXILIARY_FEATURES],
            knockout_training["penalty_shootout"],
        )

    @staticmethod
    def _corner_training_data(root: Path) -> pd.DataFrame:
        club = pd.read_csv(root / "data/club_corners_cards_2016_2025.csv")
        numeric = ["FTHG", "FTAG", "HC", "AC"]
        club[numeric] = club[numeric].apply(pd.to_numeric, errors="coerce")
        club = club.dropna(subset=numeric).copy()
        club["date"] = pd.to_datetime(
            club["Date"], format="mixed", dayfirst=True, errors="coerce"
        )
        club = club.dropna(subset=["date"]).sort_values("date")

        states: dict[str, TeamState] = {}
        rows = []
        for match in club.itertuples():
            home_state = states.setdefault(match.HomeTeam, TeamState())
            away_state = states.setdefault(match.AwayTeam, TeamState())
            expected_home = (
                home_state.attack + away_state.defense
            ) / 2.0
            expected_away = (
                away_state.attack + home_state.defense
            ) / 2.0
            expected_difference = expected_home - expected_away
            rows.append(
                {
                    "expected_home_goals": expected_home,
                    "expected_away_goals": expected_away,
                    "expected_total_goals": expected_home + expected_away,
                    "expected_goal_difference": expected_difference,
                    "absolute_elo_difference": abs(expected_difference) * 180.0,
                    "draw_probability": 0.32 * math.exp(-abs(expected_difference)),
                    "knockout": 0.0,
                    "round_code": 0.0,
                    "target": match.HC + match.AC,
                    "date": match.date,
                }
            )
            _update_state(home_state, match.FTHG, match.FTAG)
            _update_state(away_state, match.FTAG, match.FTHG)
        return pd.DataFrame(rows)

    def _world_cup_event_training(
        self, root: Path, international_features: pd.DataFrame
    ) -> pd.DataFrame:
        matches = pd.read_csv(
            root / "data/worldcup/data-csv/matches.csv",
            parse_dates=["match_date"],
        )
        matches = matches[matches["tournament_name"].str.contains("Men's")].copy()
        bookings = pd.read_csv(root / "data/worldcup/data-csv/bookings.csv")
        booking_totals = bookings.groupby("match_id").agg(
            yellow_cards=("yellow_card", "sum"),
            has_red_card=("sending_off", lambda values: int(values.sum() > 0)),
        )
        matches = matches.merge(booking_totals, on="match_id", how="left")
        matches[["yellow_cards", "has_red_card"]] = matches[
            ["yellow_cards", "has_red_card"]
        ].fillna(0)
        matches["home_team"] = matches["home_team_name"].map(self.canonical_function)
        matches["away_team"] = matches["away_team_name"].map(self.canonical_function)
        matches["date"] = matches["match_date"]

        feature_lookup = international_features.copy()
        merged = matches.merge(
            feature_lookup,
            on=["date", "home_team", "away_team"],
            how="inner",
            suffixes=("", "_feature"),
        )
        if len(merged) < 250:
            raise RuntimeError("Too few World Cup matches matched to rolling features")

        home_lambda = np.maximum(0.2, merged["home_attack"] * merged["away_defense"] / INITIAL_GOALS)
        away_lambda = np.maximum(0.2, merged["away_attack"] * merged["home_defense"] / INITIAL_GOALS)
        elo_gap = merged["elo_difference"]
        draw_probability = 0.32 * np.exp(-abs(elo_gap) / 350.0)
        stage = merged["stage_name"].str.lower()
        round_code = np.select(
            [
                stage.str.contains("final") & ~stage.str.contains("semi|quarter|third"),
                stage.str.contains("semi"),
                stage.str.contains("quarter"),
                stage.str.contains("round of 16|second round"),
            ],
            [5, 4, 3, 2],
            default=1,
        )
        return pd.DataFrame(
            {
                "expected_home_goals": home_lambda,
                "expected_away_goals": away_lambda,
                "expected_total_goals": home_lambda + away_lambda,
                "expected_goal_difference": home_lambda - away_lambda,
                "absolute_elo_difference": abs(elo_gap),
                "draw_probability": draw_probability,
                "knockout": merged["knockout_stage"].astype(float),
                "round_code": round_code.astype(float),
                "yellow_cards": merged["yellow_cards"].astype(float),
                "has_red_card": merged["has_red_card"].astype(int),
                "penalty_shootout": merged["penalty_shootout"].astype(int),
                "date": merged["date"],
            }
        )


def auxiliary_backtest(model: MatchEnsemble, root: Path) -> pd.DataFrame:
    records = []
    corner_data = model._corner_training_data(root)
    corner_train = corner_data[corner_data["date"] < "2023-07-01"]
    corner_test = corner_data[corner_data["date"] >= "2023-07-01"]
    corner_model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=12,
        min_samples_leaf=45,
        l2_regularization=2.0,
        random_state=20260610,
    )
    corner_model.fit(corner_train[AUXILIARY_FEATURES], corner_train["target"])
    corner_predictions = np.rint(
        corner_model.predict(corner_test[AUXILIARY_FEATURES])
    ).astype(int)
    corner_actual = corner_test["target"].to_numpy()
    corner_points = np.where(
        corner_predictions == corner_actual,
        10,
        np.where(abs(corner_predictions - corner_actual) <= 2, 5, 0),
    )
    records.append(
        {
            "target": "corners",
            "test_matches": len(corner_test),
            "mean_competition_points": float(corner_points.mean()),
            "accuracy": float((corner_predictions == corner_actual).mean()),
        }
    )

    international_matches = (
        pd.read_csv(root / "data/results.csv", parse_dates=["date"])
        .dropna(subset=["home_score", "away_score"])
        .query("date >= '2006-01-01'")
        .copy()
    )
    international_matches["home_team"] = international_matches["home_team"].map(
        model.canonical_function
    )
    international_matches["away_team"] = international_matches["away_team"].map(
        model.canonical_function
    )
    international_rows, _ = build_rolling_features(
        international_matches, model.importance_function
    )
    events = model._world_cup_event_training(root, international_rows)
    event_train = events[events["date"] < "2018-01-01"]
    event_test = events[events["date"] >= "2018-01-01"]

    yellow_model = HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.045,
        max_iter=180,
        max_leaf_nodes=10,
        min_samples_leaf=18,
        l2_regularization=2.5,
        random_state=20260610,
    )
    yellow_model.fit(
        event_train[AUXILIARY_FEATURES], event_train["yellow_cards"]
    )
    yellow_predictions = np.rint(
        yellow_model.predict(event_test[AUXILIARY_FEATURES])
    ).astype(int)
    yellow_actual = event_test["yellow_cards"].to_numpy()
    yellow_points = np.where(
        yellow_predictions == yellow_actual,
        10,
        np.where(abs(yellow_predictions - yellow_actual) <= 1, 5, 0),
    )
    records.append(
        {
            "target": "yellow_cards",
            "test_matches": len(event_test),
            "mean_competition_points": float(yellow_points.mean()),
            "accuracy": float((yellow_predictions == yellow_actual).mean()),
        }
    )

    for target, subset in (
        ("red_cards", event_test),
        ("penalties", event_test[event_test["knockout"] == 1]),
    ):
        column = "has_red_card" if target == "red_cards" else "penalty_shootout"
        probability = event_train[column].mean()
        prediction = int(probability >= 0.5)
        actual = subset[column].astype(int).to_numpy()
        points = 5 * (actual == prediction)
        records.append(
            {
                "target": target,
                "test_matches": len(subset),
                "mean_competition_points": float(points.mean()),
                "accuracy": float((actual == prediction).mean()),
            }
        )
    return pd.DataFrame(records)
