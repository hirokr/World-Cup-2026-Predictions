from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "comp-notebook"
FIGURES_DIR = ROOT / "figures"

COLORS = {
    "blue": "#2563eb",
    "green": "#16a34a",
    "orange": "#ea580c",
    "slate": "#475569",
    "navy": "#081426",
    "panel": "#12233f",
    "gold": "#f4c95d",
    "white": "#f8fafc",
    "muted": "#94a3b8",
}


def save_figure(figure: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_model_selection() -> None:
    trials = pd.read_csv(RESULTS_DIR / "model_backtest.csv")
    pivot = trials.pivot(
        index="half_life_years",
        columns="alpha",
        values="mean_backtest_points",
    )

    figure, axis = plt.subplots(figsize=(8, 4.8))
    image = axis.imshow(pivot, cmap="Blues", aspect="auto")
    for row_index, half_life in enumerate(pivot.index):
        for column_index, alpha in enumerate(pivot.columns):
            value = pivot.loc[half_life, alpha]
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > pivot.to_numpy().mean() else "black",
                fontsize=9,
            )
    axis.set_xticks(
        range(len(pivot.columns)), [f"{value:g}" for value in pivot.columns]
    )
    axis.set_yticks(
        range(len(pivot.index)), [f"{value:g}" for value in pivot.index]
    )
    axis.set_xlabel("Poisson regularization (alpha)")
    axis.set_ylabel("Recency half-life (years)")
    axis.set_title("Historical World Cup backtest score")
    figure.colorbar(image, ax=axis, label="Mean competition points per match")
    save_figure(figure, "model_selection.png")


def plot_ensemble_backtest() -> None:
    backtest = pd.read_csv(RESULTS_DIR / "ensemble_backtest.csv")
    summary = (
        backtest.groupby(["gradient_weight", "outcome_weight"], as_index=False)[
            "mean_total_points"
        ]
        .mean()
        .sort_values("mean_total_points", ascending=False)
        .head(12)
    )
    labels = [
        f"Goals {gradient:.2f}, outcome {outcome:.2f}"
        for gradient, outcome in zip(
            summary["gradient_weight"], summary["outcome_weight"]
        )
    ]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(len(summary))
    values = summary["mean_total_points"]
    lower_bound = float(values.min()) - 0.15
    axis.barh(
        positions,
        values - lower_bound,
        left=lower_bound,
        color=COLORS["blue"],
    )
    for position, value in zip(positions, values):
        axis.text(value + 0.005, position, f"{value:.3f}", va="center", fontsize=8)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(lower_bound, float(values.max()) + 0.12)
    axis.set_xlabel("Mean competition points per match")
    axis.set_title("Top ensemble blends across validation tournaments")
    axis.grid(axis="x", alpha=0.25)
    save_figure(figure, "ensemble_backtest.png")


def plot_prediction_summary() -> None:
    group = pd.read_csv(RESULTS_DIR / "group_predictions.csv")
    knockout = pd.read_csv(RESULTS_DIR / "knockout_predictions.csv")
    all_predictions = pd.concat([group, knockout], ignore_index=True)
    all_predictions["total_goals"] = (
        all_predictions["predicted_home_goals"]
        + all_predictions["predicted_away_goals"]
    )

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    score_counts = all_predictions["total_goals"].value_counts().sort_index()
    axes[0].bar(
        score_counts.index.astype(str),
        score_counts.values,
        color=COLORS["green"],
    )
    axes[0].set_xlabel("Predicted total goals")
    axes[0].set_ylabel("Matches")
    axes[0].set_title("Predicted scoring distribution")
    axes[0].grid(axis="y", alpha=0.25)

    auxiliary = pd.read_csv(RESULTS_DIR / "auxiliary_backtest.csv")
    axes[1].bar(
        auxiliary["target"],
        auxiliary["accuracy"] * 100,
        color=[
            COLORS["blue"],
            COLORS["orange"],
            COLORS["slate"],
            COLORS["green"],
        ],
    )
    axes[1].set_ylabel("Exact classification/count accuracy (%)")
    axes[1].set_title("Auxiliary target backtests")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    save_figure(figure, "prediction_summary.png")


def predicted_winner(match: pd.Series) -> str:
    return (
        match["predicted_home_team"]
        if match["match_winner"] == "home"
        else match["predicted_away_team"]
    )


def plot_knockout_bracket() -> None:
    knockout = pd.read_csv(RESULTS_DIR / "knockout_predictions.csv")
    championship = knockout[knockout["round"] != "Third-place playoff"].copy()
    championship["winner"] = championship.apply(predicted_winner, axis=1)

    round_order = [
        "Round of 32",
        "Round of 16",
        "Quarter-final",
        "Semi-final",
        "Final",
    ]
    round_x = dict(zip(round_order, [1.8, 7.0, 12.2, 17.4, 22.6]))
    positions: dict[int, tuple[float, float]] = {}

    first_round = championship[championship["round"] == "Round of 32"]
    first_round_y = np.linspace(15.4, 0.6, len(first_round))
    for match_id, y_position in zip(first_round["match_id"], first_round_y):
        positions[int(match_id)] = (round_x["Round of 32"], float(y_position))

    for round_name in round_order[1:]:
        for match in championship[championship["round"] == round_name].itertuples():
            source_ids = []
            for slot in (match.slot_home, match.slot_away):
                if str(slot).startswith("Winner Match "):
                    source_ids.append(int(str(slot).rsplit(" ", 1)[1]))
            y_position = float(np.mean([positions[source][1] for source in source_ids]))
            positions[int(match.match_id)] = (round_x[round_name], y_position)

    figure, axis = plt.subplots(figsize=(28, 17), facecolor=COLORS["navy"])
    axis.set_facecolor(COLORS["navy"])
    card_width = 4.25
    card_height = 0.72

    for match in championship.itertuples():
        match_id = int(match.match_id)
        x_position, y_position = positions[match_id]
        for slot in (match.slot_home, match.slot_away):
            if not str(slot).startswith("Winner Match "):
                continue
            source_id = int(str(slot).rsplit(" ", 1)[1])
            source_x, source_y = positions[source_id]
            start_x = source_x + card_width / 2
            end_x = x_position - card_width / 2
            middle_x = (start_x + end_x) / 2
            axis.plot(
                [start_x, middle_x, middle_x, end_x],
                [source_y, source_y, y_position, y_position],
                color="#3c5578",
                linewidth=1.2,
                zorder=1,
            )

    for match in championship.itertuples():
        match_id = int(match.match_id)
        x_position, y_position = positions[match_id]
        winner = match.predicted_home_team if match.match_winner == "home" else match.predicted_away_team
        card = FancyBboxPatch(
            (x_position - card_width / 2, y_position - card_height / 2),
            card_width,
            card_height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=COLORS["panel"],
            edgecolor=COLORS["gold"] if match.round == "Final" else "#355176",
            linewidth=2.0 if match.round == "Final" else 1.0,
            zorder=2,
        )
        axis.add_patch(card)

        home_color = COLORS["gold"] if winner == match.predicted_home_team else COLORS["white"]
        away_color = COLORS["gold"] if winner == match.predicted_away_team else COLORS["white"]
        axis.text(
            x_position - 1.9,
            y_position + 0.15,
            match.predicted_home_team,
            color=home_color,
            fontsize=7.2,
            fontweight="bold" if winner == match.predicted_home_team else "normal",
            ha="left",
            va="center",
            zorder=3,
        )
        axis.text(
            x_position - 1.9,
            y_position - 0.15,
            match.predicted_away_team,
            color=away_color,
            fontsize=7.2,
            fontweight="bold" if winner == match.predicted_away_team else "normal",
            ha="left",
            va="center",
            zorder=3,
        )
        axis.text(
            x_position + 1.82,
            y_position + 0.15,
            str(match.predicted_home_goals),
            color=home_color,
            fontsize=8,
            fontweight="bold",
            ha="right",
            va="center",
            zorder=3,
        )
        axis.text(
            x_position + 1.82,
            y_position - 0.15,
            str(match.predicted_away_goals),
            color=away_color,
            fontsize=8,
            fontweight="bold",
            ha="right",
            va="center",
            zorder=3,
        )

    for round_name, x_position in round_x.items():
        axis.text(
            x_position,
            16.25,
            round_name.upper(),
            color=COLORS["muted"],
            fontsize=10,
            fontweight="bold",
            ha="center",
        )

    final = championship[championship["round"] == "Final"].iloc[0]
    champion = predicted_winner(final)
    axis.text(
        26.0,
        positions[int(final["match_id"])][1] + 0.3,
        "PREDICTED CHAMPION",
        color=COLORS["muted"],
        fontsize=10,
        fontweight="bold",
        ha="center",
    )
    axis.text(
        26.0,
        positions[int(final["match_id"])][1] - 0.2,
        champion,
        color=COLORS["gold"],
        fontsize=22,
        fontweight="bold",
        ha="center",
    )

    third_place = knockout[knockout["round"] == "Third-place playoff"].iloc[0]
    axis.text(
        22.6,
        0.25,
        (
            f"THIRD PLACE: {predicted_winner(third_place)}  "
            f"{third_place.predicted_home_goals}-{third_place.predicted_away_goals}"
        ),
        color=COLORS["muted"],
        fontsize=8,
        ha="center",
    )
    axis.text(
        0.0,
        17.1,
        "FIFA WORLD CUP 2026 - PREDICTED KNOCKOUT BRACKET",
        color=COLORS["white"],
        fontsize=23,
        fontweight="bold",
        ha="left",
    )
    axis.text(
        0.0,
        16.65,
        "Winning teams are highlighted in gold",
        color=COLORS["muted"],
        fontsize=10,
        ha="left",
    )
    axis.set_xlim(-0.6, 28.2)
    axis.set_ylim(-0.2, 17.5)
    axis.axis("off")
    save_figure(figure, "world_cup_2026_predicted_bracket.png")


def optimized_group_qualifiers(knockout: pd.DataFrame) -> dict[str, dict[str, str]]:
    qualifiers: dict[str, dict[str, str]] = {
        group: {} for group in "ABCDEFGHIJKL"
    }
    first_round = knockout[knockout["round"] == "Round of 32"]
    for match in first_round.itertuples():
        for slot, team in (
            (match.slot_home, match.predicted_home_team),
            (match.slot_away, match.predicted_away_team),
        ):
            if str(slot).startswith("Winner Group "):
                qualifiers[str(slot)[-1]]["winner"] = team
            elif str(slot).startswith("Runner-up Group "):
                qualifiers[str(slot)[-1]]["runner_up"] = team
    return qualifiers


def plot_group_winners() -> None:
    knockout = pd.read_csv(RESULTS_DIR / "knockout_predictions.csv")
    qualifiers = optimized_group_qualifiers(knockout)
    figure, axes = plt.subplots(
        3,
        4,
        figsize=(18, 10),
        facecolor=COLORS["navy"],
    )
    for axis, group in zip(axes.flat, "ABCDEFGHIJKL"):
        winner = qualifiers[group]["winner"]
        runner_up = qualifiers[group]["runner_up"]
        axis.set_facecolor(COLORS["panel"])
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 3.2)
        axis.axis("off")
        axis.text(
            0.05,
            2.75,
            f"GROUP {group}",
            color=COLORS["gold"],
            fontsize=14,
            fontweight="bold",
        )
        axis.add_patch(
            FancyBboxPatch(
                (0.03, 1.35),
                0.94,
                0.72,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor="#2c3c39",
                edgecolor=COLORS["gold"],
                linewidth=1.2,
            )
        )
        axis.text(0.08, 1.72, "1", color=COLORS["white"], fontsize=10, va="center")
        axis.text(
            0.17,
            1.72,
            winner,
            color=COLORS["gold"],
            fontsize=12,
            fontweight="bold",
            va="center",
        )
        axis.text(
            0.08,
            0.72,
            "2",
            color=COLORS["white"],
            fontsize=10,
            va="center",
        )
        axis.text(
            0.17,
            0.72,
            runner_up,
            color=COLORS["white"],
            fontsize=11,
            fontweight="bold",
            va="center",
        )

    figure.suptitle(
        "FIFA WORLD CUP 2026 - PREDICTED GROUP QUALIFIERS",
        color=COLORS["white"],
        fontsize=23,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.955,
        "Exact winners and runners-up used by the optimized knockout bracket",
        color=COLORS["muted"],
        fontsize=10,
        ha="center",
    )
    save_figure(figure, "world_cup_2026_group_winners.png")


def plot_knockout_wins() -> None:
    knockout = pd.read_csv(RESULTS_DIR / "knockout_predictions.csv")
    knockout["winner"] = knockout.apply(predicted_winner, axis=1)
    win_counts = knockout["winner"].value_counts().sort_values()
    colors = [
        COLORS["gold"] if team == "Spain" else COLORS["blue"]
        for team in win_counts.index
    ]

    figure, axis = plt.subplots(figsize=(11, 7), facecolor=COLORS["navy"])
    axis.set_facecolor(COLORS["navy"])
    bars = axis.barh(win_counts.index, win_counts.values, color=colors)
    for bar, value in zip(bars, win_counts.values):
        axis.text(
            value + 0.05,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            color=COLORS["white"],
            va="center",
            fontweight="bold",
        )
    axis.set_title(
        "PREDICTED KNOCKOUT MATCH WINS",
        color=COLORS["white"],
        fontsize=20,
        fontweight="bold",
        pad=18,
    )
    axis.set_xlabel("Matches won", color=COLORS["muted"])
    axis.tick_params(colors=COLORS["white"])
    axis.grid(axis="x", color="#334155", alpha=0.45)
    for spine in axis.spines.values():
        spine.set_visible(False)
    save_figure(figure, "world_cup_2026_knockout_wins.png")


def main() -> None:
    plot_model_selection()
    plot_ensemble_backtest()
    plot_prediction_summary()
    plot_knockout_bracket()
    plot_group_winners()
    plot_knockout_wins()
    print(f"Wrote visualizations to {FIGURES_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
