from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
ELO_URL = "https://eloratings.net/World.tsv"
SQUAD_URL = "https://fdp.fifa.org/assetspublic/ce281/pdf/SquadLists-English.pdf"


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


def main() -> None:
    download(ELO_URL, ROOT / "data/world_elo_2026-06-10.tsv")
    squad_pdf = ROOT / "data/worldcup/squads_2026.pdf"
    squad_text = ROOT / "data/worldcup/squads_2026.txt"
    download(SQUAD_URL, squad_pdf)
    subprocess.run(
        ["pdftotext", "-layout", str(squad_pdf), str(squad_text)],
        check=True,
    )

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
