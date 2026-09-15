"""
Predice los próximos partidos programados de una competición usando el modelo de Poisson.

Las fuerzas de equipo se calculan combinando la temporada en curso con la anterior
(por defecto), para que las predicciones no dependan solo de los pocos partidos
jugados al arrancar una temporada nueva.

Uso:
    python -m src.predict_matchday --comp PL --season 2026
    python -m src.predict_matchday --comp CL --season 2026 --matchday 2
    python -m src.predict_matchday --comp PL --season 2026 --seasons-back 0  # solo temporada actual
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import config
from .cross_competition_strength import fill_missing_with_domestic_strength
from .poisson_model import predict_match
from .team_strength import confidence_note, team_strength_for_competition


def upcoming_fixtures(competition_code: str, season: int, matchday: int | None = None) -> pd.DataFrame:
    path = config.PROCESSED_DIR / f"matches_{competition_code}_{season}.csv"
    df = pd.read_csv(path)
    df = df[df["status"].isin(["SCHEDULED", "TIMED"])]
    if matchday is not None:
        df = df[df["matchday"] == matchday]
    return df.sort_values(["matchday", "utc_date"])


def finished_fixtures(competition_code: str, season: int, matchday: int | None = None) -> pd.DataFrame:
    path = config.PROCESSED_DIR / f"matches_{competition_code}_{season}.csv"
    df = pd.read_csv(path)
    df = df[df["status"] == "FINISHED"]
    if matchday is not None:
        df = df[df["matchday"] == matchday]
    return df.sort_values(["matchday", "utc_date"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comp", required=True, help="Código de competición, ej. PL")
    parser.add_argument("--season", type=int, required=True,
                         help="Año de inicio de la temporada en curso, ej. 2026 para 2026/27.")
    parser.add_argument("--matchday", type=int, default=None, help="Filtra por jornada")
    parser.add_argument("--seasons-back", type=int, default=1,
                         help="Cuántas temporadas anteriores sumar al calcular fuerzas de equipo "
                              "(0 = solo la temporada actual). Default: 1.")
    args = parser.parse_args()

    seasons = [args.season - i for i in range(args.seasons_back + 1)]
    strength = team_strength_for_competition(args.comp, seasons)
    fixtures = upcoming_fixtures(args.comp, args.season, args.matchday)

    if fixtures.empty:
        print("No hay partidos programados que coincidan con el filtro.")
        return

    if args.comp.upper() in config.CUP_STYLE_COMPETITIONS:
        equipos = set(fixtures["home_team"]) | set(fixtures["away_team"])
        strength, _ = fill_missing_with_domestic_strength(strength, equipos, seasons)

    comp_name = config.COMPETITIONS.get(args.comp, args.comp)
    print(f"\n== Predicciones · {comp_name} · fuerzas calculadas con temporadas {seasons} ==\n")
    for _, row in fixtures.iterrows():
        try:
            pred = predict_match(strength, row["home_team"], row["away_team"])
        except KeyError as e:
            print(f"(omitido) {row['home_team']} vs {row['away_team']}: {e}")
            continue
        print(pred.summary())
        note = confidence_note(strength, row["home_team"], row["away_team"])
        if note:
            print(f"  {note}")
        print()


if __name__ == "__main__":
    main()
