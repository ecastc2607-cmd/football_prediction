"""
Calcula la fuerza de ataque/defensa de cada equipo a partir de partidos finalizados,
siguiendo el método estándar de modelos de goles (Maher 1982 / base de Dixon-Coles):

  fuerza_ataque_local(equipo)  = (goles marcados en casa por el equipo / partidos en casa)
                                  / promedio_liga_goles_local
  fuerza_defensa_local(equipo) = (goles recibidos en casa por el equipo / partidos en casa)
                                  / promedio_liga_goles_visitante
  (y de forma simétrica para visitante)

Estas fuerzas son los insumos del modelo de Poisson en poisson_model.py.
"""
from __future__ import annotations

import pandas as pd

from . import config

MIN_MATCHES_WARNING = 5  # bajo esta cantidad de partidos, la fuerza calculada es poco fiable


def _load_season_file(competition_code: str, season: int) -> pd.DataFrame:
    path = config.PROCESSED_DIR / f"matches_{competition_code}_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corre primero: "
            f"python -m src.fetch_football_data --comp {competition_code} --season {season}"
        )
    return pd.read_csv(path)


def load_finished_matches(competition_code: str, seasons: list[int]) -> pd.DataFrame:
    """Carga y combina los partidos finalizados de una o varias temporadas.

    Combinar temporadas (ej. [2026, 2025]) sirve como respaldo al inicio de una
    temporada nueva, cuando todavía hay pocos partidos jugados para calcular
    fuerzas de equipo fiables a partir solo de la temporada en curso.
    """
    frames = [_load_season_file(competition_code, s) for s in seasons]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"] == "FINISHED"].dropna(subset=["home_goals", "away_goals"]).copy()
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    return df


def _shrink(rate: pd.Series, played: pd.Series, league_avg: float, prior_games: int) -> pd.Series:
    """Encoge una tasa promedio (goles/partido) hacia el promedio de liga cuando `played` es
    chico, con `prior_games` partidos "fantasma" al promedio de liga como peso del prior.
    Con muchos partidos jugados, el dato real domina; con pocos (ej. un recién ascendido en
    sus primeras jornadas), evita que la estimación colapse a extremos como 0.0.
    """
    return (rate.fillna(0) * played + league_avg * prior_games) / (played + prior_games)


def build_team_strength(matches: pd.DataFrame, prior_games: int = 3) -> pd.DataFrame:
    """Devuelve un DataFrame indexado por equipo con sus 4 fuerzas + partidos jugados.

    `prior_games`: cuántos "partidos fantasma" al promedio de liga se mezclan en la
    estimación de cada equipo (suavizado bayesiano simple). 0 = sin suavizado.
    """
    if matches.empty:
        raise ValueError("No hay partidos finalizados para calcular fuerzas de equipo todavía.")

    league_home_avg = matches["home_goals"].mean()
    league_away_avg = matches["away_goals"].mean()

    home = matches.groupby("home_team").agg(
        home_played=("home_goals", "count"),
        home_goals_for=("home_goals", "mean"),
        home_goals_against=("away_goals", "mean"),
    )
    away = matches.groupby("away_team").agg(
        away_played=("away_goals", "count"),
        away_goals_for=("away_goals", "mean"),
        away_goals_against=("home_goals", "mean"),
    )

    teams = home.join(away, how="outer").fillna(0)

    home_goals_for = _shrink(teams["home_goals_for"], teams["home_played"], league_home_avg, prior_games)
    home_goals_against = _shrink(teams["home_goals_against"], teams["home_played"], league_away_avg, prior_games)
    away_goals_for = _shrink(teams["away_goals_for"], teams["away_played"], league_away_avg, prior_games)
    away_goals_against = _shrink(teams["away_goals_against"], teams["away_played"], league_home_avg, prior_games)

    teams["attack_home"] = home_goals_for / league_home_avg
    teams["defense_home"] = home_goals_against / league_away_avg
    teams["attack_away"] = away_goals_for / league_away_avg
    teams["defense_away"] = away_goals_against / league_home_avg

    teams.attrs["league_home_avg"] = league_home_avg
    teams.attrs["league_away_avg"] = league_away_avg
    return teams


def team_strength_for_competition(competition_code: str, seasons: list[int]) -> pd.DataFrame:
    matches = load_finished_matches(competition_code, seasons)
    return build_team_strength(matches)


def confidence_note(team_strength: pd.DataFrame, home_team: str, away_team: str) -> str | None:
    """Devuelve una advertencia si alguno de los dos equipos no tiene (o casi no tiene)
    partidos propios como local/visitante en las temporadas cargadas: en ese caso su
    fuerza quedó casi 100% en el promedio de liga por el suavizado, no por datos reales.
    """
    home_played = team_strength.loc[home_team, "home_played"] if home_team in team_strength.index else 0
    away_played = team_strength.loc[away_team, "away_played"] if away_team in team_strength.index else 0
    flags = []
    if home_played < MIN_MATCHES_WARNING:
        flags.append(f"{home_team} solo tiene {int(home_played)} partido(s) como local en los datos cargados")
    if away_played < MIN_MATCHES_WARNING:
        flags.append(f"{away_team} solo tiene {int(away_played)} partido(s) como visitante en los datos cargados")
    if not flags:
        return None
    return "⚠ Baja confianza: " + "; ".join(flags) + " — el modelo se apoya casi todo en el promedio de liga."
