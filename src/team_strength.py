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

# Vida media del peso de un partido (días). None = sin decaimiento (todos pesan
# igual). Probado con model_eval.py (oct-2026, 236 partidos): 365/180/90 días
# EMPEORARON el log-loss de 1X2 (0.993 -> 0.994/0.998/1.009) — con solo la
# temporada actual + la anterior cargadas, restarle peso a la anterior deja al
# modelo con muy poca muestra. Queda apagado; re-evaluar con más temporadas.
HALF_LIFE_DAYS: float | None = None


def _load_season_file(competition_code: str, season: int) -> pd.DataFrame:
    path = config.matches_path(competition_code, season)
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
    # Un mismo partido puede venir en dos archivos (ej. Nations League: el
    # historial de cada selección incluye la edición en curso, que ya está en el
    # archivo de la temporada actual) — sin esto contaba doble. Se conserva la
    # primera aparición: `seasons` viene con la temporada más reciente primero.
    if "match_id" in df.columns:
        df = df.drop_duplicates(subset="match_id", keep="first")
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


def match_weights(matches: pd.DataFrame, half_life_days: float | None,
                  as_of: pd.Timestamp | None = None) -> pd.Series:
    """Peso de cada partido: 1 para uno jugado `as_of`, 0.5 para uno de hace
    `half_life_days`, 0.25 para el doble, etc. (decaimiento temporal, como en
    Dixon & Coles 1997). None = todos pesan igual."""
    if not half_life_days:
        return pd.Series(1.0, index=matches.index)
    fechas = pd.to_datetime(matches["utc_date"], utc=True, errors="coerce")
    ref = as_of if as_of is not None else fechas.max()
    edad = ((ref - fechas).dt.total_seconds() / 86400).clip(lower=0)
    return (0.5 ** (edad / half_life_days)).fillna(0.5)


def build_team_strength(matches: pd.DataFrame, prior_games: int = 3,
                        half_life_days: float | None = None,
                        as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Devuelve un DataFrame indexado por equipo con sus 4 fuerzas + partidos jugados.

    `prior_games`: cuántos "partidos fantasma" al promedio de liga se mezclan en la
    estimación de cada equipo (suavizado bayesiano simple). 0 = sin suavizado.
    `half_life_days`: si se da, los partidos viejos pesan menos (ver match_weights);
    por defecto, HALF_LIFE_DAYS. `as_of`: fecha desde la que se mide la antigüedad
    (el inicio del partido a predecir, en un backtest); por defecto, el último partido.

    home_played/away_played siguen siendo CONTEOS reales (los usa confidence_note);
    el suavizado usa en cambio el peso efectivo (home_weight/away_weight): un equipo
    con muchos partidos pero todos viejos se acerca más al promedio.
    """
    if matches.empty:
        raise ValueError("No hay partidos finalizados para calcular fuerzas de equipo todavía.")

    if half_life_days is None:
        half_life_days = HALF_LIFE_DAYS
    m = matches.assign(_w=match_weights(matches, half_life_days, as_of))
    m["_w_hg"] = m["_w"] * m["home_goals"]
    m["_w_ag"] = m["_w"] * m["away_goals"]

    league_home_avg = m["_w_hg"].sum() / m["_w"].sum()
    league_away_avg = m["_w_ag"].sum() / m["_w"].sum()

    home = m.groupby("home_team").agg(
        home_played=("home_goals", "count"), home_weight=("_w", "sum"),
        _hgf=("_w_hg", "sum"), _hga=("_w_ag", "sum"),
    )
    home["home_goals_for"] = home["_hgf"] / home["home_weight"]
    home["home_goals_against"] = home["_hga"] / home["home_weight"]
    away = m.groupby("away_team").agg(
        away_played=("away_goals", "count"), away_weight=("_w", "sum"),
        _agf=("_w_ag", "sum"), _aga=("_w_hg", "sum"),
    )
    away["away_goals_for"] = away["_agf"] / away["away_weight"]
    away["away_goals_against"] = away["_aga"] / away["away_weight"]

    teams = home.drop(columns=["_hgf", "_hga"]).join(away.drop(columns=["_agf", "_aga"]), how="outer").fillna(0)

    home_goals_for = _shrink(teams["home_goals_for"], teams["home_weight"], league_home_avg, prior_games)
    home_goals_against = _shrink(teams["home_goals_against"], teams["home_weight"], league_away_avg, prior_games)
    away_goals_for = _shrink(teams["away_goals_for"], teams["away_weight"], league_away_avg, prior_games)
    away_goals_against = _shrink(teams["away_goals_against"], teams["away_weight"], league_home_avg, prior_games)

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
