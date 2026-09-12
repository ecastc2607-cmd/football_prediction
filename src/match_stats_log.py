"""
Acumula, para siempre, cada resultado real de corners/faltas/tarjetas que
consultamos en Highlightly — sin importar si fue desde la vista en vivo o desde
otro lado. La idea: no hay cuota para reconstruir el histórico de golpe (100
peticiones/día), pero si cada consulta real queda guardada, en unas semanas hay
suficiente muestra por equipo para promedios propios.

data/tracking/match_stats_log.csv — una fila por equipo por partido.
"""
from __future__ import annotations

import pandas as pd

from . import config

LOG_PATH = config.ROOT_DIR / "data" / "tracking" / "match_stats_log.csv"

COLUMNS = [
    "competition", "season", "matchday", "utc_date",
    "home_team", "away_team", "team", "is_home",
    "corners", "faltas", "tarjetas_amarillas", "tarjetas_rojas",
]

MIN_SAMPLE_FOR_CONFIDENCE = 5  # partidos mínimos por equipo antes de confiar en el promedio


def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(LOG_PATH)


def log_match_stats(competition: str, season: int, matchday: int, utc_date: str,
                     home_team: str, away_team: str, stats: dict) -> None:
    """Guarda las stats de un partido (dict como lo devuelve highlightly_client:
    {nombre_equipo: {corners, faltas, tarjetas_amarillas, tarjetas_rojas}}).
    No duplica si ese partido ya estaba guardado.
    """
    if not stats:
        return
    log = load_log()
    key_mask = (
        (log["competition"] == competition) & (log["season"] == season)
        & (log["home_team"] == home_team) & (log["away_team"] == away_team)
    ) if not log.empty else pd.Series(dtype=bool)
    if key_mask.any():
        return  # ya lo teníamos

    rows = []
    for team_name, team_stats in stats.items():
        is_home = team_name == home_team or team_name in home_team or home_team in team_name
        rows.append({
            "competition": competition, "season": season, "matchday": matchday,
            "utc_date": utc_date, "home_team": home_team, "away_team": away_team,
            "team": team_name, "is_home": is_home,
            "corners": team_stats.get("corners"),
            "faltas": team_stats.get("faltas"),
            "tarjetas_amarillas": team_stats.get("tarjetas_amarillas"),
            "tarjetas_rojas": team_stats.get("tarjetas_rojas"),
        })
    if not rows:
        return

    config.ROOT_DIR.joinpath("data", "tracking").mkdir(parents=True, exist_ok=True)
    log = pd.concat([log, pd.DataFrame(rows)], ignore_index=True)
    log.to_csv(LOG_PATH, index=False, encoding="utf-8")


def team_averages(competition: str | None = None) -> pd.DataFrame:
    """Promedio de corners/faltas/tarjetas por equipo, local y visitante por
    separado, con cuántos partidos hay detrás de cada promedio (para marcar
    baja confianza cuando la muestra es chica)."""
    log = load_log()
    if competition:
        log = log[log["competition"] == competition]
    if log.empty:
        return pd.DataFrame(columns=[
            "team", "partidos", "corners_prom", "faltas_prom",
            "tarjetas_amarillas_prom", "tarjetas_rojas_prom", "baja_confianza",
        ])
    grouped = log.groupby("team").agg(
        partidos=("team", "count"),
        corners_prom=("corners", "mean"),
        faltas_prom=("faltas", "mean"),
        tarjetas_amarillas_prom=("tarjetas_amarillas", "mean"),
        tarjetas_rojas_prom=("tarjetas_rojas", "mean"),
    ).reset_index()
    grouped["baja_confianza"] = grouped["partidos"] < MIN_SAMPLE_FOR_CONFIDENCE
    return grouped
