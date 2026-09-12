"""
Resuelve automáticamente "temporada actual" y "jornada por defecto" de una
competición, usando los metadatos que football-data.org ya calcula
(currentSeason, currentMatchday) en vez de que el usuario los adivine.

Reglas (pedidas explícitamente por el usuario):
- Temporada: la que football-data.org marca como currentSeason. Si ya terminó
  (hoy > endDate) y aún no rotaron a la siguiente en su API, se asume la próxima
  (año + 1) como respaldo.
- Jornada: currentMatchday de esa temporada. Si TODOS los partidos de esa jornada
  ya están FINISHED (nada en juego ni programado en ella), se avanza a la
  siguiente jornada por defecto.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .fetch_football_data import FootballDataClient


@dataclass
class CompetitionStatus:
    season: int
    matchday: int
    season_start: str
    season_end: str
    reason: str  # explica por qué se eligió esa jornada, para mostrar en la UI


def resolve_current_season_and_matchday(client: FootballDataClient, code: str) -> CompetitionStatus:
    meta = client.get_competition_meta(code)
    current = meta.get("currentSeason") or {}
    start_date = current.get("startDate", "")
    end_date = current.get("endDate", "")
    season_year = int(start_date[:4]) if start_date else datetime.now(timezone.utc).year
    matchday = current.get("currentMatchday") or 1

    today = datetime.now(timezone.utc).date()
    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if today > end:
            # La API todavía no roto a la siguiente temporada: asumimos la próxima.
            season_year += 1
            matchday = 1
            return CompetitionStatus(
                season=season_year, matchday=matchday,
                season_start=start_date, season_end=end_date,
                reason="Temporada de descanso: se asumió la siguiente temporada (jornada 1).",
            )

    # ¿La jornada actual ya está completamente jugada? -> pasar a la siguiente.
    matches = client.get_matches(code, season_year)
    df = pd.DataFrame(matches.get("matches", []))
    if not df.empty:
        this_md = df[df["matchday"] == matchday]
        if not this_md.empty and (this_md["status"] == "FINISHED").all():
            matchday += 1
            return CompetitionStatus(
                season=season_year, matchday=matchday,
                season_start=start_date, season_end=end_date,
                reason=f"La jornada {matchday - 1} ya terminó por completo: se adelantó a la {matchday}.",
            )

    return CompetitionStatus(
        season=season_year, matchday=matchday,
        season_start=start_date, season_end=end_date,
        reason=f"Jornada {matchday} en curso según football-data.org.",
    )
