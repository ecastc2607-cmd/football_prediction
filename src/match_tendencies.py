"""
Sugerencias de corners, faltas y tarjetas para partidos AÚN NO JUGADOS, a partir
del promedio histórico real de cada equipo — no del modelo de goles.

La fuente es data/tracking/match_stats_log.csv, que se alimenta de dos formas:
  - Automáticamente, cada vez que la pestaña "En vivo" consulta un partido real.
  - A propósito, corriendo scripts/backfill_match_stats.py contra Goal API.

Esto es una TENDENCIA simple ("estos dos equipos, en promedio, generan más o
menos corners/faltas/tarjetas que tal línea de referencia"), no una predicción
del modelo de Poisson. Con pocos partidos en el log el promedio es poco fiable
— por eso cada pick se marca como de baja confianza cuando alguno de los dos
equipos tiene menos de match_stats_log.MIN_SAMPLE_FOR_CONFIDENCE partidos.

Líneas de referencia: son las que el propio proyecto usa para decidir "Sí" o
"No" (no la cuota exacta de ninguna casa de apuestas en particular) — elegidas
porque son los valores de over/under más comunes en estos mercados.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .match_stats_log import MIN_SAMPLE_FOR_CONFIDENCE, team_averages

CORNERS_LINE = 8.5
FALTAS_LINE = 22.5
TARJETAS_LINE = 3.5  # amarillas + rojas de ambos equipos


@dataclass
class TendencyPick:
    label: str                 # "Corners", "Faltas", "Tarjetas"
    line: float
    expected: float | None     # suma esperada de ambos equipos, o None si falta muestra
    probable: str | None       # "Sí" / "No" / None
    low_confidence: bool
    sample_home: int
    sample_away: int

    def describe(self) -> str:
        """Texto compacto para una celda de tabla."""
        if self.expected is None:
            return "Sin datos"
        marca = "⚠ " if self.low_confidence else ""
        return f"{marca}{self.probable} · {self.expected:.1f} esp."


def load_team_averages(competition_code: str) -> pd.DataFrame:
    """Promedios por equipo de esa competición, indexados por nombre de equipo
    (para no repetir el groupby en cada partido de la jornada)."""
    averages = team_averages(competition_code)
    return averages.set_index("team") if not averages.empty else averages


def _lookup(averages_by_team: pd.DataFrame, team: str, column: str) -> tuple[float | None, int]:
    if averages_by_team.empty or team not in averages_by_team.index:
        return None, 0
    row = averages_by_team.loc[team]
    return row[column], int(row["partidos"])


def pick_tendencies(averages_by_team: pd.DataFrame, home_team: str, away_team: str) -> list[TendencyPick]:
    """Sugerencias de corners/faltas/tarjetas para un partido, a partir de la
    tabla de promedios ya calculada con load_team_averages(). Nunca falla: si
    falta la muestra de alguno de los dos equipos, esa fila queda en "Sin datos".
    """
    picks = []
    for label, column, line in [
        ("Corners", "corners_prom", CORNERS_LINE),
        ("Faltas", "faltas_prom", FALTAS_LINE),
    ]:
        h_val, h_n = _lookup(averages_by_team, home_team, column)
        a_val, a_n = _lookup(averages_by_team, away_team, column)
        if h_val is None or a_val is None:
            picks.append(TendencyPick(label, line, None, None, True, h_n, a_n))
            continue
        expected = round(h_val + a_val, 1)
        picks.append(TendencyPick(
            label, line, expected, "Sí" if expected >= line else "No",
            h_n < MIN_SAMPLE_FOR_CONFIDENCE or a_n < MIN_SAMPLE_FOR_CONFIDENCE,
            h_n, a_n,
        ))

    # Tarjetas = amarillas + rojas de ambos equipos combinadas.
    h_am, h_n = _lookup(averages_by_team, home_team, "tarjetas_amarillas_prom")
    h_ro, _ = _lookup(averages_by_team, home_team, "tarjetas_rojas_prom")
    a_am, a_n = _lookup(averages_by_team, away_team, "tarjetas_amarillas_prom")
    a_ro, _ = _lookup(averages_by_team, away_team, "tarjetas_rojas_prom")
    if h_am is None or a_am is None:
        picks.append(TendencyPick("Tarjetas", TARJETAS_LINE, None, None, True, h_n, a_n))
    else:
        expected = round((h_am or 0) + (h_ro or 0) + (a_am or 0) + (a_ro or 0), 1)
        picks.append(TendencyPick(
            "Tarjetas", TARJETAS_LINE, expected, "Sí" if expected >= TARJETAS_LINE else "No",
            h_n < MIN_SAMPLE_FOR_CONFIDENCE or a_n < MIN_SAMPLE_FOR_CONFIDENCE,
            h_n, a_n,
        ))

    return picks
