"""
Fuerza para anotar por tiempo (1er y 2do), para analizar apuestas de un solo
tiempo.

Mismo método que el modelo de partido completo (team_strength + Poisson),
pero entrenado solo con los goles de cada mitad: 1er tiempo = marcador al
descanso; 2do tiempo = final menos descanso. Así cada equipo tiene su propio
perfil (hay equipos que salen fuerte y otros que anotan sobre el final), con
el mismo suavizado hacia el promedio de la liga cuando tiene pocos partidos.

Referencia (oct-2026, datos del proyecto): en las ligas, ~43-45% de los goles
caen en el 1er tiempo — la proporción que reporta la literatura.
"""
from __future__ import annotations

import math

import pandas as pd

from .poisson_model import expected_goals
from .team_strength import build_team_strength, load_finished_matches

MIN_MATCHES = 20  # menos partidos con marcador al descanso: no se calcula


def half_strengths(competition_code: str, seasons: list[int]) -> dict[str, pd.DataFrame] | None:
    """{"1T": fuerzas, "2T": fuerzas}, o None si no hay marcadores al descanso
    suficientes (ej. CSV generados antes de guardar ese dato)."""
    try:
        partidos = load_finished_matches(competition_code, seasons)
    except FileNotFoundError:
        return None
    return half_strengths_from_matches(partidos)


def half_strengths_from_matches(partidos: pd.DataFrame) -> dict[str, pd.DataFrame] | None:
    """Igual que half_strengths, pero sobre partidos ya cargados (el backtest
    le pasa solo los anteriores a cada partido)."""
    if "home_ht_goals" not in partidos.columns:
        return None
    m = partidos.dropna(subset=["home_ht_goals", "away_ht_goals"]).copy()
    if len(m) < MIN_MATCHES:
        return None
    ht_l, ht_v = m["home_ht_goals"].astype(int), m["away_ht_goals"].astype(int)
    primero = m.assign(home_goals=ht_l, away_goals=ht_v)
    segundo = m.assign(home_goals=(m["home_goals"] - ht_l).clip(lower=0),
                       away_goals=(m["away_goals"] - ht_v).clip(lower=0))
    return {"1T": build_team_strength(primero), "2T": build_team_strength(segundo)}


def half_prediction(strengths: dict[str, pd.DataFrame] | None, home_team: str, away_team: str) -> dict | None:
    """Por tiempo: goles esperados de cada equipo, probabilidad de que haya al
    menos un gol y de que haya más de 1.5. None si falta algún equipo."""
    if not strengths:
        return None
    salida = {}
    for mitad, fuerzas in strengths.items():
        try:
            local, visita = expected_goals(fuerzas, home_team, away_team)
        except KeyError:
            return None
        salida[mitad] = {"home_xg": local, "away_xg": visita, **half_probabilities(local, visita)}
    return salida


def half_probabilities(home_xg: float, away_xg: float) -> dict:
    """Probabilidad de al menos un gol y de más de 1.5 goles en una mitad,
    con Poisson sobre el total esperado de esa mitad."""
    total = home_xg + away_xg
    return {
        "p_goal": 1 - math.exp(-total),
        "p_over_1_5": 1 - math.exp(-total) * (1 + total),
    }
