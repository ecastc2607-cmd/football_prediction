"""
Modelo de goles de Poisson para predecir resultados de un partido.

Dado un DataFrame de "team_strength" (ver team_strength.py) y un cruce local/visitante,
calcula:
  - goles esperados de cada equipo (xG del modelo, no confundir con el xG de Understat)
  - matriz de probabilidad de cada marcador exacto hasta MAX_GOALS
  - probabilidad de victoria local / empate / victoria visitante
  - probabilidad de Over/Under 2.5 goles y de que ambos anoten (BTTS)

Es el mismo método que usamos "a ojo" en el dashboard de la Jornada 1 de Champions,
pero calculado desde datos reales en vez de estimado cualitativamente.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

MAX_GOALS = 8  # marcador máximo por equipo que consideramos en la matriz


@dataclass
class MatchPrediction:
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    home_win: float
    draw: float
    away_win: float
    over_2_5: float
    btts: float
    score_matrix: np.ndarray

    def top_scorelines(self, n: int = 5) -> list[tuple[tuple[int, int], float]]:
        flat = [
            ((h, a), self.score_matrix[h, a])
            for h in range(MAX_GOALS + 1)
            for a in range(MAX_GOALS + 1)
        ]
        return sorted(flat, key=lambda x: x[1], reverse=True)[:n]

    def summary(self) -> str:
        lines = [
            f"{self.home_team} vs {self.away_team}",
            f"  xG modelo: {self.home_xg:.2f} - {self.away_xg:.2f}",
            f"  1X2: Local {self.home_win:.1%} | Empate {self.draw:.1%} | Visitante {self.away_win:.1%}",
            f"  Over 2.5 goles: {self.over_2_5:.1%}  |  Ambos anotan (BTTS): {self.btts:.1%}",
            "  Marcadores más probables: "
            + ", ".join(f"{h}-{a} ({p:.1%})" for (h, a), p in self.top_scorelines(3)),
        ]
        return "\n".join(lines)


def expected_goals(team_strength: pd.DataFrame, home_team: str, away_team: str) -> tuple[float, float]:
    for team in (home_team, away_team):
        if team not in team_strength.index:
            raise KeyError(
                f"'{team}' no está en los datos de fuerza de equipo. "
                "Revisa el nombre exacto tal como lo devuelve football-data.org."
            )

    league_home_avg = team_strength.attrs["league_home_avg"]
    league_away_avg = team_strength.attrs["league_away_avg"]

    home_attack = team_strength.loc[home_team, "attack_home"]
    away_defense = team_strength.loc[away_team, "defense_away"]
    away_attack = team_strength.loc[away_team, "attack_away"]
    home_defense = team_strength.loc[home_team, "defense_home"]

    home_xg = league_home_avg * home_attack * away_defense
    away_xg = league_away_avg * away_attack * home_defense
    return float(home_xg), float(away_xg)


def predict_match(team_strength: pd.DataFrame, home_team: str, away_team: str) -> MatchPrediction:
    home_xg, away_xg = expected_goals(team_strength, home_team, away_team)

    goals = np.arange(0, MAX_GOALS + 1)
    home_probs = poisson.pmf(goals, home_xg)
    away_probs = poisson.pmf(goals, away_xg)
    matrix = np.outer(home_probs, away_probs)
    matrix /= matrix.sum()  # renormaliza por la masa recortada en MAX_GOALS

    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())

    over_2_5 = float(sum(
        matrix[h, a] for h in goals for a in goals if h + a > 2.5
    ))
    btts = float(sum(
        matrix[h, a] for h in goals for a in goals if h > 0 and a > 0
    ))

    return MatchPrediction(
        home_team=home_team, away_team=away_team,
        home_xg=home_xg, away_xg=away_xg,
        home_win=home_win, draw=draw, away_win=away_win,
        over_2_5=over_2_5, btts=btts, score_matrix=matrix,
    )
