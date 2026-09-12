"""
Prueba de humo del modelo con datos de ejemplo (no reales), sin necesitar la API key.
Sirve para validar que team_strength.py + poisson_model.py funcionan correctamente
antes de conectar la fuente de datos real.

Uso:
    python -m src.demo_sample_data
"""
from __future__ import annotations

import pandas as pd

from .poisson_model import predict_match
from .team_strength import build_team_strength

# 8 partidos de ejemplo entre 4 equipos ficticios, suficientes para calcular fuerzas.
SAMPLE_MATCHES = pd.DataFrame([
    {"home_team": "Halcones FC", "away_team": "Titanes CD", "home_goals": 3, "away_goals": 1},
    {"home_team": "Titanes CD", "away_team": "Halcones FC", "home_goals": 1, "away_goals": 2},
    {"home_team": "Halcones FC", "away_team": "Estrella UD", "home_goals": 2, "away_goals": 0},
    {"home_team": "Estrella UD", "away_team": "Halcones FC", "home_goals": 0, "away_goals": 1},
    {"home_team": "Rayo Norte", "away_team": "Halcones FC", "home_goals": 1, "away_goals": 1},
    {"home_team": "Halcones FC", "away_team": "Rayo Norte", "home_goals": 2, "away_goals": 2},
    {"home_team": "Titanes CD", "away_team": "Estrella UD", "home_goals": 1, "away_goals": 1},
    {"home_team": "Estrella UD", "away_team": "Rayo Norte", "home_goals": 0, "away_goals": 2},
])


def main():
    strength = build_team_strength(SAMPLE_MATCHES)
    print("Fuerzas de equipo calculadas:\n")
    print(strength[["home_played", "attack_home", "defense_home", "attack_away", "defense_away"]])
    print()

    pred = predict_match(strength, "Halcones FC", "Titanes CD")
    print(pred.summary())


if __name__ == "__main__":
    main()
