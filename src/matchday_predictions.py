"""
Función compartida para armar la tabla de predicciones de una jornada: la usan
por igual la CLI (predict_matchday.py), el export a JSON (export_weekend_predictions.py),
el notebook y el dashboard de Streamlit, para no repetir la misma lógica cuatro veces.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .poisson_model import predict_match
from .predict_matchday import upcoming_fixtures
from .team_strength import confidence_note, team_strength_for_competition


def matchday_predictions_df(competition_code: str, season: int, matchday: int,
                             seasons_back: int = 1) -> pd.DataFrame:
    """Devuelve un DataFrame con una fila por partido programado de esa jornada:
    equipos, xG del modelo, 1X2, Over/Under 2.5, BTTS, marcador más probable y
    aviso de baja confianza.
    """
    seasons = [season - i for i in range(seasons_back + 1)]
    strength = team_strength_for_competition(competition_code, seasons)
    fixtures = upcoming_fixtures(competition_code, season, matchday)

    rows = []
    for _, row in fixtures.iterrows():
        try:
            pred = predict_match(strength, row["home_team"], row["away_team"])
        except KeyError:
            continue
        note = confidence_note(strength, row["home_team"], row["away_team"])
        top_score, top_prob = pred.top_scorelines(1)[0]
        rows.append({
            "competition": competition_code,
            "competition_name": config.COMPETITIONS.get(competition_code, competition_code),
            "matchday": matchday,
            "utc_date": row["utc_date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_xg": round(pred.home_xg, 2),
            "away_xg": round(pred.away_xg, 2),
            "home_win": round(pred.home_win * 100, 1),
            "draw": round(pred.draw * 100, 1),
            "away_win": round(pred.away_win * 100, 1),
            "over_2_5": round(pred.over_2_5 * 100, 1),
            "btts": round(pred.btts * 100, 1),
            "top_score": f"{top_score[0]}-{top_score[1]}",
            "top_score_prob": round(top_prob * 100, 1),
            "low_confidence": note is not None,
            "confidence_note": note,
        })
    return pd.DataFrame(rows)
