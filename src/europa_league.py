"""
Europa League vía Goal API — football-data.org NO la incluye en el plan gratis
(verificado contra su API y su tabla de cobertura: hace falta su plan de
€49/mes). Toda la lógica vive en uefa_cup.py (compartida con la Conference
League); este módulo solo expone el contrato que usan app.py,
resolve_predictions.py y fetch_football_data.py vía config.goal_api_module.

La fuerza de los equipos sale solo de la fase de liga (como siempre): los que
no tienen muestra propia usan su liga doméstica — ver cross_competition_strength.py.
"""
from __future__ import annotations

from .uefa_cup import LEAGUE_PHASE_STAGES, GoalApiCup  # noqa: F401

COMPETITION_CODE = "EL"
_CUP = GoalApiCup(COMPETITION_CODE)
LEAGUE_ID = _CUP.league_id

current_season = _CUP.current_season
resolve_current_season_and_matchday = _CUP.resolve_current_season_and_matchday
fetch_competition = _CUP.fetch_competition
get_standings_map = _CUP.get_standings_map
get_live_matches = _CUP.get_live_matches
