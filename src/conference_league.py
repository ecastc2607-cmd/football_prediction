"""
UEFA Conference League vía Goal API (football-data.org no la incluye en el
plan gratis; verificado oct-2026). Misma lógica que la Europa League — ver
uefa_cup.py.

Diferencia clave: 28 de los 36 equipos de la fase de liga 2026/27 no juegan
ninguna liga que descarguemos (Copenhagen, Crvena zvezda, Trabzonspor, Kairat...),
así que la fuerza se calcula también con la previa y las eliminatorias de la
Conference (`strength_from_all_stages`), que casi todos jugaron, y con los
partidos de Europa League de la misma temporada (varios cayeron desde su
previa) leídos del historial versionado, sin gastar cuota. Los 8 que sí
juegan en las ligas descargadas (Brighton, Getafe, Atalanta, Freiburg, Monaco,
Ajax, Twente, Braga) además usan su liga doméstica donde la muestra propia no
alcanza — ver cross_competition_strength.py.
"""
from __future__ import annotations

from .uefa_cup import GoalApiCup

COMPETITION_CODE = "ECL"
_CUP = GoalApiCup(COMPETITION_CODE, strength_from_all_stages=True, extra_strength_from=("EL",))
LEAGUE_ID = _CUP.league_id

current_season = _CUP.current_season
resolve_current_season_and_matchday = _CUP.resolve_current_season_and_matchday
fetch_competition = _CUP.fetch_competition
get_standings_map = _CUP.get_standings_map
get_live_matches = _CUP.get_live_matches
