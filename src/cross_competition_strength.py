"""
Respaldo de fuerza de equipo entre competiciones, para copas con mucha rotación
de participantes (Europa League): muchos equipos no tienen NINGÚN partido
propio en la copa todavía (ni esta temporada ni la anterior), así que en vez de
omitir el partido del todo, se usa la fuerza de ataque/defensa del equipo en SU
LIGA DOMÉSTICA como aproximación.

Por qué esto es razonable, y sus límites: attack_home/defense_home/etc. (ver
team_strength.py) ya son proporciones respecto al promedio de SU PROPIA
competición (1.0 = equipo promedio de esa liga), no goles absolutos — así que
"Juventus ataca 40% más que el equipo promedio de Serie A" es una señal
transferible, aunque imperfecta, a Europa League: se multiplica por el promedio
de goles DE EUROPA LEAGUE (no el de Serie A), tal como hace poisson_model.py
con cualquier otro equipo. No es lo mismo que tener partidos reales del propio
torneo, así que todo partido que use esto queda marcado explícitamente en
`confidence_note` — nunca se hace pasar por dato propio de la competición.

Verificado con datos reales (Europa League, jornada 1 2026/27): 28 de 36
equipos no tenían fase de liga 2025/26 propia; 10 de esos 28 son de las 5
grandes ligas que este proyecto ya descarga, así que se benefician de esto sin
necesitar ninguna fuente de datos nueva.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .goal_api_client import normalize_team_name, team_name_similarity
from .team_strength import build_team_strength, load_finished_matches

# Por debajo de esto, dos nombres normalizados parecidos pero no idénticos se
# consideran equipos DISTINTOS en vez de arriesgar un cruce falso — acá solo
# hay una señal (el nombre solo, sin fecha/rival como en goal_api_client), así
# que el umbral es más exigente que el de find_fixture_id.
FUZZY_THRESHOLD = 0.85

# Ligas donde se busca el respaldo, y en qué orden (todas las que football-data.org
# ya nos trae completas). No incluye "CL"/"EL": mezclar copa-con-copa no tiene el
# mismo sustento (la razón de ser de esto es aprovechar datos de LIGA que sí
# existen para casi cualquier equipo top).
DOMESTIC_LEAGUES = ["PL", "PD", "SA", "BL1", "FL1"]


def _domestic_teams_by_normalized_name(seasons: list[int]) -> dict[str, tuple[str, str]]:
    """{nombre_normalizado: (nombre_real_en_esa_liga, código_de_liga)} para todo
    equipo de las 5 grandes ligas, en las temporadas dadas.

    Se indexa por nombre NORMALIZADO (sin acentos/sufijos de club/números) y no
    por el nombre tal cual, porque Goal API (de donde vienen los partidos de
    Europa League) usa nombres cortos ("Crystal Palace", "Milan") y
    football-data.org usa los oficiales largos ("Crystal Palace FC", "AC Milan")
    — el mismo problema que ya resolvió goal_api_client.py para emparejar
    partidos, reutilizado acá para emparejar EQUIPOS.
    """
    mapping: dict[str, tuple[str, str]] = {}
    for season in seasons:
        for code in DOMESTIC_LEAGUES:
            path = config.PROCESSED_DIR / f"matches_{code}_{season}.csv"
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, usecols=["home_team", "away_team"])
            except (ValueError, pd.errors.EmptyDataError):
                continue
            for team in pd.concat([df["home_team"], df["away_team"]]).dropna().unique():
                mapping.setdefault(normalize_team_name(team), (team, code))
    return mapping


def _match_domestic_team(team: str, domestic_by_norm: dict[str, tuple[str, str]]) -> tuple[str, str] | None:
    """(nombre_real, código_de_liga) del equipo doméstico que mejor calza con
    `team` (nombre de Goal API), o None si ninguno es lo bastante parecido."""
    normalizado = normalize_team_name(team)
    exacto = domestic_by_norm.get(normalizado)
    if exacto:
        return exacto
    if not domestic_by_norm:
        return None
    mejor_norm = max(domestic_by_norm, key=lambda n: team_name_similarity(n, normalizado))
    if team_name_similarity(mejor_norm, normalizado) >= FUZZY_THRESHOLD:
        return domestic_by_norm[mejor_norm]
    return None


def fill_missing_with_domestic_strength(
    strength: pd.DataFrame, missing_teams: set[str], seasons: list[int],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Agrega al DataFrame de fuerzas una fila por cada equipo de `missing_teams`
    que no esté ya en el índice, tomada de su liga doméstica.

    Devuelve (strength_ampliado, {equipo: nombre_de_liga_usada}) — el segundo
    valor es justo para poder avisarlo en la UI, nunca en silencio.
    """
    faltantes = {t for t in missing_teams if t not in strength.index}
    if not faltantes:
        return strength, {}

    domestic_by_norm = _domestic_teams_by_normalized_name(seasons)
    fuente_usada: dict[str, str] = {}
    filas_nuevas = []
    cache_por_liga: dict[str, pd.DataFrame] = {}

    for team in faltantes:
        encontrado = _match_domestic_team(team, domestic_by_norm)
        if encontrado is None:
            continue  # no juega ninguna de las 5 grandes ligas: no hay de dónde tomarlo
        nombre_domestico, code = encontrado

        if code not in cache_por_liga:
            try:
                matches = load_finished_matches(code, seasons)
                cache_por_liga[code] = build_team_strength(matches) if not matches.empty else pd.DataFrame()
            except (FileNotFoundError, ValueError):
                cache_por_liga[code] = pd.DataFrame()

        domestic = cache_por_liga[code]
        if domestic.empty or nombre_domestico not in domestic.index:
            continue

        # .rename(team): la fila queda indexada con el nombre TAL COMO lo usa la
        # copa (Goal API), que es como predict_match la va a buscar — aunque los
        # datos de fondo vengan del nombre oficial de su liga doméstica.
        filas_nuevas.append(domestic.loc[nombre_domestico].rename(team))
        fuente_usada[team] = config.COMPETITIONS.get(code, code)

    if not filas_nuevas:
        return strength, {}

    # Conserva los promedios de gol de LA COPA (no los de la liga doméstica): el
    # ratio prestado se multiplica por el promedio de Europa League, igual que
    # para cualquier otro equipo — ver poisson_model.expected_goals.
    attrs_originales = dict(strength.attrs)
    combinado = pd.concat([strength, pd.DataFrame(filas_nuevas)])
    combinado.attrs = attrs_originales
    return combinado, fuente_usada
