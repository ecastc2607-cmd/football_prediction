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
from .team_strength import MIN_MATCHES_WARNING, build_team_strength, load_finished_matches

# Partidos mínimos en la copa, por rol (local/visitante), para fiarse de la
# fuerza propia — el mismo umbral con que confidence_note marca baja confianza.
MIN_ROLE_MATCHES = MIN_MATCHES_WARNING


def _league_name(code: str) -> str:
    return config.COMPETITIONS.get(code) or config.SUPPORT_LEAGUES.get(code, code)

# Por debajo de esto, dos nombres normalizados parecidos pero no idénticos se
# consideran equipos DISTINTOS en vez de arriesgar un cruce falso — acá solo
# hay una señal (el nombre solo, sin fecha/rival como en goal_api_client), así
# que el umbral es más exigente que el de find_fixture_id.
FUZZY_THRESHOLD = 0.85

# Ligas donde se busca el respaldo, y en qué orden (todas las que football-data.org
# ya nos trae completas). No incluye "CL"/"EL": mezclar copa-con-copa no tiene el
# mismo sustento (la razón de ser de esto es aprovechar datos de LIGA que sí
# existen para casi cualquier equipo top).
DOMESTIC_LEAGUES = ["PL", "PD", "SA", "BL1", "FL1", *config.SUPPORT_LEAGUES]


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
            path = config.matches_path(code, season)
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, usecols=["home_team", "away_team"])
            except (ValueError, pd.errors.EmptyDataError):
                continue
            for team in pd.concat([df["home_team"], df["away_team"]]).dropna().unique():
                mapping.setdefault(normalize_team_name(team), (team, code))
    return mapping


# Nombres que ni normalizando se parecen entre fuentes (Goal API -> football-data.org).
_ALIASES = {
    "rennes": "rennais",  # "Rennes" vs "Stade Rennais FC 1901"
    "benfica": "sport lisboa e benfica",  # "Benfica" vs "Sport Lisboa e Benfica"
}


def _match_domestic_team(team: str, domestic_by_norm: dict[str, tuple[str, str]]) -> tuple[str, str] | None:
    """(nombre_real, código_de_liga) del equipo doméstico que mejor calza con
    `team` (nombre de Goal API), o None si ninguno es lo bastante parecido."""
    normalizado = normalize_team_name(team)
    normalizado = _ALIASES.get(normalizado, normalizado)
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
    before: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Para cada equipo de `missing_teams` (normalmente: todos los de la
    jornada), usa su fuerza en su liga doméstica donde la de la copa no
    alcanza: fila completa si no tiene ningún partido en la copa, o solo el
    rol (local/visitante) con menos de MIN_ROLE_MATCHES partidos.

    `before`: solo usa partidos de liga anteriores a ese instante (para el
    backtest, que no debe "ver" resultados posteriores al partido que predice).

    Devuelve (strength_ampliado, {equipo: nombre_de_liga_usada}) — el segundo
    valor es justo para poder avisarlo en la UI, nunca en silencio.
    """
    # Qué rol (local/visitante) de cada equipo tiene muy poca muestra en la
    # copa. No basta con mirar si el equipo falta del todo: en Europa League,
    # desde la jornada 2 todos tienen 1 partido, pero de un solo rol — y la
    # jornada siguiente casi siempre les toca el otro, con 0 partidos, así que
    # sin esto el modelo caía exacto al promedio de la copa para casi todos.
    roles_flojos: dict[str, list[str]] = {}
    for t in missing_teams:
        if t not in strength.index:
            roles_flojos[t] = ["home", "away"]
            continue
        flojos = [r for r in ("home", "away") if strength.loc[t, f"{r}_played"] < MIN_ROLE_MATCHES]
        if flojos:
            roles_flojos[t] = flojos
    if not roles_flojos:
        return strength, {}

    strength = strength.copy()  # se modifican filas: no tocar el DataFrame de quien llama
    domestic_by_norm = _domestic_teams_by_normalized_name(seasons)
    fuente_usada: dict[str, str] = {}
    filas_nuevas = []
    cache_por_liga: dict[str, pd.DataFrame] = {}

    for team, roles in roles_flojos.items():
        encontrado = _match_domestic_team(team, domestic_by_norm)
        if encontrado is None:
            continue  # no juega ninguna liga que descargamos: no hay de dónde tomarlo
        nombre_domestico, code = encontrado

        if code not in cache_por_liga:
            try:
                matches = load_finished_matches(code, seasons)
                if before is not None:
                    matches = matches[pd.to_datetime(matches["utc_date"], utc=True) < before]
                cache_por_liga[code] = build_team_strength(matches) if not matches.empty else pd.DataFrame()
            except (FileNotFoundError, ValueError):
                cache_por_liga[code] = pd.DataFrame()

        domestic = cache_por_liga[code]
        if domestic.empty or nombre_domestico not in domestic.index:
            continue

        fila_domestica = domestic.loc[nombre_domestico]
        if team in strength.index:
            # Solo se reemplazan las fuerzas del rol flojo; los partidos jugados
            # (home_played/away_played) quedan los de la copa, así confidence_note
            # sigue avisando que la muestra propia es chica.
            for rol in roles:
                for col in (f"attack_{rol}", f"defense_{rol}"):
                    strength.loc[team, col] = fila_domestica[col]
        else:
            # .rename(team): la fila queda indexada con el nombre TAL COMO lo usa
            # la copa (Goal API), que es como predict_match la va a buscar.
            filas_nuevas.append(fila_domestica.rename(team))
        fuente_usada[team] = _league_name(code)

    if not fuente_usada:
        return strength, {}
    if not filas_nuevas:
        return strength, fuente_usada

    # Conserva los promedios de gol de LA COPA (no los de la liga doméstica): el
    # ratio prestado se multiplica por el promedio de Europa League, igual que
    # para cualquier otro equipo — ver poisson_model.expected_goals.
    attrs_originales = dict(strength.attrs)
    combinado = pd.concat([strength, pd.DataFrame(filas_nuevas)])
    combinado.attrs = attrs_originales
    return combinado, fuente_usada
