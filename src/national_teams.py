"""
Selecciones nacionales vía Goal API — football-data.org NO trae Nations League
ni amistosos (verificado contra /v4/competitions: solo tiene Mundial y
Eurocopa, y encima solo durante el torneo mismo). Igual que Europa League
(ver src/europa_league.py), esta es una fuente aislada a propósito: si falla o
cambia de forma, nunca debe tumbar las otras 6 ligas + Champions + Europa League.

Cubre dos cosas, ambas elegidas por el usuario entre las opciones disponibles
en Goal API (también existían eliminatorias mundialistas, descartadas por
tener selecciones con poco o ningún historial propio):

  - UEFA Nations League: SÍ tiene fase de liga con jornadas (League A/B/C/D,
    6 jornadas cada una) — encaja directo en el mismo concepto de "jornada"
    que ya usa el resto del proyecto, así que corre por matchday_predictions_df
    normal (ver fetch_competition).
  - Amistosos internacionales: NO tienen jornada ni tabla — cualquier
    selección contra cualquiera, cualquier día. Se muestran aparte, en una
    ventana de días (ver fetch_friendlies_window), fuera del flujo de jornada.

Fuerza de cada selección: la Nations League juega muy pocos partidos por
ciclo (6 cada 2 años) para calcular fuerza solo con la edición en curso, así
que se combina con su historial de amistosos vía /teams/:id/fixtures (un solo
pedido por selección trae TODOS sus partidos, de cualquier torneo) — mucho más
barato que paginar todo el calendario de amistosos del mundo (5.000+ partidos,
casi todo fútbol juvenil) para luego filtrar por equipo.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from . import config
from .competition_status import CompetitionStatus
from .goal_api_client import (
    GOAL_API_KEY,
    LEAGUE_IDS,
    GoalApiClient,
    GoalApiRateLimited,
    live_fixtures,
    memoized,
)
from .live_matches import REGULATION_MINUTES, live_win_probabilities
from .poisson_model import expected_goals, predict_match

COMPETITION_CODE = "NT"

# IDs fijos de Goal API, verificados a mano contra /leagues junto con su país
# (mismo cuidado que LEAGUE_IDS en goal_api_client.py: hay decenas de ligas con
# nombres parecidos).
NATIONS_LEAGUE_ID = LEAGUE_IDS[COMPETITION_CODE]  # UEFA Nations League
FRIENDLIES_LEAGUE_ID = "cmr77dwv800nzrx06faqngjwh"  # Friendlies (World) — TODAS las categorías, se filtra por nombre

# Fase de liga de la Nations League: la que tiene jornadas 1-6 con formato de
# grupos, igual de espíritu a la "League Phase" de Champions/Europa League. Se
# excluyen Play-offs/Quarter-finals/Semi-finals/Final (eliminatoria a partido
# único en marzo, sin jornadas ni tabla).
GROUP_STAGE_NAMES = {"League A", "League B", "League C", "League D"}

# Cuántos días atrás se aceptan partidos de Nations League/amistosos al armar
# el historial de una selección: 900 días (~2.5 años) asegura cubrir la edición
# completa anterior de la Nations League (cada 2 años) más amistosos recientes,
# sin arrastrar resultados demasiado viejos como para representar la plantilla actual.
HISTORY_LOOKBACK_DAYS = 900

# El historial (≈60 peticiones: una por selección) se guarda en un archivo
# versionado en git, igual que match_stats_log.csv: sobrevive a los reinicios
# de Streamlit Cloud (data/processed/ no se versiona y se pierde en cada uno)
# y solo se vuelve a descargar si tiene más de HISTORY_REFRESH_DAYS. Son
# resultados ya jugados de ~2.5 años: una semana de atraso no mueve la fuerza
# de ninguna selección.
HISTORY_TRACKED_PATH = config.ROOT_DIR / "data" / "tracking" / "nt_history.csv"
HISTORY_REFRESH_DAYS = 7

# Amistosos: Goal API mete en la misma liga selecciones absolutas y TODAS las
# categorías juveniles/femeninas ("England U17", "Spain W"...). Sin esto,
# "Amistosos internacionales" mostraría en su mayoría partidos Sub-19.
_YOUTH_RE = re.compile(r"\bU1[5-9]\b|\bU2[0-3]\b", re.IGNORECASE)


def _is_senior_team(name: str | None) -> bool:
    if not name:
        return False
    if _YOUTH_RE.search(name):
        return False
    if name.rstrip().endswith(" W") or "women" in name.lower():
        return False
    return True


def _client() -> GoalApiClient:
    return GoalApiClient(api_key=GOAL_API_KEY)


def _get_with_retry(client: GoalApiClient, path: str, params: dict, attempts: int = 2):
    """Mismo reintento ante timeout puntual que europa_league.py — pensado para
    /leagues/:id/fixtures y /teams/:id/fixtures, que a veces tardan."""
    for intento in range(attempts):
        try:
            return client._get(path, params)
        except requests.exceptions.Timeout:
            if intento == attempts - 1:
                raise
            time.sleep(1.5)


def _paginate(client: GoalApiClient, path: str, page_size: int = 100, hard_limit: int = 2000) -> list[dict]:
    items: list[dict] = []
    offset = 0
    while True:
        page = _get_with_retry(client, path, {"limit": page_size, "offset": offset})
        if not page:
            break
        items.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
        if offset > hard_limit:
            break
    return items


def _normalize_utc(kickoff_iso: str) -> str:
    return kickoff_iso[:19] + "Z"


def current_season() -> int:
    """Temporada que Goal API considera vigente para la Nations League (su
    campo 'season', ej. "2026/2027" -> 2026) — mismo espíritu que
    europa_league.current_season(). En memoria 6 h: una carga del dashboard
    la pedía 5-6 veces y solo cambia una vez cada dos años."""
    def _pedir():
        try:
            data = _client()._get(f"/leagues/{NATIONS_LEAGUE_ID}", {})
        except Exception:
            # Sin API (ej. cuota agotada): el historial versionado sabe de qué
            # edición es, así que no hace falta el API para esto.
            if HISTORY_TRACKED_PATH.exists():
                guardado = pd.read_csv(HISTORY_TRACKED_PATH, usecols=["season_start_year"])
                if not guardado.empty:
                    return int(guardado["season_start_year"].max()) + 1
            raise
        season_label = (data or {}).get("season", "")
        if not season_label or "/" not in season_label:
            return datetime.now(timezone.utc).year
        return int(season_label.split("/")[0])
    return memoized("nt_current_season", 6 * 3600, _pedir)


# Todo el calendario histórico de la Nations League (~6 páginas). Se pide una
# vez y lo comparten la jornada actual, la temporada en curso y el historial —
# antes cada uno lo descargaba por su cuenta en la misma carga.
FIXTURES_MEMO_SECONDS = 15 * 60


def _fetch_group_stage_fixtures(client: GoalApiClient, season: int) -> list[dict]:
    all_fixtures = memoized(
        "nt_all_fixtures", FIXTURES_MEMO_SECONDS,
        lambda: _paginate(client, f"/leagues/{NATIONS_LEAGUE_ID}/fixtures"),
    )
    year_label = f"{season}/{season + 1}"
    return [
        f for f in all_fixtures
        if f.get("leagueYear") == year_label and f.get("stageName") in GROUP_STAGE_NAMES
    ]


def _team_history(client: GoalApiClient, team_id: str) -> list[dict]:
    return _paginate(client, f"/teams/{team_id}/fixtures", hard_limit=500)


def _row_from_fixture(f: dict, season_label: int) -> dict:
    home_goals = pd.to_numeric(f.get("homeTeamScore"), errors="coerce")
    away_goals = pd.to_numeric(f.get("awayTeamScore"), errors="coerce")
    winner = None
    if f.get("matchStatus") == "FINISHED" and pd.notna(home_goals) and pd.notna(away_goals):
        winner = "HOME_TEAM" if home_goals > away_goals else "AWAY_TEAM" if away_goals > home_goals else "DRAW"

    matchday = None
    round_raw = f.get("matchRound")
    if round_raw is not None:
        try:
            matchday = int(round_raw)
        except (TypeError, ValueError):
            matchday = None  # rondas de eliminatoria ("Final", "Semi-finals"...) no tienen jornada

    return {
        "match_id": f.get("id"),
        "competition": COMPETITION_CODE,
        "season_start_year": season_label,
        "matchday": matchday,
        "utc_date": _normalize_utc(f.get("kickoffUtc", "")),
        "status": f.get("matchStatus"),
        "home_team": (f.get("homeTeam") or {}).get("name") or f.get("homeTeamName"),
        "away_team": (f.get("awayTeam") or {}).get("name") or f.get("awayTeamName"),
        "home_team_short": f.get("homeTeamName"),
        "away_team_short": f.get("awayTeamName"),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "winner": winner,
    }


def resolve_current_season_and_matchday() -> CompetitionStatus:
    """Igual de espíritu a europa_league.resolve_current_season_and_matchday:
    la jornada actual es la primera de la fase de liga que no está 100%
    FINISHED todavía."""
    season = current_season()
    client = _client()
    try:
        fixtures = _fetch_group_stage_fixtures(client, season)
    except Exception:
        # Sin API (ej. cuota agotada): el calendario guardado en disco basta —
        # la primera jornada con algún partido que no haya empezado hace más
        # de 3 h (margen para lo que está en juego).
        path = config.PROCESSED_DIR / f"matches_{COMPETITION_CODE}_{season}.csv"
        if not path.exists():
            raise
        cal = pd.read_csv(path)
        inicio = pd.to_datetime(cal["utc_date"], utc=True, errors="coerce")
        vigentes = cal[inicio > pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=3)]
        md = int(vigentes["matchday"].min()) if not vigentes.empty else int(cal["matchday"].max())
        return CompetitionStatus(
            season=season, matchday=md, season_start="", season_end="",
            reason=f"Jornada {md} según el calendario guardado (Goal API no disponible ahora).",
        )

    if not fixtures:
        return CompetitionStatus(
            season=season, matchday=1, season_start="", season_end="",
            reason="Aún no hay calendario de la fase de liga para esta temporada (Goal API).",
        )

    rounds = sorted({int(f["matchRound"]) for f in fixtures if str(f.get("matchRound", "")).isdigit()})
    for md in rounds:
        md_fixtures = [f for f in fixtures if str(f.get("matchRound", "")) == str(md)]
        if not all(f.get("matchStatus") == "FINISHED" for f in md_fixtures):
            return CompetitionStatus(
                season=season, matchday=md, season_start="", season_end="",
                reason=f"Jornada {md} en curso según Goal API (fase de liga).",
            )

    ultima = rounds[-1] if rounds else 1
    return CompetitionStatus(
        season=season, matchday=ultima, season_start="", season_end="",
        reason=f"La fase de liga llegó hasta la jornada {ultima} según los datos disponibles.",
    )


def _fetch_current_edition(client: GoalApiClient, season: int) -> pd.DataFrame:
    print(f"Descargando UEFA Nations League (temporada {season}) vía Goal API...")
    fixtures = _fetch_group_stage_fixtures(client, season)
    rows = [_row_from_fixture(f, season) for f in fixtures]
    df = pd.DataFrame(rows)

    csv_path = config.PROCESSED_DIR / f"matches_{COMPETITION_CODE}_{season}.csv"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    finished = (df["status"] == "FINISHED").sum() if not df.empty else 0
    print(f"  {len(df)} partidos guardados en {csv_path.relative_to(config.ROOT_DIR)} "
          f"({finished} finalizados, {len(df) - finished} pendientes/en curso).")
    return df


def _load_tracked_history(season_label: int) -> pd.DataFrame | None:
    """El historial versionado en git, o None si no existe o es de otra
    edición (al cambiar de ciclo de la Nations League hay que rearmarlo)."""
    if not HISTORY_TRACKED_PATH.exists():
        return None
    df = pd.read_csv(HISTORY_TRACKED_PATH)
    if df.empty or "fetched_at" not in df or (df["season_start_year"] != season_label).any():
        return None
    return df


def _history_age_days(df: pd.DataFrame) -> float:
    descargado = pd.to_datetime(df["fetched_at"], utc=True, errors="coerce").max()
    if pd.isna(descargado):
        return float("inf")
    return (pd.Timestamp.now(tz="UTC") - descargado).total_seconds() / 86400


def _fetch_history_pool(client: GoalApiClient, current: int) -> pd.DataFrame:
    """Arma el "historial" que team_strength_for_competition pide como
    respaldo (season - 1). Se guarda con la etiqueta `current - 1` para que
    encaje sin tocar nada en team_strength.py/matchday_predictions.py — son
    ellos los que ya piden [season, season-1].

    Primero usa el archivo versionado (0 peticiones). Solo si tiene más de
    HISTORY_REFRESH_DAYS lo vuelve a descargar; y si esa descarga falla (ej.
    cuota agotada), sigue con el viejo en vez de dejar la jornada sin predecir."""
    season_label = current - 1
    guardado = _load_tracked_history(season_label)
    if guardado is not None and _history_age_days(guardado) <= HISTORY_REFRESH_DAYS:
        df = guardado
    else:
        try:
            df = _download_history_pool(client, current)
        except Exception:
            if guardado is None:
                raise
            df = guardado  # desactualizado pero válido: mejor que nada

    csv_path = config.PROCESSED_DIR / f"matches_{COMPETITION_CODE}_{season_label}.csv"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return df


def _download_history_pool(client: GoalApiClient, current: int) -> pd.DataFrame:
    """≈60 peticiones: resultados FINISHED de Nations League + amistosos de
    cada selección de la edición en curso, vía /teams/:id/fixtures (un pedido
    por selección, no por partido). Guarda el resultado en el archivo
    versionado para no repetirlo en cada carga."""
    season_label = current - 1
    print("Descargando historial de selecciones (Nations League + amistosos) vía Goal API...")
    current_fixtures = _fetch_group_stage_fixtures(client, current)

    teams: dict[str, dict] = {}
    for f in current_fixtures:
        for lado in ("home", "away"):
            tid = f.get(f"{lado}TeamId")
            if tid and tid not in teams:
                teams[tid] = f.get(f"{lado}TeamName")

    cutoff = (datetime.now(timezone.utc) - timedelta(days=HISTORY_LOOKBACK_DAYS)).date().isoformat()
    seen_match_ids: set[str] = set()
    rows = []
    for team_id in teams:
        try:
            historial = _team_history(client, team_id)
        except GoalApiRateLimited:
            # Cortar acá: guardar un historial a medias como "fresco" dejaría
            # selecciones sin datos por una semana entera.
            raise
        except Exception:
            continue  # una selección sin datos no debe tumbar a las demás (aislamiento)
        for f in historial:
            if f.get("matchStatus") != "FINISHED":
                continue
            if f.get("leagueId") not in (NATIONS_LEAGUE_ID, FRIENDLIES_LEAGUE_ID):
                continue
            match_id = f.get("id")
            if not match_id or match_id in seen_match_ids:
                continue
            if (f.get("matchDate") or "") < cutoff:
                continue
            seen_match_ids.add(match_id)
            rows.append(_row_from_fixture(f, season_label))

    df = pd.DataFrame(rows)
    df["fetched_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    HISTORY_TRACKED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HISTORY_TRACKED_PATH, index=False, encoding="utf-8")
    print(f"  {len(df)} partidos históricos guardados en "
          f"{HISTORY_TRACKED_PATH.relative_to(config.ROOT_DIR)} (de {len(teams)} selecciones).")
    return df


def fetch_competition(season: int) -> pd.DataFrame:
    """Mismo contrato que europa_league.fetch_competition (y fetch_football_data
    para las 6 ligas): guarda data/processed/matches_NT_{season}.csv y lo
    devuelve, para que el resto del pipeline no distinga de dónde vino el dato.

    A diferencia de las demás fuentes, `season` no siempre significa "pedime
    la edición de ese año": la Nations League corre cada 2 años, así que
    `season == temporada_actual - 1` (lo que matchday_predictions_df pide como
    respaldo de historial) arma en cambio el pool histórico vía
    /teams/:id/fixtures — ver _fetch_history_pool.
    """
    client = _client()
    current = current_season()
    if season == current:
        return _fetch_current_edition(client, season)
    if season == current - 1:
        return _fetch_history_pool(client, current)

    # Temporada fuera de lo que este módulo sabe ofrecer (ni la edición en
    # curso ni el respaldo histórico que arma junto a ella) — nada que hacer,
    # pero con el esquema correcto para no romper a quien encadena/concatena.
    df = pd.DataFrame(columns=[
        "match_id", "competition", "season_start_year", "matchday", "utc_date",
        "status", "home_team", "away_team", "home_team_short", "away_team_short",
        "home_goals", "away_goals", "winner",
    ])
    csv_path = config.PROCESSED_DIR / f"matches_{COMPETITION_CODE}_{season}.csv"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return df


def get_standings_map(season: int) -> dict[str, int]:
    """{selección: posición EN SU GRUPO} — no es comparable entre selecciones
    de distinto grupo/liga (A/B/C/D), a diferencia de get_standings_map de
    Europa League (una sola tabla). Sirve igual como referencia rápida en la
    columna "Posiciones". Nunca lanza: ante cualquier problema devuelve {} y
    esa columna ya sabe mostrar "-"."""
    try:
        rows = _client()._get(f"/standings/{NATIONS_LEAGUE_ID}", {"limit": 100})
    except Exception:
        return {}
    if not isinstance(rows, list):
        return {}
    posiciones = {}
    for r in rows:
        nombre = (r.get("team") or {}).get("name") or r.get("teamName")
        posicion = r.get("overallLeaguePosition")
        if nombre and posicion is not None:
            try:
                posiciones[nombre] = int(posicion)
            except (TypeError, ValueError):
                continue
    return posiciones


def get_live_matches(ensure_data=None) -> pd.DataFrame:
    """Nations League + amistosos (solo selecciones absolutas) en vivo ahora
    mismo, mismas columnas que live_matches.get_live_matches — pensado para
    concatenar directo en app.py. Nunca lanza: cualquier problema devuelve un
    DataFrame vacío."""
    try:
        todos_en_vivo = live_fixtures(GOAL_API_KEY)
    except Exception:
        return pd.DataFrame()

    propios = [
        f for f in (todos_en_vivo or [])
        if f.get("leagueId") in (NATIONS_LEAGUE_ID, FRIENDLIES_LEAGUE_ID)
        and _is_senior_team(f.get("homeTeamName")) and _is_senior_team(f.get("awayTeamName"))
    ]
    if not propios:
        return pd.DataFrame()

    season = current_season()
    try:
        if ensure_data:
            ensure_data(COMPETITION_CODE, season)
            ensure_data(COMPETITION_CODE, season - 1)
        else:
            fetch_competition(season)
            fetch_competition(season - 1)
        strength = team_strength_for_competition_safe([season, season - 1])
    except Exception:
        strength = None

    rows = []
    for f in propios:
        home = (f.get("homeTeam") or {}).get("name") or f.get("homeTeamName")
        away = (f.get("awayTeam") or {}).get("name") or f.get("awayTeamName")
        hg = pd.to_numeric(f.get("homeTeamScore"), errors="coerce")
        ag = pd.to_numeric(f.get("awayTeamScore"), errors="coerce")
        hg, ag = (0 if pd.isna(hg) else int(hg)), (0 if pd.isna(ag) else int(ag))
        minute = int(f.get("matchElapsed") or 0)
        remaining_frac = max(REGULATION_MINUTES - minute, 0) / REGULATION_MINUTES

        live_h = live_d = live_a = None
        if strength is not None:
            try:
                home_xg, away_xg = expected_goals(strength, home, away)
                live_h, live_d, live_a = live_win_probabilities(
                    hg, ag, home_xg * remaining_frac, away_xg * remaining_frac
                )
            except KeyError:
                pass

        rows.append({
            "competition": COMPETITION_CODE,
            "competition_name": config.COMPETITIONS.get(COMPETITION_CODE, COMPETITION_CODE),
            "home_team": home, "away_team": away,
            "utc_date": _normalize_utc(f.get("kickoffUtc", "")),
            "home_goals": hg, "away_goals": ag,
            "status": f.get("matchStatus"),
            "minuto_estimado": minute,
            "live_home_win": round(live_h * 100, 1) if live_h is not None else None,
            "live_draw": round(live_d * 100, 1) if live_d is not None else None,
            "live_away_win": round(live_a * 100, 1) if live_a is not None else None,
        })
    return pd.DataFrame(rows)


def team_strength_for_competition_safe(seasons: list[int]):
    """team_strength_for_competition puntual a este módulo: importado adentro
    de la función (no arriba del archivo) para evitar el ciclo
    team_strength -> config -> ... que no existe hoy, pero así queda mismo
    patrón que cross_competition_strength.py por si algún día lo hay."""
    from .team_strength import team_strength_for_competition
    return team_strength_for_competition(COMPETITION_CODE, seasons)


def fetch_friendlies_window(days_back: int = 1, days_ahead: int = 8) -> pd.DataFrame:
    """Amistosos de selecciones ABSOLUTAS (sin juveniles/femenino) entre
    `hoy - days_back` y `hoy + days_ahead`. Sin jornada ni tabla (no aplica el
    concepto), así que se muestran aparte de la vista "por jornada" — ver
    app.py, sección "Amistosos internacionales".

    Una petición por día de la ventana (fixtures/date/:fecha ya viene acotada
    a esta liga), no por partido — barato incluso con 9-10 días de ventana.
    """
    client = _client()
    season_label = current_season()  # una sola vez: no cambia partido a partido
    hoy = datetime.now(timezone.utc).date()
    rows = []
    for delta in range(-days_back, days_ahead + 1):
        fecha = (hoy + timedelta(days=delta)).isoformat()
        try:
            fixtures = _get_with_retry(
                client, f"/fixtures/date/{fecha}", {"limit": 100, "leagueId": FRIENDLIES_LEAGUE_ID}
            ) or []
        except Exception:
            continue
        for f in fixtures:
            home, away = f.get("homeTeamName"), f.get("awayTeamName")
            if not (_is_senior_team(home) and _is_senior_team(away)):
                continue
            rows.append(_row_from_fixture(f, season_label))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset="match_id")
    return df.sort_values("utc_date").reset_index(drop=True)


def predict_friendlies(fixtures: pd.DataFrame, strength) -> pd.DataFrame:
    """Corre el modelo de Poisson partido a partido sobre `fixtures` (la salida
    de fetch_friendlies_window). Selecciones sin historial propio (ninguna de
    las dos fuentes, NL ni amistosos, las tiene) se omiten en silencio — mismo
    criterio que matchday_predictions_df con equipos desconocidos."""
    from .team_strength import confidence_note

    rows = []
    for _, row in fixtures.iterrows():
        try:
            pred = predict_match(strength, row["home_team"], row["away_team"])
        except KeyError:
            continue
        note = confidence_note(strength, row["home_team"], row["away_team"])
        top_score, top_prob = pred.top_scorelines(1)[0]
        rows.append({
            "utc_date": row["utc_date"],
            "status": row["status"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_team_short": row.get("home_team_short") or row["home_team"],
            "away_team_short": row.get("away_team_short") or row["away_team"],
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
