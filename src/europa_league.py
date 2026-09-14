"""
Europa League vía Goal API — football-data.org NO la incluye en el plan gratis
(verificado contra su API y su tabla de cobertura: hace falta su plan de
€49/mes). Goal API sí la tiene completa (fixtures, resultados, tabla), así que
esta es una SEGUNDA fuente de datos, aislada a propósito en este único archivo.

Regla explícita del usuario: si esta fuente falla o cambia de forma, NUNCA debe
tumbar el flujo de las otras 6 competiciones (que corren sobre football-data.org).
Por eso:
  - Todo lo específico de Goal-API-como-fuente-de-partidos vive aquí adentro,
    no mezclado en fetch_football_data.py/live_matches.py/competition_status.py.
  - Cada función de acá está pensada para fallar "limpio" (devolver vacío/None,
    o dejar pasar una excepción clara) y quien la llama en app.py la envuelve
    en su propio try/except antes de tocar cualquier estado compartido con las
    otras ligas.

Una vez que fetch_competition() guarda el CSV en el mismo formato y ruta que
fetch_football_data.py (data/processed/matches_EL_{season}.csv), el resto del
pipeline (team_strength, poisson_model, matchday_predictions, predict_matchday,
parlay_builder, match_tendencies...) no necesita saber de dónde vino el dato:
todos leen ese CSV igual para las 7 competiciones.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from . import config
from .competition_status import CompetitionStatus
from .goal_api_client import GOAL_API_KEY, GoalApiClient, LEAGUE_IDS
from .live_matches import REGULATION_MINUTES, live_win_probabilities
from .poisson_model import expected_goals
from .team_strength import team_strength_for_competition

COMPETITION_CODE = "EL"
LEAGUE_ID = LEAGUE_IDS[COMPETITION_CODE]

# La fase de liga (8 jornadas, formato "una tabla" desde la reforma 2024/25) es
# la que corresponde al concepto de "jornada" que usa el resto del proyecto —
# se ignoran clasificación previa y eliminatorias, igual que el proyecto ya
# trata la Champions League solo en su fase de liga.
LEAGUE_PHASE_STAGE = "League Phase"


def _client() -> GoalApiClient:
    return GoalApiClient(api_key=GOAL_API_KEY)


def _get_with_retry(client: GoalApiClient, path: str, params: dict, attempts: int = 2):
    """Un reintento ante un timeout de red puntual — /leagues/:id/fixtures trae
    todo el historial de la competición (~12 páginas para la Europa League), y
    sin esto un solo timeout a mitad de camino tira toda la descarga (nos pasó
    en pruebas). No reintenta GoalApiRateLimited ni otros errores del propio API:
    solo problemas de red/tiempo, donde repetir sí tiene sentido."""
    for intento in range(attempts):
        try:
            return client._get(path, params)
        except requests.exceptions.Timeout:
            if intento == attempts - 1:
                raise
            time.sleep(1.5)


def _paginate(client: GoalApiClient, path: str, page_size: int = 100, hard_limit: int = 3000) -> list[dict]:
    """Junta todas las páginas de un endpoint de lista de Goal API."""
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
        if offset > hard_limit:  # cortafuegos: no debería hacer falta tantas páginas
            break
    return items


def _normalize_utc(kickoff_iso: str) -> str:
    """'2026-09-16T16:45:00.000Z' -> '2026-09-16T16:45:00Z' — el resto del
    pipeline (estimate_elapsed_minutes, exports) asume el formato exacto de
    football-data.org, sin milisegundos."""
    return kickoff_iso[:19] + "Z"


def current_season() -> int:
    """Temporada que Goal API considera vigente para la Europa League ahora
    mismo (su propio campo 'season' en /leagues/:id, ej. "2026/2027" -> 2026),
    igual de espíritu a currentSeason en football-data.org."""
    data = _client()._get(f"/leagues/{LEAGUE_ID}", {})
    season_label = (data or {}).get("season", "")
    if not season_label or "/" not in season_label:
        return datetime.now(timezone.utc).year
    return int(season_label.split("/")[0])


def _fetch_league_phase_fixtures(client: GoalApiClient, season: int) -> list[dict]:
    all_fixtures = _paginate(client, f"/leagues/{LEAGUE_ID}/fixtures")
    year_label = f"{season}/{season + 1}"
    return [
        f for f in all_fixtures
        if f.get("leagueYear") == year_label and f.get("stageName") == LEAGUE_PHASE_STAGE
    ]


def resolve_current_season_and_matchday() -> CompetitionStatus:
    """Igual que competition_status.resolve_current_season_and_matchday, pero
    sin depender de football-data.org: la "jornada actual" es la primera del
    calendario que no está 100% FINISHED todavía."""
    season = current_season()
    client = _client()
    fixtures = _fetch_league_phase_fixtures(client, season)

    if not fixtures:
        return CompetitionStatus(
            season=season, matchday=1, season_start="", season_end="",
            reason="Aún no hay calendario de la fase de liga para esta temporada (Goal API).",
        )

    rounds = sorted({int(f["matchRound"]) for f in fixtures if f.get("matchRound")})
    for md in rounds:
        md_fixtures = [f for f in fixtures if int(f.get("matchRound", -1)) == md]
        if not all(f.get("matchStatus") == "FINISHED" for f in md_fixtures):
            return CompetitionStatus(
                season=season, matchday=md, season_start="", season_end="",
                reason=f"Jornada {md} en curso según Goal API (fase de liga).",
            )

    # Todas las jornadas conocidas ya terminaron: se queda en la última hasta que
    # Goal API publique más fechas (la fase de liga se calendariza de a poco).
    ultima = rounds[-1] if rounds else 1
    return CompetitionStatus(
        season=season, matchday=ultima, season_start="", season_end="",
        reason=f"La fase de liga llegó hasta la jornada {ultima} según los datos disponibles.",
    )


def fetch_competition(season: int) -> pd.DataFrame:
    """Descarga y guarda la fase de liga de la Europa League para `season`, en
    el MISMO formato y ruta que fetch_football_data.fetch_competition — así el
    resto del pipeline no distingue de dónde vino el dato.

    No atrapa excepciones: si Goal API falla, se lo hace saber a quien llama
    (app.py) para que decida cómo aislarlo, en vez de guardar un CSV a medias.
    """
    print(f"Descargando Europa League (temporada {season}) vía Goal API...")
    client = _client()
    fixtures = _fetch_league_phase_fixtures(client, season)

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = config.RAW_DIR / f"{COMPETITION_CODE}_matches_{season}.json"
    raw_path.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for f in fixtures:
        home_goals = pd.to_numeric(f.get("homeTeamScore"), errors="coerce")
        away_goals = pd.to_numeric(f.get("awayTeamScore"), errors="coerce")
        winner = None
        if f.get("matchStatus") == "FINISHED" and pd.notna(home_goals) and pd.notna(away_goals):
            winner = "HOME_TEAM" if home_goals > away_goals else "AWAY_TEAM" if away_goals > home_goals else "DRAW"
        rows.append({
            "match_id": f.get("id"),
            "competition": COMPETITION_CODE,
            "season_start_year": season,
            "matchday": int(f["matchRound"]) if f.get("matchRound") else None,
            "utc_date": _normalize_utc(f.get("kickoffUtc", "")),
            "status": f.get("matchStatus"),
            # Goal API da el nombre completo del equipo anidado y uno corto al
            # nivel superior del fixture — al revés de football-data.org (donde
            # el corto está anidado), pero el mismo PAPEL en el esquema.
            "home_team": (f.get("homeTeam") or {}).get("name") or f.get("homeTeamName"),
            "away_team": (f.get("awayTeam") or {}).get("name") or f.get("awayTeamName"),
            "home_team_short": f.get("homeTeamName"),
            "away_team_short": f.get("awayTeamName"),
            "home_goals": home_goals,
            "away_goals": away_goals,
            "winner": winner,
        })
    df = pd.DataFrame(rows)

    csv_path = config.PROCESSED_DIR / f"matches_{COMPETITION_CODE}_{season}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    finished = (df["status"] == "FINISHED").sum() if not df.empty else 0
    print(f"  {len(df)} partidos guardados en {csv_path.relative_to(config.ROOT_DIR)} "
          f"({finished} finalizados, {len(df) - finished} pendientes/en curso).")
    return df


def get_standings_map(season: int) -> dict[str, int]:
    """{nombre_equipo: posición} de la fase de liga — mismo contrato que
    match_context.get_standings_map. Devuelve {} ante cualquier problema, nunca
    lanza: la columna "Posiciones" ya sabe mostrar "-" cuando esto viene vacío."""
    try:
        rows = _client()._get(f"/standings/{LEAGUE_ID}", {"limit": 100})
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
    """Partidos de Europa League en vivo ahora mismo, con el 1X2 recalculado —
    mismas columnas que live_matches.get_live_matches, para poder concatenar
    ambos resultados directo en app.py.

    A diferencia de las otras 6 ligas (donde el minuto es una estimación desde
    la hora de inicio), Goal API sí entrega el minuto real (`matchElapsed`), así
    que acá "minuto_estimado" es en realidad exacto — se deja el mismo nombre
    de columna para no bifurcar el resto del código.

    Nunca lanza: cualquier problema devuelve un DataFrame vacío, ya que quien
    llama (app.py) espera poder ignorar esta fuente sin que rompa las demás.
    """
    try:
        client = _client()
        todos_en_vivo = client._get("/fixtures/live", {"limit": 100})
    except Exception:
        return pd.DataFrame()

    propios = [f for f in (todos_en_vivo or []) if f.get("leagueId") == LEAGUE_ID]
    if not propios:
        return pd.DataFrame()

    season = current_season()
    try:
        if ensure_data:
            ensure_data(COMPETITION_CODE, season)
        else:
            fetch_competition(season)
        strength = team_strength_for_competition(COMPETITION_CODE, [season])
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
