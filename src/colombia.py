"""
Liga BetPlay Dimayor (Colombia) vía 365scores — ninguna de las otras fuentes
del proyecto la tiene (football-data.org, Goal API en su plan, The Odds API sin
región Colombia). 365scores da calendario, resultados, estadísticas y cuotas de
BWIN Colombia (verificado oct-2026). Mismo contrato de módulo que
europa_league.py / national_teams.py (ver config.goal_api_module).

Temporada = año calendario, con DOS torneos (Apertura ene-jun, Clausura
jul-dic) y sus finales. Jornadas: las fechas del Apertura van como jornada
1..N y las del Clausura siguen numerándose (N+1...), así una temporada no
repite números de jornada. Las eliminatorias (cuartos, semis, final) quedan
sin jornada: cuentan para la fuerza de los equipos pero no se predicen como
jornada (igual que la previa de la Conference).
365scores etiqueta mal el Clausura 2025 (todo como "Final"): esos partidos
también quedan sin jornada — solo se usan como historial de fuerza.

Ahorro de consultas (pedido del usuario: guardar lo ya consultado): /games/
solo acepta rangos de ~1 mes, así que se pide mes a mes; los partidos
terminados se guardan versionados en data/tracking/col_history/ y un mes ya
cerrado (todo terminado) no se vuelve a pedir.
"""
from __future__ import annotations

import calendar
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config, scores365
from .competition_status import CompetitionStatus
from .goal_api_client import memoized
from .live_matches import REGULATION_MINUTES, live_win_probabilities
from .poisson_model import expected_goals
from .team_strength import team_strength_for_competition

COMPETITION_CODE = "COL"
COMPETITION_ID = 620
HISTORY_PATH = config.ROOT_DIR / "data" / "tracking" / "col_history" / "COL_fixtures.csv"
REGULAR_STAGES = ("Apertura", "Clausura")
STATUS = {2: "SCHEDULED", 3: "IN_PLAY", 4: "FINISHED"}
FIXTURES_MEMO_SECONDS = 15 * 60
HISTORY_COLUMNS = ["id", "start_utc", "status_group", "status_text", "home_team", "away_team", "home_id", "away_id",
                   "home_goals", "away_goals", "stage", "round_num", "round_name"]


def _flatten(g: dict) -> dict:
    local, visita = g.get("homeCompetitor") or {}, g.get("awayCompetitor") or {}
    return {
        "id": g.get("id"), "start_utc": scores365._utc(g.get("startTime")), "status_group": g.get("statusGroup"),
        "status_text": g.get("statusText"), "home_team": local.get("name"), "away_team": visita.get("name"),
        "home_id": local.get("id"), "away_id": visita.get("id"),
        "home_goals": scores365._score(local), "away_goals": scores365._score(visita),
        "stage": g.get("stageName"), "round_num": g.get("roundNum"), "round_name": g.get("roundName"),
    }


def _load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.read_csv(HISTORY_PATH)


def _save_history(df: pd.DataFrame) -> None:
    terminados = df[df["status_group"] == 4]
    if terminados.empty:
        return
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    terminados.drop_duplicates("id", keep="last").sort_values("start_utc").reindex(columns=HISTORY_COLUMNS) \
        .to_csv(HISTORY_PATH, index=False, encoding="utf-8")


def _month_closed(hist: pd.DataFrame, y: int, m: int, now: datetime) -> bool:
    """Mes ya terminado hace >3 días y con partidos guardados (todos terminados)."""
    fin = datetime(y, m, calendar.monthrange(y, m)[1], 23, 59, tzinfo=timezone.utc)
    if fin > now - timedelta(days=3) or hist.empty:
        return False
    return hist["start_utc"].astype(str).str.startswith(f"{y}-{m:02d}").any()


def _download_year(year: int) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    hist = _load_history()
    frescos = []
    for m in range(1, 13):
        if _month_closed(hist, year, m, now):
            continue
        ult = calendar.monthrange(year, m)[1]
        data = scores365._get("/games/", {"competitions": COMPETITION_ID,
                                          "startDate": f"01/{m:02d}/{year}", "endDate": f"{ult:02d}/{m:02d}/{year}"})
        frescos += [_flatten(g) for g in (data or {}).get("games", [])]
    todo = pd.concat([hist, pd.DataFrame(frescos, columns=HISTORY_COLUMNS)], ignore_index=True) \
        .drop_duplicates("id", keep="last")
    try:
        _save_history(todo)
    except OSError:
        pass
    return todo[todo["start_utc"].astype(str).str.startswith(str(year))]


def season_fixtures(year: int) -> pd.DataFrame:
    return memoized(f"COL_{year}", FIXTURES_MEMO_SECONDS, lambda: _download_year(year))


def current_season() -> int:
    return datetime.now(timezone.utc).year


def _matchdays(df: pd.DataFrame) -> pd.Series:
    """Jornada: fecha del Apertura tal cual; del Clausura, sumando el total del Apertura."""
    regular = df["stage"].isin(REGULAR_STAGES) & df["round_num"].notna()
    apertura_max = df.loc[regular & (df["stage"] == "Apertura"), "round_num"].max()
    offset = int(apertura_max) if pd.notna(apertura_max) else 0
    md = df["round_num"].where(regular)
    md = md.where(df["stage"] != "Clausura", md + offset)
    return md.astype("Int64")


def fetch_competition(season: int) -> pd.DataFrame:
    """Guarda la temporada en el formato y ruta de fetch_football_data
    (data/processed/matches_COL_{season}.csv). No atrapa excepciones."""
    print(f"Descargando Liga BetPlay (temporada {season}) vía 365scores...")
    df = season_fixtures(season).copy()
    rows = pd.DataFrame({
        "match_id": "365:" + df["id"].astype(str), "competition": COMPETITION_CODE, "season_start_year": season,
        "matchday": _matchdays(df) if not df.empty else pd.Series(dtype="Int64"), "stage": df["stage"],
        "utc_date": df["start_utc"], "status": df["status_group"].map(STATUS).fillna("POSTPONED"),
        "home_team": df["home_team"], "away_team": df["away_team"],
        "home_team_short": df["home_team"], "away_team_short": df["away_team"],
        "home_goals": pd.to_numeric(df["home_goals"], errors="coerce"),
        "away_goals": pd.to_numeric(df["away_goals"], errors="coerce"),
        "home_ht_goals": pd.NA, "away_ht_goals": pd.NA,  # 365scores no da el descanso en el listado
    })
    rows["winner"] = None
    fin = rows["status"] == "FINISHED"
    rows.loc[fin & (rows["home_goals"] > rows["away_goals"]), "winner"] = "HOME_TEAM"
    rows.loc[fin & (rows["home_goals"] < rows["away_goals"]), "winner"] = "AWAY_TEAM"
    rows.loc[fin & (rows["home_goals"] == rows["away_goals"]), "winner"] = "DRAW"
    rows = rows.sort_values("utc_date")
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PROCESSED_DIR / f"matches_{COMPETITION_CODE}_{season}.csv"
    rows.to_csv(path, index=False, encoding="utf-8")
    print(f"  {len(rows)} partidos guardados en {path.relative_to(config.ROOT_DIR)} "
          f"({int(fin.sum())} finalizados).")
    return rows


def resolve_current_season_and_matchday() -> CompetitionStatus:
    season = current_season()
    df = season_fixtures(season)
    if df.empty:
        return CompetitionStatus(season=season, matchday=1, season_start="", season_end="",
                                 reason="Sin calendario de la Liga BetPlay en 365scores.")
    md = _matchdays(df)
    con_jornada = df.assign(md=md, inicio=pd.to_datetime(df["start_utc"], utc=True, errors="coerce"))[md.notna()]
    # La jornada del PRÓXIMO partido (o del que está en juego), no la primera con
    # pendientes: en Colombia se aplazan partidos sueltos y quedan "colgados" en
    # fechas viejas (ej. 8 partidos de la fecha 5 del Clausura 2026 movidos a
    # fines de octubre, cuando ya se jugaba la 17).
    ahora = pd.Timestamp.now(tz="UTC")
    vigentes = con_jornada[(con_jornada["status_group"] != 4) & (con_jornada["inicio"] > ahora - pd.Timedelta(hours=3))]
    if not vigentes.empty:
        jornada = int(vigentes.sort_values("inicio").iloc[0]["md"])
        return CompetitionStatus(season=season, matchday=jornada, season_start="", season_end="",
                                 reason=f"Jornada {jornada}: la del próximo partido según 365scores.")
    ultima = int(con_jornada["md"].max()) if not con_jornada.empty else 1
    return CompetitionStatus(season=season, matchday=ultima, season_start="", season_end="",
                             reason="Fase de todos contra todos terminada (siguen eliminatorias o pausa).")


def get_standings_map(season: int) -> dict[str, int]:
    """{equipo: posición} del torneo en curso. {} ante cualquier problema."""
    try:
        data = scores365._get("/standings/", {"competitions": COMPETITION_ID}) or {}
        filas = ((data.get("standings") or [{}])[0]).get("rows") or []
        return {r["competitor"]["name"]: int(r["position"]) for r in filas if r.get("competitor") and r.get("position")}
    except Exception:
        return {}


def get_live_matches(ensure_data=None) -> pd.DataFrame:
    """Partidos en juego de la Liga BetPlay — mismas columnas que live_matches.get_live_matches."""
    try:
        hoy = datetime.now(timezone.utc)
        data = scores365._get("/games/", {"competitions": COMPETITION_ID,
                                          "startDate": (hoy - timedelta(days=1)).strftime("%d/%m/%Y"),
                                          "endDate": hoy.strftime("%d/%m/%Y")})
        en_juego = [g for g in (data or {}).get("games", []) if g.get("statusGroup") == 3]
    except Exception:
        return pd.DataFrame()
    if not en_juego:
        return pd.DataFrame()
    season = current_season()
    try:
        for s in (season, season - 1):
            (ensure_data or (lambda c, y: fetch_competition(y)))(COMPETITION_CODE, s)
        strength = team_strength_for_competition(COMPETITION_CODE, [season, season - 1])
    except Exception:
        strength = None
    filas = []
    for g in en_juego:
        f = _flatten(g)
        hg, ag = f["home_goals"] or 0, f["away_goals"] or 0
        minuto = int(float(g.get("gameTime") or 0)) if (g.get("gameTime") or 0) >= 0 else 0
        resto = max(REGULATION_MINUTES - minuto, 0) / REGULATION_MINUTES
        lh = ld = la = home_xg = away_xg = None
        if strength is not None:
            try:
                home_xg, away_xg = expected_goals(strength, f["home_team"], f["away_team"])
                lh, ld, la = live_win_probabilities(hg, ag, home_xg * resto, away_xg * resto)
            except KeyError:
                pass
        filas.append({
            "competition": COMPETITION_CODE, "competition_name": config.COMPETITIONS.get(COMPETITION_CODE),
            "home_team": f["home_team"], "away_team": f["away_team"], "utc_date": f["start_utc"],
            "home_goals": hg, "away_goals": ag, "status": "IN_PLAY", "minuto_estimado": minuto,
            "live_home_win": round(lh * 100, 1) if lh is not None else None,
            "live_draw": round(ld * 100, 1) if ld is not None else None,
            "live_away_win": round(la * 100, 1) if la is not None else None,
            "pre_home_xg": home_xg, "pre_away_xg": away_xg,
        })
    return pd.DataFrame(filas)
