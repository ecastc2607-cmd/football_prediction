"""
Copa BetPlay Dimayor (Colombia) vía 365scores — mismo contrato de módulo que
colombia.py (ver config.goal_api_module), pedido del usuario (nov-2026).

No tiene modelo propio: cada equipo juega pocos partidos de copa (fase de
grupos y eliminatorias a ida y vuelta, con rotación), así que se predice con
la fuerza de los equipos en la Liga BetPlay (config.STRENGTH_FROM). Los de
Primera B no tienen esa fuerza y quedan sin predicción. El modelo tampoco sabe
el marcador global de una eliminatoria: en la vuelta, un equipo al que le
sirve el empate puede no salir a ganar.

Jornadas: cada fecha de la fase de grupos es una jornada y cada ronda de
eliminación (octavos, cuartos, semis, final) otra, con ida y vuelta juntas —
así la jornada de semifinales muestra las cuatro.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from . import colombia, config, scores365
from .competition_status import CompetitionStatus
from .goal_api_client import memoized
from .live_matches import REGULATION_MINUTES, live_win_probabilities
from .poisson_model import expected_goals
from .team_strength import team_strength_for_competition

COMPETITION_CODE = "CBP"
COMPETITION_ID = 616
HISTORY_PATH = config.ROOT_DIR / "data" / "tracking" / "col_history" / "CBP_fixtures.csv"
STATUS = colombia.STATUS


def season_fixtures(year: int) -> pd.DataFrame:
    return memoized(f"CBP_{year}", colombia.FIXTURES_MEMO_SECONDS,
                    lambda: colombia._download_year(year, COMPETITION_ID, HISTORY_PATH))


def current_season() -> int:
    return datetime.now(timezone.utc).year


def _matchdays(df: pd.DataFrame) -> pd.Series:
    """Una jornada por fecha de grupos o por ronda de eliminación, en orden de fecha."""
    if df.empty:
        return pd.Series(dtype="Int64")
    ronda = df["stage"].fillna("Fase de grupos").astype(str) + "|" + df["round_num"].fillna(-1).astype(str)
    inicio = pd.to_datetime(df["start_utc"], utc=True, errors="coerce")
    orden = inicio.groupby(ronda).min().sort_values()
    numero = {r: i + 1 for i, r in enumerate(orden.index)}
    return ronda.map(numero).astype("Int64")


def fetch_competition(season: int) -> pd.DataFrame:
    """Guarda la temporada en data/processed/matches_CBP_{season}.csv (mismo
    formato que colombia.fetch_competition). No atrapa excepciones."""
    print(f"Descargando Copa BetPlay (temporada {season}) vía 365scores...")
    df = season_fixtures(season).copy()
    rows = pd.DataFrame({
        "match_id": "365:" + df["id"].astype(str), "competition": COMPETITION_CODE, "season_start_year": season,
        "matchday": _matchdays(df), "stage": df["stage"],
        "utc_date": df["start_utc"], "status": df["status_group"].map(STATUS).fillna("POSTPONED"),
        "home_team": df["home_team"], "away_team": df["away_team"],
        "home_team_short": df["home_team"], "away_team_short": df["away_team"],
        "home_goals": pd.to_numeric(df["home_goals"], errors="coerce"),
        "away_goals": pd.to_numeric(df["away_goals"], errors="coerce"),
        "home_ht_goals": pd.NA, "away_ht_goals": pd.NA,
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
                                 reason="Sin calendario de la Copa BetPlay en 365scores.")
    d = df.assign(md=_matchdays(df), inicio=pd.to_datetime(df["start_utc"], utc=True, errors="coerce"))
    ahora = pd.Timestamp.now(tz="UTC")
    vigentes = d[(d["status_group"] != 4) & (d["inicio"] > ahora - pd.Timedelta(hours=3))]
    if not vigentes.empty:
        jornada = int(vigentes.sort_values("inicio").iloc[0]["md"])
        return CompetitionStatus(season=season, matchday=jornada, season_start="", season_end="",
                                 reason=f"Ronda {jornada}: la del próximo partido según 365scores.")
    return CompetitionStatus(season=season, matchday=int(d["md"].max()), season_start="", season_end="",
                             reason="Copa terminada o en pausa.")


def get_standings_map(season: int) -> dict[str, int]:
    """Las eliminatorias no tienen tabla: posiciones de la Liga BetPlay como referencia."""
    return colombia.get_standings_map(season)


def get_live_matches(ensure_data=None) -> pd.DataFrame:
    """Partidos en juego de la Copa BetPlay, con la fuerza de la Liga BetPlay."""
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
            (ensure_data or (lambda c, y: colombia.fetch_competition(y)))(colombia.COMPETITION_CODE, s)
        strength = team_strength_for_competition(colombia.COMPETITION_CODE, [season, season - 1])
    except Exception:
        strength = None
    filas = []
    for g in en_juego:
        f = colombia._flatten(g)
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
