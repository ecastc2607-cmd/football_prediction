"""
Predicciones del MISMO día: loguearlas antes de que empiecen y resolverlas
apenas terminan, sin esperar a la tarea de los lunes (pedido del usuario,
oct-2026: "quiero ver las predicciones de hoy en Reportes").

  - log_day(): todos los partidos del día (de los CSV locales), antes del
    inicio — log_predictions ya descarta lo que empezó, así que nunca entra un
    resultado conocido. Lo corre la captura automática (capture_live_odds.py)
    al abrirse la ventana de partidos, 3 h antes del primero.
  - resolve_finished(): marcador final de 365scores (gratis) para los
    pendientes que ya terminaron -> resultado, goles, ambos anotan y, si ya
    están guardadas las estadísticas finales, corners/faltas/tarjetas.
    resolved_source="365scores": la tarea de los lunes completa después lo
    que falte (marcador al descanso, estadísticas) con su fuente habitual.
  - backfill_day(): SOLO para un día ya jugado que no se logueó a tiempo; usa
    datos previos al partido y se marca con source "backtest_mismo_dia_v1"
    (cuenta como backtest en Reportes, nunca como predicción a tiempo).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config
from .log_predictions import log_competition
from .odds_api import find_event
from .prediction_log import append_predictions, load_log, result_letter, save_log
from .timezones import to_bogota

BACKFILL_SOURCE = "backtest_mismo_dia_v1"
FINISHED_AFTER = timedelta(minutes=150)


def _day_fixtures(fecha: str) -> list[tuple[str, int, int, set]]:
    """[(competición, temporada, jornada, {(local, visitante)})] del día (Colombia)."""
    salida = []
    for code in config.COMPETITIONS:
        archivos = sorted(config.PROCESSED_DIR.glob(f"matches_{code}_*.csv"), reverse=True)[:1]
        if not archivos:
            continue
        df = pd.read_csv(archivos[0])
        if "matchday" in df.columns:
            df = df[df["matchday"].notna()]
        dia = df["utc_date"].map(lambda d: to_bogota(d).date().isoformat() if to_bogota(d) else "")
        for (season, md), partidos in df[dia == fecha].groupby(["season_start_year", "matchday"]):
            salida.append((code, int(season), int(md), set(zip(partidos["home_team"], partidos["away_team"]))))
    return salida


def _rows_for_day(fecha: str, only_future: bool) -> list[dict]:
    filas = []
    for code, season, md, partidos in _day_fixtures(fecha):
        try:
            rows = log_competition(code, season, md, only_future=only_future)
        except Exception:
            continue
        filas += [r for r in rows if (r["home_team"], r["away_team"]) in partidos]
    return filas


def log_day(fecha: str) -> int:
    """Loguea los partidos del día que todavía no empezaron. Devuelve cuántos agregó."""
    agregados, _ = append_predictions(_rows_for_day(fecha, only_future=True))
    return agregados


def backfill_day(fecha: str) -> int:
    filas = [dict(r, source=BACKFILL_SOURCE) for r in _rows_for_day(fecha, only_future=False)]
    agregados, _ = append_predictions(filas)
    return agregados


def resolve_finished(now: datetime | None = None) -> int:
    """Resuelve con 365scores los pendientes cuyo partido ya terminó. Nunca lanza."""
    try:
        from .resolve_predictions import _resolve_stat_markets
        from .tips_log import _final_scores

        log = load_log()
        if log.empty:
            return 0
        now = now or datetime.now(timezone.utc)
        inicio = pd.to_datetime(log["utc_date"], utc=True, errors="coerce")
        pendientes = log.index[(log["status"] == "pending") & inicio.notna() & (inicio + FINISHED_AFTER < now)]
        if len(pendientes) == 0:
            return 0
        for col in ("actual_result", "hit", "actual_over_2_5", "hit_over_2_5", "actual_btts", "hit_btts",
                    "status", "resolved_at", "resolved_source", "actual_corners_total", "hit_corners",
                    "actual_faltas_total", "hit_faltas", "actual_tarjetas_total", "hit_tarjetas"):
            log[col] = log[col].astype(object)
        cache: dict = {}
        resueltos = 0
        for i in pendientes:
            r = log.loc[i]
            clave = (r["competition"], inicio[i].strftime("%Y-%m-%d"))
            if clave not in cache:
                cache[clave] = _final_scores(*clave)
            ev = find_event(cache[clave], r["home_team"], r["away_team"], r["utc_date"]) if cache[clave] else None
            if not ev:
                continue
            hg, ag = int(ev["hg"]), int(ev["ag"])
            real = result_letter(hg, ag)
            log.loc[i, ["actual_home_goals", "actual_away_goals", "actual_result"]] = [hg, ag, real]
            log.loc[i, "hit"] = r["favored_side"] == real
            if pd.notna(r["over_2_5_pct"]):
                log.loc[i, "actual_over_2_5"] = hg + ag > 2.5
                log.loc[i, "hit_over_2_5"] = (r["over_2_5_pct"] >= 50) == (hg + ag > 2.5)
            if pd.notna(r["btts_pct"]):
                ambos = hg > 0 and ag > 0
                log.loc[i, "actual_btts"] = ambos
                log.loc[i, "hit_btts"] = (r["btts_pct"] >= 50) == ambos
            for campo, valor in _resolve_stat_markets(r["competition"], r["home_team"], r["away_team"],
                                                      r["utc_date"], r).items():
                log.loc[i, campo] = valor
            log.loc[i, "status"] = "resolved"
            log.loc[i, "resolved_source"] = "365scores"
            log.loc[i, "resolved_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            resueltos += 1
        if resueltos:
            save_log(log)
        return resueltos
    except Exception:
        return 0
