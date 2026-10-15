"""
Captura automática de cuotas (365scores / BWIN) para la futura estrategia de
apuestas en vivo — pensada para correr cada 5 minutos con el Programador de
tareas de Windows (ver tools/install_live_odds_task.ps1).

En cada corrida:
  1. Partidos EN JUEGO de las competiciones del proyecto: cuota en vivo +
     estadísticas en vivo de 365scores (xG real, remates, rojas...) + la
     probabilidad del modelo ajustada en vivo en ese minuto -> se guardan en
     data/tracking/live_odds_snapshots.csv (solo si la cuota o el marcador
     cambiaron) y en live_snapshots.csv (una foto cada 10' de juego, para
     evaluar el ajuste en vivo).
  2. Partidos que empiezan en las próximas PREMATCH_HOURS: cuota prepartido
     (odds_snapshots.csv, máx. una consulta cada 20 min por competición). La
     última antes del inicio es la "cuota de cierre" que ancla el ajuste en vivo.
  3. Partidos ya terminados: corners/faltas/tarjetas FINALES de 365scores en
     match_stats_log.csv (si no estaban), para no pedirlos a Goal API.
  4. Si ya no hay nada en juego y esos archivos cambiaron: commit + push.

No usa football-data.org ni Goal API (cero cuota): marcador, minuto y
estadísticas salen de 365scores, y la fuerza de los equipos de los CSV locales.
Sin partidos en juego ni por empezar, hace UNA consulta y termina.

Uso manual:
    python -m src.capture_live_odds
"""
from __future__ import annotations

import subprocess
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config, live_odds, odds_api, scores365
from .live_matches import REGULATION_MINUTES, adjusted_live_probabilities, live_win_probabilities
from .live_snapshots import record as record_live_snapshot
from .market_anchor import anchored_prematch_xg
from .match_stats_log import get_cached_stats, log_match_stats
from .matchday_predictions import matchday_predictions_df

LOG_PATH = config.ROOT_DIR / "logs" / "capture_live_odds.log"
PREMATCH_HOURS = 3
TRACKED_FILES = [
    "data/tracking/match_stats_log.csv",
    "data/tracking/live_odds_snapshots.csv",
    "data/tracking/odds_snapshots.csv",
    "data/tracking/live_snapshots.csv",
]


def _log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {msg}\n")


def _todays_games() -> list[dict]:
    """Partidos de ayer a mañana de todas las competiciones (UNA consulta)."""
    hoy = datetime.now(timezone.utc)
    data = scores365._get("/games/", {
        "competitions": ",".join(str(i) for i in scores365.COMPETITION_IDS.values()),
        "startDate": (hoy - timedelta(days=1)).strftime("%d/%m/%Y"),
        "endDate": (hoy + timedelta(days=1)).strftime("%d/%m/%Y"),
    })
    return (data or {}).get("games", [])


# --- Probabilidad del modelo para un partido de 365scores ---

_PRED_CACHE: dict = {}


def _our_fixture(code: str, home: str, away: str, kickoff: str) -> dict | None:
    """Fila del CSV local (temporada, jornada, nombres nuestros) de ese partido."""
    archivos = sorted(config.PROCESSED_DIR.glob(f"matches_{code}_*.csv"), reverse=True)[:1]
    if not archivos:
        return None
    df = pd.read_csv(archivos[0])
    if "matchday" in df.columns:
        df = df[df["matchday"].notna()]
    eventos = [dict(r, commence_time=r["utc_date"]) for r in df.to_dict("records")]
    # find_event compara cada "evento" (acá: nuestros partidos) contra los
    # nombres de 365scores; el criterio (hora + parecido de nombres) es simétrico.
    return odds_api.find_event(eventos, home, away, kickoff)


def _prematch_xg(code: str, fixture: dict) -> tuple[float, float] | None:
    clave = (code, int(fixture["season_start_year"]), int(fixture["matchday"]))
    if clave not in _PRED_CACHE:
        try:
            _PRED_CACHE[clave] = matchday_predictions_df(code, clave[1], clave[2], seasons_back=1,
                                                         include_played=True)
        except Exception:
            _PRED_CACHE[clave] = pd.DataFrame()
    preds = _PRED_CACHE[clave]
    if preds.empty:
        return None
    fila = preds[(preds["home_team"] == fixture["home_team"]) & (preds["away_team"] == fixture["away_team"])]
    if fila.empty:
        return None
    return float(fila.iloc[0]["home_xg"]), float(fila.iloc[0]["away_xg"])


def capture_live(rows: list[dict]) -> int:
    guardadas = 0
    for r in rows:
        code = r.get("competition")
        stats = {}
        try:
            stats = scores365.live_stats(int(r["event_id"].split(":")[1]), r.get("home_id"), r.get("away_id"))
        except Exception:
            pass
        for lado in ("home", "away"):
            for k, v in (stats.get(lado) or {}).items():
                r[f"{lado}_{k}"] = v
        r["origin"] = "capture_task"

        hg, ag = r.get("home_score") or 0, r.get("away_score") or 0
        minuto = int(float(r.get("game_minute") or 0))
        fixture = _our_fixture(code, r["home_team"], r["away_team"], r["commence_time"])
        pre = _prematch_xg(code, fixture) if fixture else None
        if pre:
            base_l, base_v, ancla = anchored_prematch_xg(
                pre[0], pre[1], code, fixture["home_team"], fixture["away_team"], fixture["utc_date"],
                fixture.get("home_team_short") or "", fixture.get("away_team_short") or "")
            ajuste = adjusted_live_probabilities(hg, ag, minuto, base_l, base_v,
                                                 stats.get("home"), stats.get("away"))
            r.update(model_home=round(ajuste["home_win"], 4), model_draw=round(ajuste["draw"], 4),
                     model_away=round(ajuste["away_win"], 4),
                     model_home_xg_rem=round(ajuste["home_xg_remaining"], 3),
                     model_away_xg_rem=round(ajuste["away_xg_remaining"], 3))
            resto = max(REGULATION_MINUTES - minuto, 0) / REGULATION_MINUTES
            ph, pdr, pa = live_win_probabilities(hg, ag, pre[0] * resto, pre[1] * resto)
            sl, sv = stats.get("home") or {}, stats.get("away") or {}
            record_live_snapshot({
                "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "competition": code, "home_team": fixture["home_team"], "away_team": fixture["away_team"],
                "utc_date": fixture["utc_date"], "minute": minuto, "home_goals": hg, "away_goals": ag,
                "pre_home_xg": pre[0], "pre_away_xg": pre[1],
                "base_home_xg": base_l, "base_away_xg": base_v, "market_anchor": ancla is not None,
                "home_shots": sl.get("remates_totales"), "home_shots_on": sl.get("remates_a_puerta"),
                "away_shots": sv.get("remates_totales"), "away_shots_on": sv.get("remates_a_puerta"),
                "pre_home": ph, "pre_draw": pdr, "pre_away": pa,
                "adj_home": ajuste["home_win"], "adj_draw": ajuste["draw"], "adj_away": ajuste["away_win"],
            })
        guardadas += live_odds.record([r])
    return guardadas


def capture_final_stats(games: list[dict]) -> int:
    """Corners/faltas/tarjetas FINALES de los partidos ya terminados (365scores,
    gratis) que todavía no estén en match_stats_log.csv: así "Resultados ya
    jugados" y las tendencias no tienen que pedirlos a Goal API."""
    ids = {v: k for k, v in scores365.COMPETITION_IDS.items()}
    guardados = 0
    for g in games:
        code = ids.get(g.get("competitionId"))
        if g.get("statusGroup") != scores365.STATUS_FINISHED or not code:
            continue
        local, visita = g.get("homeCompetitor") or {}, g.get("awayCompetitor") or {}
        fixture = _our_fixture(code, local.get("name"), visita.get("name"), scores365._utc(g.get("startTime")))
        if not fixture or get_cached_stats(code, fixture["home_team"], fixture["away_team"], fixture["utc_date"]):
            continue
        stats = scores365.live_stats(g["id"], local.get("id"), visita.get("id"))
        if not stats:
            continue
        log_match_stats(code, int(fixture["season_start_year"]), int(fixture["matchday"]), fixture["utc_date"],
                        fixture["home_team"], fixture["away_team"],
                        {fixture["home_team"]: stats["home"], fixture["away_team"]: stats["away"]})
        guardados += 1
    return guardados


def capture_prematch(games: list[dict]) -> int:
    """Cuota prepartido de las competiciones con partidos por empezar pronto."""
    ahora = datetime.now(timezone.utc)
    limite = ahora + timedelta(hours=PREMATCH_HOURS)
    ids = {v: k for k, v in scores365.COMPETITION_IDS.items()}
    por_comp: dict[str, set[str]] = {}
    for g in games:
        inicio = scores365._utc(g.get("startTime"))
        code = ids.get(g.get("competitionId"))
        if g.get("statusGroup") != scores365.STATUS_SCHEDULED or not inicio or not code:
            continue
        t = datetime.strptime(inicio, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if ahora < t <= limite:
            por_comp.setdefault(code, set()).add(t.strftime("%Y-%m-%d"))
    total = 0
    for code, dias in por_comp.items():
        for dia in dias:
            desde = f"{dia}T00:00:00Z"
            hasta = (datetime.strptime(dia, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
            filas, _, _ = odds_api.get_odds_table(code, desde, hasta, paid_fallback=False)
            total += len(filas)
    return total


def commit_if_changed() -> None:
    raiz = str(config.ROOT_DIR)
    cambios = subprocess.run(["git", "status", "--porcelain", "--", *TRACKED_FILES],
                             cwd=raiz, capture_output=True, text=True).stdout.strip()
    if not cambios:
        return
    subprocess.run(["git", "add", "--", *TRACKED_FILES], cwd=raiz, check=True)
    msg = f"Datos: cuotas y fotos en vivo {datetime.now(timezone.utc):%Y-%m-%d}"
    # Solo esos archivos (pathspec): nunca arrastra otros cambios en curso.
    hecho = subprocess.run(["git", "commit", "-m", msg, "--", *TRACKED_FILES],
                           cwd=raiz, capture_output=True, text=True)
    if hecho.returncode == 0:
        push = subprocess.run(["git", "push", "-q"], cwd=raiz, capture_output=True, text=True)
        _log("commit y push" if push.returncode == 0 else f"commit hecho; push falló: {push.stderr.strip()[:200]}")


def main() -> None:
    try:
        games = _todays_games()
        en_juego = [g for g in games if g.get("statusGroup") == scores365.STATUS_LIVE]
        if en_juego:
            filas = scores365.live_rows(list(scores365.COMPETITION_IDS), odds_api.PRICE_KEYS)
            n = capture_live(filas)
            _log(f"en vivo: {len(filas)} partidos, {n} fotos de cuota nuevas")
        n_final = capture_final_stats(games)
        if n_final:
            _log(f"estadísticas finales: {n_final} partidos")
        n_pre = capture_prematch(games)
        if n_pre:
            _log(f"prepartido: {n_pre} partidos con cuota")
        if not en_juego:
            commit_if_changed()
    except Exception:
        _log("ERROR " + traceback.format_exc().replace("\n", " | "))


if __name__ == "__main__":
    main()
