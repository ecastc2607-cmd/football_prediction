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
  4. Tips del día: una vez por día arma el Plan del día (mismas reglas que la
     pestaña Tips, presupuesto TIPS_BUDGET) y lo guarda en tips_log.csv; y
     resuelve los boletos cuyos partidos ya terminaron (pestaña Reportes). Así
     el historial no depende de abrir la app (en Streamlit Cloud no se guarda).
  5. Si ya no hay nada en juego y esos archivos cambiaron: commit + push.

No usa football-data.org ni Goal API (cero cuota): marcador, minuto y
estadísticas salen de 365scores, y la fuerza de los equipos de los CSV locales.

Ventana de partidos (pedido del usuario: que fuera de los partidos no haga
nada): el Programador de tareas no sabe de fixtures, así que la tarea igual se
dispara cada 5 min, pero lo PRIMERO que hace es mirar los horarios guardados
en logs/match_schedule.json (se renuevan con 1 consulta cada
SCHEDULE_MAX_AGE_HOURS). Si no hay un partido que empiece en menos de
PREMATCH_HOURS o haya empezado hace menos de POSTMATCH_MINUTES, termina ahí:
sin consultas, sin cargar el modelo, sin escribir nada (~0.3 s).

Uso:
    python -m src.capture_live_odds              # una corrida (si hay ventana)
    python -m src.capture_live_odds --servicio   # bucle: lo que arranca la tarea de Windows
"""
from __future__ import annotations

import json
import subprocess
import traceback
from datetime import datetime, timedelta, timezone

# Solo lo liviano arriba: fuera de la ventana de partidos la corrida termina
# sin cargar pandas ni el modelo (ver _load_heavy).
from . import config, scores365

LOG_PATH = config.ROOT_DIR / "logs" / "capture_live_odds.log"
SCHEDULE_PATH = config.ROOT_DIR / "logs" / "match_schedule.json"
SCHEDULE_MAX_AGE_HOURS = 6
PREMATCH_HOURS = 3
# 90' + descanso + añadido + margen para guardar las estadísticas finales y
# hacer el commit cuando ya no queda nada en juego.
POSTMATCH_MINUTES = 150
TRACKED_FILES = [
    "data/tracking/tips_log.csv",
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


# --- Ventana de partidos (sin cargar nada pesado) ---

def _parse_utc(texto: str) -> datetime:
    return datetime.strptime(texto, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _kickoffs(ahora: datetime) -> list[str]:
    """Inicios (UTC) de ayer a mañana, del archivo local si tiene menos de
    SCHEDULE_MAX_AGE_HOURS; si no, 1 consulta a 365scores y se guarda."""
    try:
        guardado = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
        if ahora - _parse_utc(guardado["fetched_at"]) < timedelta(hours=SCHEDULE_MAX_AGE_HOURS):
            return guardado["kickoffs"]
    except (OSError, ValueError, KeyError):
        pass
    inicios = sorted({scores365._utc(g.get("startTime")) for g in _todays_games()} - {""})
    SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_PATH.write_text(json.dumps({"fetched_at": ahora.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                         "kickoffs": inicios}), encoding="utf-8")
    return inicios


def in_match_window(ahora: datetime | None = None) -> bool:
    ahora = ahora or datetime.now(timezone.utc)
    for k in _kickoffs(ahora):
        inicio = _parse_utc(k)
        if inicio - timedelta(hours=PREMATCH_HOURS) <= ahora <= inicio + timedelta(minutes=POSTMATCH_MINUTES):
            return True
    return False


TIPS_BUDGET = 50_000
TIPS_BANKROLL = 250_000  # solo para el monto de ¼ de Kelly; el plan usa TIPS_BUDGET


def _load_heavy() -> None:
    """Importa pandas y el modelo recién cuando hay algo que capturar."""
    global pd, live_odds, odds_api, REGULATION_MINUTES, adjusted_live_probabilities
    global live_win_probabilities, record_live_snapshot, anchored_prematch_xg
    global get_cached_stats, log_match_stats, matchday_predictions_df, tips_log
    import pandas as pd
    from . import live_odds, odds_api
    from .live_matches import REGULATION_MINUTES, adjusted_live_probabilities, live_win_probabilities
    from .live_snapshots import record as record_live_snapshot
    from .market_anchor import anchored_prematch_xg
    from .match_stats_log import get_cached_stats, log_match_stats
    from .matchday_predictions import matchday_predictions_df
    from . import tips_log


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


def local_day_predictions(fecha: str) -> "pd.DataFrame":
    """Predicciones de los partidos por jugar en `fecha` (día de Colombia) de
    todas las competiciones, solo con los CSV locales (sin consultar APIs)."""
    from .timezones import to_bogota

    frames = []
    for code in config.COMPETITIONS:
        archivos = sorted(config.PROCESSED_DIR.glob(f"matches_{code}_*.csv"), reverse=True)[:1]
        if not archivos:
            continue
        df = pd.read_csv(archivos[0])
        if "matchday" in df.columns:
            df = df[df["matchday"].notna()]
        dia = df["utc_date"].map(lambda d: to_bogota(d).date().isoformat() if to_bogota(d) else "")
        del_dia = df[(dia == fecha) & df["status"].isin(["SCHEDULED", "TIMED"])]
        for (season, md), partidos in del_dia.groupby(["season_start_year", "matchday"]):
            try:
                preds = matchday_predictions_df(code, int(season), int(md), seasons_back=1)
            except Exception:
                continue
            claves = set(zip(partidos["home_team"], partidos["away_team"]))
            frames.append(preds[[(h, a) in claves for h, a in zip(preds["home_team"], preds["away_team"])]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def capture_daily_tips(fecha: str) -> int:
    """Arma y guarda el Plan del día (una vez por día, antes del primer partido:
    la ventana abre PREMATCH_HOURS antes). Devuelve cuántos boletos guardó."""
    from .daily_plan import build_plans
    from .value_tips import candidate_picks, evaluate

    hist = tips_log.load()
    if not hist.empty and ((hist["fecha"] == fecha) & (hist["origin"] == "capture_task")).any():
        return 0
    preds = local_day_predictions(fecha)
    candidatos = candidate_picks(preds) if not preds.empty else pd.DataFrame()
    if candidatos.empty:
        return 0
    inicio = datetime.strptime(fecha, "%Y-%m-%d").replace(hour=5, tzinfo=timezone.utc)  # 00:00 Colombia (UTC-5)
    desde, hasta = inicio.strftime("%Y-%m-%dT%H:%M:%SZ"), (inicio + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    candidatos["cuota"] = None
    for code in candidatos["competition"].unique():
        eventos, _, _ = odds_api.get_odds_table(code, desde, hasta, paid_fallback=False)
        for idx, r in candidatos[candidatos["competition"] == code].iterrows():
            ev = odds_api.find_event(eventos, r.home_team, r.away_team, r.utc_date, r.home_team_short, r.away_team_short)
            precio = odds_api.row_price(ev, r.odds_key) if ev else None
            if precio:
                candidatos.at[idx, "cuota"] = precio[0]
    evaluado = evaluate(candidatos, TIPS_BANKROLL)
    guardados = 0
    for plan in build_plans(evaluado, TIPS_BUDGET):
        guardados += tips_log.record(fecha, plan.name, tips_log.tickets_from_plan(plan, evaluado, TIPS_BUDGET),
                                     "capture_task")
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


def run_once() -> None:
    """Una captura completa (sin mirar la ventana: eso lo decide quien llama)."""
    _load_heavy()
    _PRED_CACHE.clear()  # en modo servicio el proceso vive días: fuerzas frescas cada vez
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
    hoy_col = (datetime.now(timezone.utc) - timedelta(hours=5)).date().isoformat()
    n_tips = capture_daily_tips(hoy_col)
    if n_tips:
        _log(f"tips del día: {n_tips} boletos guardados")
    n_res = tips_log.resolve()
    if n_res:
        _log(f"tips resueltos: {n_res} picks")
    if not en_juego:
        commit_if_changed()


def main() -> None:
    """Una corrida suelta: solo trabaja dentro de la ventana de partidos."""
    try:
        if in_match_window():
            run_once()
    except Exception:
        _log("ERROR " + traceback.format_exc().replace("\n", " | "))


# --- Modo servicio: el código decide cuándo trabajar ---
# La tarea de Windows solo lo arranca al iniciar sesión; de ahí en adelante
# este bucle duerme hasta la próxima ventana de partidos y, dentro de ella,
# captura cada CAPTURE_EVERY_MINUTES.
CAPTURE_EVERY_MINUTES = 5
# Nunca duerme más que esto de un tirón: así vuelve a mirar el reloj tras una
# suspensión del PC y renueva los horarios por si se agregó o movió un partido.
MAX_SLEEP_MINUTES = 30


def next_window_start(ahora: datetime) -> datetime | None:
    futuros = [_parse_utc(k) - timedelta(hours=PREMATCH_HOURS) for k in _kickoffs(ahora)]
    futuros = [t for t in futuros if t > ahora]
    return min(futuros) if futuros else None


def serve() -> None:
    import time

    _log("servicio iniciado")
    plan_anunciado = None
    while True:
        ahora = datetime.now(timezone.utc)
        try:
            if in_match_window(ahora):
                plan_anunciado = None
                run_once()
                espera = timedelta(minutes=CAPTURE_EVERY_MINUTES)
            else:
                proxima = next_window_start(ahora)
                if proxima != plan_anunciado:  # se anota solo cuando cambia el plan
                    _log(f"en espera: próxima ventana {proxima:%Y-%m-%d %H:%M} UTC" if proxima
                         else "en espera: sin partidos en los horarios guardados")
                    plan_anunciado = proxima
                espera = timedelta(minutes=MAX_SLEEP_MINUTES)
                if proxima:
                    espera = min(proxima - ahora, espera)
        except Exception:
            _log("ERROR " + traceback.format_exc().replace("\n", " | "))
            espera = timedelta(minutes=CAPTURE_EVERY_MINUTES)
        time.sleep(max(espera.total_seconds(), 30))


if __name__ == "__main__":
    import sys

    serve() if "--servicio" in sys.argv else main()
