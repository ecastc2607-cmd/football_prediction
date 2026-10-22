"""
Dashboard de Streamlit: conecta en vivo con el pipeline (src/) para mostrar
predicciones reales de la jornada y la calibración histórica del proyecto.

Correr local:
    streamlit run app.py

Desplegado en Streamlit Community Cloud: configura FOOTBALL_DATA_API_KEY en
"Secrets" (no se sube el .env al repo).
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import streamlit as st

from src import (config, half_backtest, national_teams, odds_api, reports, same_day, scores365, season_store,
                 tips_log)
from src.live_odds import record as record_live_odds
from src.competition_status import resolve_current_season_and_matchday
from src.fetch_football_data import FootballDataClient, fetch_competition
from src.goal_api_client import (
    GOAL_API_KEY,
    GoalApiClient,
    GoalApiRateLimited,
    clear_memo as goal_clear_memo,
    get_match_stats as goal_get_match_stats,
)
from src.highlightly_client import (
    HIGHLIGHTLY_API_KEY,
    HighlightlyRateLimited,
    get_match_stats as highlightly_get_match_stats,
)
from src.live_matches import adjusted_live_probabilities, get_live_matches
from src.live_snapshots import evaluate as evaluate_live_snapshots
from src.live_snapshots import record as record_live_snapshot
from src.market_anchor import anchored_prematch_xg
from src.match_stats_log import get_cached_stats, log_match_stats
from src.match_context import get_standings_map, rivalry_label
from src.match_tendencies import load_team_averages, pick_tendencies
from src.matchday_predictions import matchday_predictions_df
from src.calibration import Calibrator
from src.parlay_backtest import backtest as parlay_backtest
from src.parlay_backtest import summary as parlay_summary
from src.parlay_builder import MIN_LEG_PROB
from src.parlay_builder import TIERS as PARLAY_TIERS
from src.parlay_builder import build_parlays
from src.predict_matchday import LIVE_STATUSES, finished_fixtures
from src.backtest_walkforward import SOURCE as BACKTEST_SOURCE
from src.support_leagues import ensure_support_leagues
from src.prediction_log import MARKETS
from src.daily_plan import MAX_TICKETS, build_plans, ticket_rows
from src.timezones import BOGOTA_TZ, format_bogota, to_bogota
from src.value_tips import (LIVE_MODEL_WEIGHT, MAX_DAILY_EXPOSURE, MIN_MODEL_PROB, candidate_picks,
                             day_summary, evaluate, live_candidate_picks, totals_candidates)

st.set_page_config(page_title="Football Analytics · Jornada", page_icon="⚽", layout="wide")

# --- API key: variable de entorno / .env local, o st.secrets en Streamlit Cloud ---
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY") or config.FOOTBALL_DATA_API_KEY
if not API_KEY and "FOOTBALL_DATA_API_KEY" in st.secrets:
    API_KEY = st.secrets["FOOTBALL_DATA_API_KEY"]

if not API_KEY:
    st.error(
        "Falta FOOTBALL_DATA_API_KEY. En local, ponla en el .env del proyecto. "
        "En Streamlit Cloud, agrégala en Settings → Secrets."
    )
    st.stop()

# Corners, faltas y tarjetas: Goal API es la fuente principal (1.000 peticiones/día
# gratis contra las 100 de Highlightly, y una sola petición por partido en vez de
# dos). Highlightly queda como respaldo automático. Ambas son opcionales: si faltan
# las dos, ese bloque simplemente no se muestra, sin error.
GOAL_KEY = os.getenv("GOAL_API_KEY") or GOAL_API_KEY
if not GOAL_KEY and "GOAL_API_KEY" in st.secrets:
    GOAL_KEY = st.secrets["GOAL_API_KEY"]

HIGHLIGHTLY_KEY = os.getenv("HIGHLIGHTLY_API_KEY") or HIGHLIGHTLY_API_KEY
if not HIGHLIGHTLY_KEY and "HIGHLIGHTLY_API_KEY" in st.secrets:
    HIGHLIGHTLY_KEY = st.secrets["HIGHLIGHTLY_API_KEY"]

STATS_KEY_PRESENT = bool(GOAL_KEY or HIGHLIGHTLY_KEY)

# Cuotas reales para "Tips" (The Odds API, plan gratis). Opcional: sin key, los
# Tips funcionan igual con la cuota escrita a mano.
ODDS_KEY = os.getenv("ODDS_API_KEY") or odds_api.ODDS_API_KEY
if not ODDS_KEY:
    # Opcional de verdad: en local sin secrets.toml, st.secrets lanza al solo
    # preguntar por una clave, y eso no debe tumbar la app entera.
    try:
        ODDS_KEY = st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        ODDS_KEY = ""


@st.cache_data(ttl=1800, show_spinner="Descargando datos de la competición...")
def load_competition(code: str, season: int) -> pd.DataFrame:
    # La Europa League no está en el plan gratis de football-data.org (verificado
    # contra su API y su tabla de cobertura), así que corre sobre Goal API en vez
    # de football-data.org — ver src/europa_league.py. Si esto falla, la excepción
    # sube tal cual y la queda contenida por el propio try/except de la pestaña
    # Jornada (por competición, no afecta a las otras 6).
    if code in config.GOAL_API_COMPETITIONS:
        return _load_goal_api_competition(code, season)
    # Temporada cerrada: sale del archivo versionado, sin consultar la API.
    cerrada = season_store.closed_path(code, season)
    if cerrada.exists():
        return pd.read_csv(cerrada)
    client = FootballDataClient(api_key=API_KEY)
    df = fetch_competition(client, code, season)
    season_store.save_if_closed(df, code, season)
    return df


@st.cache_data(ttl=3 * 3600, show_spinner="Descargando datos de la competición...")
def _load_goal_api_competition(code: str, season: int) -> pd.DataFrame:
    """Europa League y selecciones: 3 h de caché en vez de 30 min, porque Goal
    API da solo 1.000 peticiones/día para TODO (incluidos corners/faltas/tarjetas)
    — el "en vivo" usa su propia consulta aparte, así que esto no atrasa marcadores.

    Si la descarga falla (ej. cuota agotada) y hay una copia en disco, se usa
    esa con un aviso, en vez de dejar toda la jornada sin predicciones."""
    try:
        return config.goal_api_module(code).fetch_competition(season)
    except Exception as e:
        path = config.PROCESSED_DIR / f"matches_{code}_{season}.csv"
        if not path.exists():
            raise
        st.warning(
            f"⏳ {config.COMPETITIONS.get(code, code)}: no se pudo actualizar desde Goal API "
            f"({e}). Se muestran los últimos datos guardados."
        )
        return pd.read_csv(path)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_support_leagues(season: int) -> None:
    """Eredivisie/Primeira Liga para el respaldo doméstico de Europa League.
    Vienen versionadas en el repo; esto solo baja lo que falte o tenga más de
    una semana (como mucho una vez al día por el caché). Si falla, se sigue
    con lo guardado — nunca debe tumbar la jornada."""
    try:
        ensure_support_leagues(season, api_key=API_KEY)
    except Exception:
        pass


def _csv_sin_descanso(code: str, season: int) -> bool:
    path = config.matches_path(code, season)
    return path.exists() and "home_ht_goals" not in pd.read_csv(path, nrows=0).columns


def _prepare_competition(code: str, season: int, matchday: int | None = None) -> None:
    # Asegura que los datos de esta y la temporada anterior estén descargados.
    # Si la jornada que se mira ya terminó por completo en el archivo local, la
    # temporada en curso no se vuelve a pedir: nada de esa jornada cambia.
    if matchday is None or not season_store.jornada_finished_locally(code, season, matchday):
        load_competition(code, season)
    load_competition(code, season - 1)
    # Archivos bajados por una versión anterior del código (sin marcador al
    # descanso) siguen en disco mientras la caché de descarga está vigente tras
    # un deploy — pasó con Premier League (oct-2026): columnas por tiempo en "-"
    # hasta forzar la actualización. Se detecta y se vuelve a descargar solo.
    if any(_csv_sin_descanso(code, s) for s in (season, season - 1)):
        load_competition.clear()
        _load_goal_api_competition.clear()
        load_competition(code, season)
        load_competition(code, season - 1)
    if code in config.CUP_STYLE_COMPETITIONS:
        load_support_leagues(season)


@st.cache_data(ttl=1800, show_spinner="Calculando predicciones...")
def get_predictions(code: str, season: int, matchday: int) -> pd.DataFrame:
    _prepare_competition(code, season, matchday)
    return matchday_predictions_df(code, season, matchday, seasons_back=1)


@st.cache_data(ttl=300, show_spinner="Calculando predicciones...")
def get_predictions_con_jugados(code: str, season: int, matchday: int) -> pd.DataFrame:
    """Igual que get_predictions, pero también incluye los partidos ya en vivo
    o terminados de la jornada (get_predictions se queda solo con lo pendiente,
    a propósito, porque otros consumidores como log_predictions.py no deben
    mezclar partidos ya jugados). Usada por "Detalle por partido" para no
    hacer desaparecer un partido de la tabla apenas arranca o termina — TTL
    más corto porque el estado en vivo cambia rápido."""
    _prepare_competition(code, season, matchday)
    return matchday_predictions_df(code, season, matchday, seasons_back=1, include_played=True)


@st.cache_data(ttl=1800, show_spinner="Detectando jornada actual...")
def auto_status(code: str):
    if code in config.GOAL_API_COMPETITIONS:
        return config.goal_api_module(code).resolve_current_season_and_matchday()
    client = FootballDataClient(api_key=API_KEY)
    return resolve_current_season_and_matchday(client, code)


@st.cache_data(ttl=180, show_spinner="Buscando partidos en vivo...")
def load_live_matches(codes: tuple[str, ...]) -> pd.DataFrame:
    client = FootballDataClient(api_key=API_KEY)
    football_data_codes = [c for c in codes if c not in config.GOAL_API_COMPETITIONS]
    # Reutiliza el caché de 30 min de load_competition en vez de re-descargar
    # la liga si la vista principal ya la trajo hace un momento.
    live_df = get_live_matches(client, football_data_codes, ensure_data=load_competition)

    # Europa League y selecciones corren aparte, sobre Goal API (ver
    # src/europa_league.py y src/national_teams.py). Aisladas a propósito: si
    # una de las dos falla, se ignora y el resto sigue mostrando su "en vivo"
    # con total normalidad — nunca debe tumbar la pestaña.
    for goal_code in (c for c in codes if c in config.GOAL_API_COMPETITIONS):
        try:
            extra_live = config.goal_api_module(goal_code).get_live_matches(ensure_data=load_competition)
        except Exception:
            extra_live = pd.DataFrame()
        if not extra_live.empty:
            live_df = pd.concat([live_df, extra_live], ignore_index=True)

    return live_df


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def upcoming_kickoffs() -> dict:
    """{competición: [inicios UTC ISO de partidos por jugar o en juego]}.

    Para no consultar APIs cada vez que alguien abre/refresca la app:
      1. Primero los CSV de partidos ya descargados (0 consultas): traen el
         calendario completo de la temporada.
      2. Solo las competiciones SIN CSV (típico tras un reinicio de Streamlit
         Cloud): football-data.org con UNA consulta para todas las que falten;
         Europa League/selecciones, Goal API de hoy y los 2 días siguientes.
    Caché de 6 h compartida por todas las sesiones: los horarios casi nunca
    cambian, y el filtro "en juego o por jugar" se aplica afuera, en
    nearest_competition, con la hora del momento."""
    hoy = datetime.now(timezone.utc).date()
    inicios: dict[str, list[str]] = {}
    faltan_fd = []
    for code in config.COMPETITIONS:
        locales = season_store.local_upcoming_kickoffs(code)
        if locales is not None:
            inicios[code] = locales
        elif code not in config.GOAL_API_COMPETITIONS:
            faltan_fd.append(code)
    if faltan_fd:
        try:
            datos = FootballDataClient(api_key=API_KEY)._get("/matches", {
                "dateFrom": hoy.isoformat(), "dateTo": (hoy + pd.Timedelta(days=9)).isoformat(),
                "competitions": ",".join(faltan_fd),
            })
            for m in datos.get("matches", []):
                if m.get("status") in ("SCHEDULED", "TIMED", *LIVE_STATUSES):
                    inicios.setdefault(m["competition"]["code"], []).append(m["utcDate"])
        except Exception:
            pass
    for code in config.GOAL_API_COMPETITIONS:
        if code in inicios:
            continue
        if GOAL_KEY:
            for delta in range(3):
                try:
                    fixtures = GoalApiClient(api_key=GOAL_KEY).fixtures_by_date(
                        (hoy + pd.Timedelta(days=delta)).isoformat(), code)
                except Exception:
                    continue
                inicios.setdefault(code, []).extend(f.get("kickoffUtc") for f in fixtures if f.get("kickoffUtc"))
    return inicios


# Un partido que empezó hace menos de esto se considera EN CURSO (90' + descanso
# + añadido) y cuenta como el más próximo de todos (pedido del usuario).
IN_PLAY_WINDOW = pd.Timedelta(minutes=115)


def nearest_competition() -> str | None:
    """La competición con un partido en curso o, si no hay, la del próximo
    partido por empezar. Entre varias con partidos en curso, la del que empezó
    más recientemente (le queda más por jugar)."""
    ahora = pd.Timestamp.now(tz="UTC")
    mejor, mejor_clave = None, None
    for code, lista in upcoming_kickoffs().items():
        if code not in config.COMPETITIONS:
            continue
        inicios = [t for t in pd.to_datetime(pd.Series(lista), utc=True, errors="coerce")
                   if pd.notna(t) and t > ahora - IN_PLAY_WINDOW]
        if not inicios:
            continue
        en_curso = [t for t in inicios if t <= ahora]
        # En curso: (0, más reciente primero); por empezar: (1, el más cercano).
        clave = (0, -max(en_curso).value) if en_curso else (1, min(inicios).value)
        if mejor_clave is None or clave < mejor_clave:
            mejor, mejor_clave = code, clave
    return mejor


@st.cache_data(ttl=3600, show_spinner=False)
def load_live_evaluation() -> pd.DataFrame:
    return evaluate_live_snapshots()


@st.cache_data(ttl=600, show_spinner=False)
def load_day_fixtures(competition_code: str, date: str) -> list:
    """IDs de Goal API de TODOS los partidos de esa liga y fecha, en una sola
    petición. Se cachea aparte para que consultar 8 partidos en vivo de la misma
    liga cueste 1 + 8 peticiones y no 8 + 8."""
    if not GOAL_KEY:
        return []
    try:
        return GoalApiClient(api_key=GOAL_KEY).fixtures_by_date(date, competition_code)
    except GoalApiRateLimited:
        raise
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def load_match_stats(home: str, away: str, date_iso: str, competition_code: str):
    """Goal API primero; si no devuelve nada, se intenta con Highlightly."""
    if GOAL_KEY:
        fixtures = load_day_fixtures(competition_code, date_iso[:10])
        stats = goal_get_match_stats(
            home, away, date_iso, competition_code, api_key=GOAL_KEY, fixtures=fixtures
        )
        if stats:
            return stats
    if HIGHLIGHTLY_KEY:
        return highlightly_get_match_stats(
            home, away, date_iso, competition_code, api_key=HIGHLIGHTLY_KEY
        )
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_standings(code: str, season: int) -> dict:
    if code in config.GOAL_API_COMPETITIONS:
        try:
            return config.goal_api_module(code).get_standings_map(season)
        except Exception:
            return {}  # la columna "Posiciones" ya sabe mostrar "-" si esto viene vacío
    client = FootballDataClient(api_key=API_KEY)
    return get_standings_map(client, code, season)


@st.cache_data(ttl=3 * 3600, show_spinner="Buscando amistosos internacionales...")
def load_friendlies() -> pd.DataFrame:
    # 3 h: son ~10 peticiones a Goal API por consulta (una por día de la
    # ventana), y esta pestaña se ejecuta en cada carga aunque nadie la mire.
    return national_teams.fetch_friendlies_window()


@st.cache_data(ttl=1800, show_spinner="Calculando predicciones de amistosos...")
def get_friendlies_predictions() -> pd.DataFrame:
    fixtures = load_friendlies()
    if fixtures.empty:
        return pd.DataFrame()
    season = national_teams.current_season()
    load_competition("NT", season)
    load_competition("NT", season - 1)
    strength = national_teams.team_strength_for_competition_safe([season, season - 1])
    return national_teams.predict_friendlies(fixtures, strength)


@st.cache_data(ttl=1800, show_spinner="Reuniendo los partidos del día en todas las competiciones...")
def load_day_predictions(fecha_iso: str) -> pd.DataFrame:
    """Predicciones de TODOS los partidos por jugar en `fecha_iso` (día de
    Colombia), de cualquier competición + amistosos. Mira la jornada actual y
    la siguiente de cada liga: un mismo día puede caer en el cierre de una y
    el arranque de la otra. Una competición que falle se salta sin afectar al resto."""
    frames = []
    for code in config.COMPETITIONS:
        try:
            estado = auto_status(code)
        except Exception:
            continue
        for md in (estado.matchday, estado.matchday + 1):
            try:
                df = get_predictions(code, int(estado.season), int(md))
            except Exception:
                continue
            if not df.empty:
                frames.append(df)
    try:
        amistosos = get_friendlies_predictions()
        if not amistosos.empty:
            frames.append(amistosos.assign(competition="AMI", competition_name="Amistoso internacional"))
    except Exception:
        pass
    if not frames:
        return pd.DataFrame()

    todo = pd.concat(frames, ignore_index=True)
    dia_local = todo["utc_date"].map(lambda d: to_bogota(d).date().isoformat() if to_bogota(d) else "")
    todo = todo[dia_local == fecha_iso]
    todo = todo[todo["status"].isin(["SCHEDULED", "TIMED"])]
    return todo.drop_duplicates(subset=["home_team", "away_team", "utc_date"]).sort_values("utc_date")


@st.cache_data(ttl=600, show_spinner="Consultando cuotas de las casas...")
def load_odds_table(code: str, desde_utc: str, hasta_utc: str, paid_fallback: bool = True) -> tuple:
    """(eventos, créditos_restantes, error). Pasa por el historial versionado
    (odds_api.get_odds_table): una ventana consultada hace menos de
    SNAPSHOT_MAX_AGE_HOURS no gasta créditos, ni siquiera tras un reinicio."""
    return odds_api.get_odds_table(code, desde_utc, hasta_utc, api_key=ODDS_KEY, paid_fallback=paid_fallback)


@st.cache_data(ttl=60, show_spinner=False)
def load_live_odds(codes: tuple) -> list:
    """Cuotas en vivo de 365scores (gratis): 1 consulta + 1 por partido en juego."""
    return scores365.live_rows(list(codes), odds_api.PRICE_KEYS)


def render_daily_plan(evaluado: pd.DataFrame, presupuesto: float, banca: float,
                      fecha: str | None = None) -> None:
    """Plan del día: el presupuesto concentrado en hasta 3 boletos (ver
    src/daily_plan.py), con tres planes comparados por riesgo. Con `fecha`,
    los tres planes quedan guardados en el historial de Reportes."""
    st.subheader(f"🎯 Plan del día — hasta {MAX_TICKETS} apuestas")
    planes = build_plans(evaluado, presupuesto)
    if not planes:
        st.info("Ningún pick tiene valor hoy: no hay plan que armar.")
        return
    if fecha:
        for p in planes:
            tips_log.record(fecha, p.name, tips_log.tickets_from_plan(p, evaluado, presupuesto), "app")
    comparacion = pd.DataFrame([{
        "Plan": p.name,
        "Boletos": len(p.tickets),
        "Apostado": p.metrics["apostado"],
        "Prob. de ganar algo": round(p.metrics["prob_gana"] * 100),
        "Prob. de perderlo todo": round(p.metrics["prob_pierde_todo"] * 100),
        "Ganancia esperada": round(p.metrics["esperado"]),
        "Ganancia máxima": round(p.metrics["max_ganancia"]),
    } for p in planes])
    st.dataframe(
        comparacion, hide_index=True, width="stretch",
        column_config={
            "Apostado": st.column_config.NumberColumn(format="$%d"),
            "Prob. de ganar algo": st.column_config.NumberColumn(format="%d%%"),
            "Prob. de perderlo todo": st.column_config.NumberColumn(format="%d%%"),
            "Ganancia esperada": st.column_config.NumberColumn(
                format="$%d", help="Promedio a la larga si se repitiera muchas veces este mismo plan."),
            "Ganancia máxima": st.column_config.NumberColumn(format="$%d", help="Si aciertan todos los picks."),
        },
    )
    nombres = [p.name for p in planes]
    elegido = st.radio("Plan a jugar", nombres, index=nombres.index("Equilibrado") if "Equilibrado" in nombres else 0,
                       horizontal=True, key="tips_plan")
    plan = planes[nombres.index(elegido)]
    st.caption(plan.description)
    # st.table (no st.dataframe): respeta los saltos de línea dentro de la
    # celda, así cada pick de una combinada va en su propio renglón.
    # El índice es la columna "Boleto" (st.table siempre muestra el índice);
    # los números no se parten en pantallas angostas ("$12, / 000").
    boletos = ticket_rows(plan, evaluado).set_index("Boleto")
    pesos = lambda v: f"${v:,.0f}"  # noqa: E731
    numeros = [c for c in boletos.columns if c != "Picks"]
    st.table(
        boletos.style
        .format({"Cuota": "{:.2f}", "Prob.": "{:.0f}%", "Apostar": pesos,
                 "Recibes si acierta": pesos, "Ganancia neta": pesos})
        .set_properties(subset=["Picks"], **{"white-space": "pre-wrap", "min-width": "16em"})
        .set_properties(subset=numeros, **{"white-space": "nowrap"})
    )
    m = plan.metrics
    st.caption(
        f"Apuestas ${m['apostado']:,.0f} en {len(plan.tickets)} boleto(s), repartidos según el valor de cada uno. "
        f"Ganas algo {m['prob_gana']:.0%} de las veces, lo pierdes todo {m['prob_pierde_todo']:.0%}; "
        f"en promedio {m['esperado']:+,.0f} por día jugado así."
    )
    if banca and presupuesto > 0.2 * banca:
        st.warning(
            f"⚠ Este presupuesto es el {presupuesto / banca:.0%} de tu banca (${banca:,.0f}). Con "
            f"{m['prob_pierde_todo']:.0%} de perderlo todo en un día, dos o tres días malos seguidos "
            "(algo normal) se llevan buena parte de la banca. La regla prudente es no arriesgar más del "
            "20% de la banca por día."
        )


def live_refresh_button(key: str) -> None:
    """Botón para traer marcador, minuto y cuotas en vivo frescos. Limpia la
    caché ANTES de que la sección los cargue, en la misma ejecución del clic:
    sin st.rerun(), que devolvía la vista a la pestaña "Jornada"."""
    if st.button("🔄 Actualizar en vivo", key=key, help="Marcador, minuto y cuotas BWIN del momento"):
        load_live_matches.clear()
        load_live_odds.clear()


def live_data_time(cuotas: list) -> str:
    """Hora (Colombia) de la foto de cuotas en vivo que se está mostrando."""
    horas = [c.get("fetched_at") for c in cuotas or [] if c.get("fetched_at")]
    local = to_bogota(max(horas)) if horas else None
    return local.strftime("%H:%M:%S") if local else ""


@st.cache_data(ttl=60, show_spinner=False)
def load_live_stats(game_id: int, home_id, away_id) -> dict:
    """Estadísticas en vivo de 365scores (gratis): remates, xG real, corners,
    faltas, tarjetas. Reemplaza a Goal API en vivo, que costaba 2 de sus 1.000
    peticiones/día por partido cada 5 min y agotó la cuota (oct-2026)."""
    return scores365.live_stats(game_id, home_id, away_id)


def live_event_state(row, cuotas: list):
    """(row, evento_365) con marcador y minuto REALES de 365scores cuando el
    partido aparece ahí (football-data no da el minuto: se estimaba desde la
    hora de inicio), y las estadísticas en vivo {equipo: {...}} o None."""
    evento = odds_api.find_event(cuotas, row.home_team, row.away_team, row.utc_date) if cuotas else None
    if not evento:
        return row, None, None
    row = row.copy()
    if evento.get("home_score") is not None and evento.get("away_score") is not None:
        row["home_goals"], row["away_goals"] = int(evento["home_score"]), int(evento["away_score"])
    if evento.get("game_minute") not in (None, "") and float(evento["game_minute"]) >= 0:
        row["minuto_estimado"] = int(float(evento["game_minute"]))
    stats = {}
    try:
        stats = load_live_stats(int(str(evento["event_id"]).split(":")[1]),
                                evento.get("home_id"), evento.get("away_id"))
    except Exception:
        stats = {}
    stats_vivo = {row.home_team: stats["home"], row.away_team: stats["away"]} if stats else None
    return row, evento, stats_vivo


def live_adjustment(row, stats_vivo: dict | None = None) -> tuple[dict | None, dict | None, dict | None]:
    """(ajuste, ancla, estadísticas) del 1X2/goles EN VIVO de un partido de
    load_live_matches — compartido por las pestañas "En vivo" y "Tips" para
    que ambas muestren exactamente la misma probabilidad. `stats_vivo` viene
    de live_event_state (365scores): acá ya no se consulta Goal API."""
    if pd.isna(row.get("pre_home_xg")) or pd.isna(row.get("pre_away_xg")):
        return None, None, stats_vivo
    # Base pre-partido anclada al mercado si hay cuotas previas guardadas
    # (sin gastar créditos); si no, la del modelo.
    base_l, base_v, ancla = anchored_prematch_xg(
        float(row.pre_home_xg), float(row.pre_away_xg), row.competition,
        row.home_team, row.away_team, row.utc_date,
    )
    ajuste = adjusted_live_probabilities(
        int(row.home_goals), int(row.away_goals), int(row.minuto_estimado),
        base_l, base_v, (stats_vivo or {}).get(row.home_team), (stats_vivo or {}).get(row.away_team),
    )
    return dict(ajuste, base_home_xg=base_l, base_away_xg=base_v), ancla, stats_vivo


def odds_window(df: pd.DataFrame) -> tuple[str, str]:
    """Ventana UTC de días completos que cubre los partidos de `df` — estable
    entre consultas, para que el historial reconozca la misma ventana."""
    inicio = pd.to_datetime(df["utc_date"], utc=True)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return inicio.min().floor("D").strftime(fmt), (inicio.max().floor("D") + pd.Timedelta(days=1)).strftime(fmt)


def make_odds_lookup(tables: dict, only_colombia: bool = True):
    """odds_lookup para build_parlays: cuota real de un pick a partir de las
    tablas {competición: eventos} ya traídas."""
    def lookup(row, clave):
        eventos = tables.get(row.get("competition"))
        if not eventos:
            return None
        evento = odds_api.find_event(eventos, row["home_team"], row["away_team"], row["utc_date"],
                                     row.get("home_team_short") or "", row.get("away_team_short") or "")
        return odds_api.row_price(evento, clave, only_colombia) if evento else None
    return lookup


def parlays_with_real_odds(df: pd.DataFrame, only_colombia: bool = True,
                           paid_fallback: bool = True) -> tuple[list, list[str]]:
    """Arma las combinadas, y solo para las competiciones que aparecen en ellas
    trae cuotas reales (2 créditos por competición, o 0 si están en el
    historial reciente) y las vuelve a armar ya con esas cuotas.
    Devuelve (combinadas, avisos)."""
    calibracion = load_calibrator()
    base = build_parlays(df, calibracion)
    if not base:
        return base, []
    comp_por_partido = {(r.home_team, r.away_team): r.competition for r in df.itertuples()}
    comps = {comp_por_partido.get((l.home_team, l.away_team)) for p in base for l in p.legs}
    tablas, avisos, restante = {}, [], None
    for code in sorted(c for c in comps if c in odds_api.SPORT_KEYS):
        desde, hasta = odds_window(df[df["competition"] == code])
        eventos, rest, error = load_odds_table(code, desde, hasta, paid_fallback and bool(ODDS_KEY))
        restante = rest if rest is not None else restante
        if error:
            avisos.append(f"{config.COMPETITIONS.get(code, code)}: {error}")
        else:
            tablas[code] = eventos
    if restante is not None:
        avisos.append(f"Créditos de The Odds API restantes este mes: {restante}")
    return build_parlays(df, calibracion, make_odds_lookup(tablas, only_colombia)), avisos


@st.cache_data(ttl=1800, show_spinner=False)
def load_tendency_averages(code: str) -> pd.DataFrame:
    """Promedios de corners/faltas/tarjetas por equipo, desde el log histórico
    (data/tracking/match_stats_log.csv). Se cachea 30 min porque el archivo
    solo crece cuando alguien mira "En vivo" o corre el backfill — no hace
    falta releerlo en cada interacción."""
    return load_team_averages(code)


def _render_stats_table(stats: dict) -> None:
    """Solo el dibujo de la tabla a partir de un dict de stats ya obtenido —
    sin tocar la API ni el log. Compartido por el camino "fresco" (API) y el
    camino "cacheado" (log persistente)."""
    # Solo las columnas que la fuente realmente entregó: Goal API trae
    # posesión y remates además de lo básico, Highlightly no siempre, y un
    # resultado cacheado antiguo puede no tener los campos "bonus".
    etiquetas = {
        "corners": "Corners", "faltas": "Faltas",
        "tarjetas_amarillas": "T. amarillas", "tarjetas_rojas": "T. rojas",
        "posesion": "Posesión %", "remates_a_puerta": "Remates a puerta",
    }
    stat_df = pd.DataFrame(stats).T
    columnas = [c for c in etiquetas if c in stat_df.columns and stat_df[c].notna().any()]
    st.dataframe(
        stat_df[columnas].rename(columns=etiquetas), width="stretch",
        # Fija el nombre del equipo (índice) para no perder la referencia al
        # revisar las demás columnas — se usa igual en "En vivo" y en
        # "Resultados ya jugados", que comparten esta misma función.
        column_config={"_index": st.column_config.Column(pinned=True)},
    )


def render_stats_block(home_team: str, away_team: str, utc_date: str, competition_code: str,
                        season: int = 0, matchday: int = 0) -> None:
    """Corners/faltas/tarjetas REALES de un partido EN VIVO: siempre pide el
    dato fresco al API (el marcador y las estadísticas cambian mientras se
    juega, así que cachear de más mostraría un dato viejo) — el único límite es
    el caché corto de 5 min de load_match_stats. De paso deja el resultado
    guardado en el log, así que retroalimenta el promedio que usan las
    tendencias de jornadas futuras (pick_tendencies) y, cuando el partido
    termine, sirve como caché permanente vía render_finished_stats_block.

    `season`/`matchday` quedan en 0 cuando no se conocen con certeza (esta
    pestaña mezcla varias competiciones a la vez), lo cual no afecta a
    team_averages (agrupa por equipo, no por jornada) ni a get_cached_stats
    (busca por fecha, no por season).
    """
    if not STATS_KEY_PRESENT:
        st.caption("Configura GOAL_API_KEY (o HIGHLIGHTLY_API_KEY) para ver corners/faltas/tarjetas.")
        return

    rate_limited = False
    try:
        stats = load_match_stats(home_team, away_team, utc_date, competition_code)
    except (GoalApiRateLimited, HighlightlyRateLimited) as e:
        stats, rate_limited = None, True
        st.warning(f"⏳ {e}")

    if stats:
        log_match_stats(competition_code, season, matchday, utc_date, home_team, away_team, stats)
        _render_stats_table(stats)
    elif not rate_limited:
        st.caption("Sin estadísticas disponibles todavía para este partido.")


def render_finished_stats_block(home_team: str, away_team: str, utc_date: str, competition_code: str,
                                 season: int, matchday: int) -> None:
    """Igual que render_stats_block, pero para un partido YA TERMINADO: el
    resultado y las estadísticas ya no cambian, así que primero se busca en el
    log persistente (data/tracking/match_stats_log.csv) y solo si NO está ahí
    se llama al API.

    Antes, "Resultados ya jugados" pedía el dato al API cada vez que se abría
    esa sección — un mismo partido consultado dos veces (por ejemplo, verlo una
    vez y volver más tarde) gastaba cuota dos veces, y eso agotó las 1.000
    peticiones/día de Goal API en un solo día de uso normal. Con esto, un
    partido consultado una vez no vuelve a tocar el API nunca más.
    """
    cached = get_cached_stats(competition_code, home_team, away_team, utc_date)
    if cached:
        st.caption("📦 Desde el histórico guardado — no gastó cuota de API.")
        _render_stats_table(cached)
        return
    render_stats_block(home_team, away_team, utc_date, competition_code, season, matchday)


@st.cache_data(ttl=1800, show_spinner="Armando combinadas...")
def load_parlays(code: str, season: int, matchday: int) -> list:
    df = get_predictions(code, season, matchday)
    return build_parlays(df, load_calibrator()) if not df.empty else []


@st.cache_data(ttl=1800, show_spinner=False)
def load_calibrator() -> Calibrator:
    """Calibración por mercado ajustada con TODAS las predicciones ya resueltas
    — se actualiza sola a medida que el log crece."""
    return Calibrator.from_log(load_calibration_log(), half_backtest=half_backtest.load())


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_half_market_calibration() -> pd.DataFrame:
    """Acierto real vs prometido de los mercados de un solo tiempo, según el
    backtest guardado (data/tracking/half_markets_backtest.csv)."""
    bt = half_backtest.load()
    if bt.empty:
        return pd.DataFrame()
    tabla = half_backtest.leg_calibration(bt)
    return tabla.assign(Acierto=tabla["Acierto"] * 100, Prometido=tabla["Prometido"] * 100)


@st.cache_data(ttl=6 * 3600, show_spinner="Evaluando la estrategia en jornadas pasadas...")
def load_parlay_backtest() -> pd.DataFrame:
    return parlay_summary(parlay_backtest(load_calibration_log()))


_TIER_META = {
    "Alta probabilidad": ("🟢", "Gana aprox. 2 de cada 5 veces o más."),
    "Equilibrada": ("🟡", "Gana aprox. 1 de cada 3-4 veces."),
    "Cuota alta": ("🔴", "Gana aprox. 1 de cada 4-8 veces: paga más, falla la mayoría."),
}


def render_parlays(parlays: list, bankroll: float | None = None, avisos: list[str] | None = None) -> None:
    """Combinadas como cupones: probabilidad calibrada, cuota justa, lo que
    paga la casa (real si se trajo, estimada si no), valor esperado y monto
    sugerido. Debajo, cómo le habría ido a la estrategia en jornadas ya
    resueltas — para que cada número se pueda contrastar."""
    st.caption(
        f"Probabilidades **calibradas** con los resultados reales del proyecto (no las del modelo "
        f"en bruto): solo entran picks con ≥{MIN_LEG_PROB:.0%} calibrado, así que un pick que en el "
        "histórico acertó poco queda fuera aunque el modelo le dé más. Un pick por partido como mucho "
        "(dos del mismo partido estarían correlacionados). En cada nivel va primero la de mayor "
        "**valor esperado** (probabilidad × lo que paga la casa). 'Paga' usa la mejor cuota real "
        "cuando está disponible (✔) y si no, la justa menos ~5% de margen por pick. Doble "
        "oportunidad se deriva del 1X2 de la misma casa; Ambos anotan no tiene cuota real. "
        "Estadística sobre datos históricos, no garantía."
    )
    for aviso in avisos or []:
        st.caption(f"ℹ️ {aviso}")
    if not parlays:
        st.info("Hoy no hay combinadas recomendables: no hay al menos 2 picks que lleguen al mínimo "
                "calibrado en partidos distintos.")
    else:
        if not any(p.risk == "Alta probabilidad" for p in parlays):
            st.warning("No hay combinadas de **alta probabilidad** con estos partidos: lo que queda "
                       "abajo falla la mayoría de las veces.")
        for nombre, _, _ in PARLAY_TIERS:
            del_nivel = [p for p in parlays if p.risk == nombre]
            if not del_nivel:
                continue
            emoji, explicacion = _TIER_META[nombre]
            st.markdown(f"**{emoji} {nombre}** — {explicacion}")
            for i, p in enumerate(del_nivel):
                with st.container(border=True):
                    etiqueta = "⭐ Recomendada · " if i == 0 else ""
                    paga = f"paga `{p.estimated_odds}x`" + (" ✔ real" if p.all_real_odds else " (estimada)")
                    st.markdown(
                        f"{etiqueta}prob. **{p.combined_probability:.0%}** · cuota justa `{p.combined_odds}x` · {paga}"
                    )
                    if p.all_real_odds and p.expected_value > 0:
                        monto = (f" → apostar **${p.stake(bankroll):,.0f}** (1% de la banca)" if bankroll
                                 else " → monto sugerido: 1% de tu banca")
                        st.success(f"Con valor: +{p.expected_value:.1%} esperado por peso a cuota real{monto}")
                    elif p.all_real_odds:
                        st.caption(f"Sin valor a cuota real ({p.expected_value:+.1%} esperado): la casa "
                                   "paga menos de lo que vale — mejor no apostarla.")
                    else:
                        st.caption("Valor sin confirmar: falta la cuota real de algún pick.")
                    st.dataframe(
                        pd.DataFrame([
                            {"Partido": l.match, "Mercado": l.market, "Pick": l.pick,
                             "Prob.": f"{l.probability:.0%}",
                             "Cuota": f"{l.book_odds:.2f}" if l.book_odds else f"~{l.payout_odds:.2f}",
                             "Casa": l.bookmaker or "estimada"}
                            for l in p.legs
                        ]),
                        width="stretch", hide_index=True,
                    )

    with st.expander("📈 ¿Cómo le habría ido a esta estrategia en jornadas ya jugadas?"):
        try:
            resumen = load_parlay_backtest()
        except Exception as e:
            resumen = pd.DataFrame()
            st.caption(f"No se pudo calcular ({e}).")
        if resumen.empty:
            st.caption("Todavía no hay jornadas resueltas suficientes.")
        else:
            st.dataframe(
                resumen, width="stretch", hide_index=True,
                column_config={
                    "Acierto real": st.column_config.NumberColumn(format="percent"),
                    "Prob. prometida": st.column_config.NumberColumn(format="percent"),
                    "Retorno por $1": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            st.caption(
                "Cada jornada resuelta se armó con una calibración ajustada SIN esa jornada (no ve su "
                "propio resultado). 'Retorno por $1' usa la cuota estimada en casa: por debajo de 1.00 "
                "se habría perdido plata. Muestra todavía chica: tómalo como tendencia, no como promesa."
            )


@st.cache_data(ttl=1800)
def load_calibration_log() -> pd.DataFrame:
    path = config.ROOT_DIR / "data" / "tracking" / "predictions_log.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _to_bool_series(s: pd.Series) -> pd.Series:
    """'True'/'False' (como quedan tras pasar por CSV) -> booleano real, NaN
    para lo que no se pudo resolver en ese mercado (no cuenta ni como acierto
    ni como fallo — ver STAT_MARKETS en prediction_log.py)."""
    texto = s.astype(str).str.strip().str.lower()
    return texto.map({"true": True, "false": False})


# --- Sidebar ---
st.sidebar.title("⚽ Football Analytics")
st.sidebar.caption("Modelo de Poisson sobre datos reales de football-data.org")

# Por defecto, la competición con el partido más próximo que aún no empezó. Se
# decide UNA vez por sesión (al abrir/refrescar el navegador): recalcularlo en
# cada interacción le cambiaría la liga al usuario apenas arranque un partido.
if "_comp_por_defecto" not in st.session_state:
    try:
        st.session_state["_comp_por_defecto"] = nearest_competition() or "PL"
    except Exception:
        st.session_state["_comp_por_defecto"] = "PL"
_opciones_comp = list(config.COMPETITIONS.keys())
comp_code = st.sidebar.selectbox(
    "Competición",
    options=_opciones_comp,
    index=_opciones_comp.index(st.session_state["_comp_por_defecto"]),
    format_func=lambda c: config.COMPETITIONS[c],
)

# Temporada/jornada se auto-detectan al cambiar de competición (football-data.org
# ya calcula cuál está en curso); el usuario puede seguir ajustándolas a mano.
if st.session_state.get("_last_comp") != comp_code:
    try:
        status = auto_status(comp_code)
        st.session_state["season_input"] = status.season
        st.session_state["matchday_input"] = status.matchday
        st.session_state["_status_reason"] = status.reason
    except Exception as e:
        # Nunca debe tumbar TODA la app (las dos pestañas, las 7 competiciones):
        # si la auto-detección falla (ej. un timeout puntual de la API), se cae a
        # un valor por defecto razonable y se avisa, en vez de dejar todo en blanco.
        st.session_state.setdefault("season_input", datetime.now(timezone.utc).year)
        st.session_state.setdefault("matchday_input", 1)
        st.session_state["_status_reason"] = (
            f"No se pudo detectar la jornada automáticamente ({e}). Ajusta temporada/jornada a mano."
        )
    st.session_state["_last_comp"] = comp_code

season = st.sidebar.number_input("Temporada (año de inicio)", step=1, key="season_input")
matchday = st.sidebar.number_input("Jornada", min_value=1, step=1, key="matchday_input")
st.sidebar.caption(f"🎯 {st.session_state.get('_status_reason', '')}")

if st.sidebar.button("🔄 Forzar actualización de datos"):
    load_competition.clear()
    _load_goal_api_competition.clear()
    goal_clear_memo()
    get_predictions.clear()
    get_predictions_con_jugados.clear()
    auto_status.clear()
    load_live_matches.clear()
    load_tendency_averages.clear()
    load_parlays.clear()
    load_calibrator.clear()
    load_friendlies.clear()
    get_friendlies_predictions.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    "Los equipos marcados de baja confianza tienen muy pocos partidos propios en "
    "los datos cargados (casi siempre recién ascendidos): el modelo se apoya "
    "sobre todo en el promedio de liga para ellos."
)

tab_jornada, tab_vivo, tab_amistosos, tab_tips, tab_reportes = st.tabs(
    ["📅 Jornada", "🔴 En vivo", "🌍 Amistosos", "💡 Tips", "📊 Reportes"]
)

# ============================== PESTAÑA: JORNADA ==============================
with tab_jornada:
    st.title(f"{config.COMPETITIONS[comp_code]} · Jornada {matchday}")

    try:
        df = get_predictions(comp_code, int(season), int(matchday))
    except Exception as e:
        st.error(f"No se pudo calcular la jornada: {e}")
        st.stop()

    if df.empty:
        st.info(
            "Esta jornada ya terminó por completo (sin partidos por jugar), así que no hay "
            "predicciones que mostrar aquí — pero sí puedes ver sus resultados y estadísticas "
            "más abajo, en 'Resultados ya jugados'."
        )
    else:
        # --- Métricas resumen ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Partidos", len(df))
        c2.metric("Over 2.5 promedio", f"{df['over_2_5'].mean():.0f}%")
        c3.metric("BTTS promedio", f"{df['btts'].mean():.0f}%")
        c4.metric("Con baja confianza", int(df["low_confidence"].sum()))

        st.divider()

        # --- Gráfico 1X2 ---
        melt = df.melt(
            id_vars=["home_team", "away_team", "home_team_short", "away_team_short"],
            value_vars=["home_win", "draw", "away_win"],
            var_name="resultado", value_name="probabilidad",
        )
        # Nombres cortos ("Sunderland vs Leeds Utd") para que el eje no trunque ni
        # encime etiquetas; el nombre oficial completo queda en el tooltip.
        melt["partido"] = melt["home_team_short"] + " vs " + melt["away_team_short"]
        melt["partido_completo"] = melt["home_team"] + " vs " + melt["away_team"]
        orden_map = {"home_win": (0, "Local"), "draw": (1, "Empate"), "away_win": (2, "Visitante")}
        melt["orden"] = melt["resultado"].map(lambda k: orden_map[k][0])
        melt["resultado"] = melt["resultado"].map(lambda k: orden_map[k][1])

        chart = alt.Chart(melt).mark_bar(height=18).encode(
            x=alt.X("probabilidad:Q", stack="zero", axis=alt.Axis(format=".0f", title="Probabilidad (%)")),
            y=alt.Y("partido:N", sort=None, title=None,
                    axis=alt.Axis(labelLimit=220, labelFontSize=12, labelPadding=8)),
            color=alt.Color(
                "resultado:N",
                scale=alt.Scale(domain=["Local", "Empate", "Visitante"], range=["#2a78d6", "#eb6834", "#1baf7a"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            order=alt.Order("orden:Q"),
            tooltip=["partido_completo", "resultado", alt.Tooltip("probabilidad:Q", format=".1f")],
        ).properties(height=40 * len(df) + 40)

        st.altair_chart(chart, use_container_width=True)

        # --- Tabla detallada ---
        st.subheader("Detalle por partido")
        standings = load_standings(comp_code, int(season))
        # A diferencia del gráfico de arriba (solo lo pendiente), esta tabla
        # también trae los partidos ya en vivo o terminados de la jornada —
        # matchday_predictions_df ya los deja ordenados al final.
        show = get_predictions_con_jugados(comp_code, int(season), int(matchday)).copy()
        show["Partido"] = show["home_team"] + " vs " + show["away_team"]
        show["Fecha"] = show["utc_date"].map(lambda d: format_bogota(d))
        show["1X2"] = show.apply(lambda r: f"{r.home_win:.0f}% / {r.draw:.0f}% / {r.away_win:.0f}%", axis=1)
        show["xG"] = show.apply(lambda r: f"{r.home_xg} - {r.away_xg}", axis=1)

        def _por_tiempo(r, prefijo):
            local, visita = r.get(f"{prefijo}_home_xg"), r.get(f"{prefijo}_away_xg")
            if pd.isna(local) or pd.isna(visita):
                return "-"
            # Entre paréntesis, la probabilidad de que ESE equipo anote al menos
            # uno en la mitad; "algún gol" es la del partido (cualquiera de los dos).
            anota_l, anota_v = 1 - math.exp(-local), 1 - math.exp(-visita)
            return (f"L {local:.2f} ({anota_l:.0%}) – V {visita:.2f} ({anota_v:.0%}) · "
                    f"algún gol {r.get(f'{prefijo}_p_goal'):.0f}%")

        show["1er tiempo"] = show.apply(lambda r: _por_tiempo(r, "h1"), axis=1) if "h1_home_xg" in show else "-"
        show["2do tiempo"] = show.apply(lambda r: _por_tiempo(r, "h2"), axis=1) if "h2_home_xg" in show else "-"
        show["⚠"] = show["low_confidence"].map({True: "Baja confianza", False: ""})
        show["Posiciones"] = show.apply(
            lambda r: f"{standings.get(r.home_team, '-')}° vs {standings.get(r.away_team, '-')}°"
            if standings else "-",
            axis=1,
        )

        def _contexto(r):
            rivalidad = rivalry_label(r.home_team_short, r.away_team_short) or ""
            if r.status in LIVE_STATUSES:
                estado = "🔴 En vivo"
            elif r.status == "FINISHED":
                estado = "✅ Finalizado"
            else:
                estado = ""
            if estado and rivalidad:
                return f"{estado} · {rivalidad}"
            return estado or rivalidad

        show["Contexto"] = show.apply(_contexto, axis=1)

        # --- Tendencias de corners/faltas/tarjetas (promedio histórico, no del
        # modelo de goles) para partidos que TODAVÍA no se juegan. ---
        tendency_averages = load_tendency_averages(comp_code)
        columnas_tabla = ["Partido", "Fecha", "Posiciones", "xG", "1X2",
                           "over_2_5", "btts", "1er tiempo", "2do tiempo", "top_score"]
        st.caption(
            "⏱️ **1er / 2do tiempo**: goles esperados de cada equipo (L = local, V = visitante) solo en "
            "esa mitad; entre paréntesis, la probabilidad de que ESE equipo marque al menos un gol en "
            "la mitad; 'algún gol' = que marque cualquiera de los dos. Ej.: L 1.27 (72%) – V 0.23 (21%) "
            "· algún gol 78% → el local marca en esa mitad 72% de las veces, el visitante 21%, y hay "
            "algún gol 78%. '-' = sin historial propio con marcador al descanso. Para apuestas de un "
            "solo tiempo: las casas cobran más margen en esos mercados, así que una cuota más alta no "
            "es por sí sola una mejor apuesta."
        )
        if tendency_averages.empty:
            st.caption(
                "📐 Corners/faltas/tarjetas: todavía no hay historial guardado para esta "
                "competición (se acumula automáticamente al ver la pestaña 'En vivo', o de una "
                "vez con `python scripts/backfill_match_stats.py`)."
            )
        else:
            picks_por_partido = [
                pick_tendencies(tendency_averages, r.home_team, r.away_team) for r in show.itertuples()
            ]
            for i, label in enumerate(("Corners", "Faltas", "Tarjetas")):
                linea = picks_por_partido[0][i].line
                show[f"{label} (+{linea})"] = [p[i].describe() for p in picks_por_partido]
            columnas_tabla += [c for c in show.columns if c.startswith(("Corners (", "Faltas (", "Tarjetas ("))]
            st.caption(
                "📐 Corners/faltas/tarjetas: tendencia a partir del promedio histórico real de cada "
                "equipo (no es el modelo de goles). ⚠ = alguno de los dos equipos tiene menos de "
                "5 partidos registrados todavía — el número es orientativo, no confiable aún."
            )

        columnas_tabla += ["Contexto", "⚠"]  # al final: son avisos/contexto, no el pronóstico en sí

        st.dataframe(
            show[columnas_tabla].rename(
                columns={"over_2_5": "Over 2.5 %", "btts": "BTTS %", "top_score": "Marcador top"}
            ),
            width="stretch", hide_index=True,
            # Fija "Partido" para no perder la referencia de qué equipos son al
            # revisar las columnas de pronóstico más a la derecha.
            column_config={"Partido": st.column_config.Column(pinned=True)},
        )

    # --- Resultados ya jugados en esta jornada ---
    st.divider()
    st.subheader("✅ Resultados ya jugados en esta jornada")
    played = finished_fixtures(comp_code, int(season), int(matchday))
    if played.empty:
        st.caption("Ningún partido de esta jornada ha terminado todavía.")
    else:
        for _, row in played.iterrows():
            marcador = (
                f"{int(row['home_goals'])}-{int(row['away_goals'])}"
                if pd.notna(row["home_goals"]) else "?-?"
            )
            label = f"{row['home_team']} {marcador} {row['away_team']} · {format_bogota(row['utc_date'])}"
            with st.expander(label):
                render_finished_stats_block(
                    row["home_team"], row["away_team"], row["utc_date"], comp_code,
                    season=int(season), matchday=int(matchday),
                )

    # --- Combinadas sugeridas ---
    st.divider()
    st.subheader("🎯 Combinadas sugeridas")
    # Cuotas reales automáticas desde 365scores (BWIN, gratis). The Odds API
    # (gasta ~2 de los 500 créditos/mes) solo a pedido, si 365scores no trae nada.
    clave_cuotas = f"cuotas_reales_{comp_code}_{int(season)}_{int(matchday)}"
    df_combinadas = get_predictions(comp_code, int(season), int(matchday))
    if comp_code in odds_api.SPORT_KEYS and not df_combinadas.empty:
        combinadas, avisos_cuotas = parlays_with_real_odds(
            df_combinadas, paid_fallback=bool(st.session_state.get(clave_cuotas)))
        sin_365 = any("365scores no trajo" in a for a in avisos_cuotas)
        if sin_365 and ODDS_KEY and not st.session_state.get(clave_cuotas):
            if st.button("💱 Traer cuotas de The Odds API (~2 créditos)", key=f"btn_{clave_cuotas}"):
                st.session_state[clave_cuotas] = True
                combinadas, avisos_cuotas = parlays_with_real_odds(df_combinadas, paid_fallback=True)
        render_parlays(combinadas, avisos=avisos_cuotas)
    else:
        render_parlays(load_parlays(comp_code, int(season), int(matchday)))

    # --- Calibración histórica ---
    st.divider()
    st.subheader("📊 Calibración del proyecto")
    log = load_calibration_log()
    resolved = log[log["status"] == "resolved"] if not log.empty else pd.DataFrame()

    # Backtest = partidos ya jugados predichos después, con datos anteriores a
    # cada partido (ver src/backtest_walkforward.py). Útil como referencia, pero
    # nunca mezclado por defecto con lo que de verdad se predijo antes.
    origenes = {
        "Antes del partido": lambda d: d[d["source"] != BACKTEST_SOURCE],
        "Backtest (jornadas ya jugadas)": lambda d: d[d["source"] == BACKTEST_SOURCE],
        "Todo": lambda d: d,
    }
    origen = st.radio(
        "Origen de las predicciones", list(origenes), horizontal=True, key="calib_origen",
        help="'Antes del partido': logueadas antes de jugarse (la medida honesta). 'Backtest': "
             "Nations League y Europa League, integradas con la temporada en marcha — cada "
             "partido se predijo después, pero solo con resultados anteriores a su inicio.",
    )
    if not resolved.empty:
        resolved = origenes[origen](resolved)

    if resolved.empty:
        st.caption("Todavía no hay predicciones resueltas para este origen.")
    else:
        resolved = resolved.copy()
        resolved["hit"] = _to_bool_series(resolved["hit"])
        overall = resolved["hit"].mean()
        st.metric(f"Acierto 1X2 · {origen.lower()} (todas las competiciones)", f"{overall:.0%}",
                   f"{int(resolved['hit'].sum())}/{len(resolved)}")

        by_comp = resolved.groupby("competition")["hit"].agg(aciertos="sum", total="count").reset_index()
        by_comp["tasa"] = by_comp["aciertos"] / by_comp["total"] * 100
        by_comp["competencia"] = by_comp["competition"].map(config.COMPETITIONS).fillna(by_comp["competition"])

        bar = alt.Chart(by_comp).mark_bar(color="#2a78d6").encode(
            x=alt.X("competencia:N", title=None, sort="-y"),
            y=alt.Y("tasa:Q", title="% de acierto", scale=alt.Scale(domain=[0, 100])),
            tooltip=["competencia", "aciertos", "total", alt.Tooltip("tasa:Q", format=".0f")],
        ).properties(height=280)
        st.altair_chart(bar, use_container_width=True)

        # --- Tendencia por liga y por mercado ---
        # El % de arriba mezcla en un solo número resultado/goles/córners/etc.
        # Esta tabla abre esa caja: por liga, qué tan bien acierta CADA mercado
        # por separado — para saber si un 75% de acierto viene sobre todo del
        # ganador del partido, de los goles, o de otra cosa.
        st.markdown("**Tendencia por liga y mercado**")
        st.caption(
            "% de acierto de cada mercado por separado. Corners/Faltas/Tarjetas solo tienen "
            "datos desde que se empezó a registrar la tendencia junto con la predicción — "
            "'—' significa que todavía no hay muestra para ese cruce, no que haya fallado."
        )
        filas = []
        for comp, grupo in resolved.groupby("competition"):
            fila = {"Liga": config.COMPETITIONS.get(comp, comp)}
            for label, columna in MARKETS:
                if columna == "hit":
                    serie = grupo["hit"]  # ya convertida arriba
                elif columna in grupo.columns:
                    serie = _to_bool_series(grupo[columna])
                else:
                    serie = pd.Series(dtype=object)
                validos = serie.dropna()
                fila[label] = f"{validos.mean():.0%} ({len(validos)})" if len(validos) > 0 else "—"
            filas.append(fila)

        tabla_mercados = pd.DataFrame(filas)
        st.dataframe(
            tabla_mercados, width="stretch", hide_index=True,
            column_config={"Liga": st.column_config.Column(pinned=True)},
        )

    # Mercados de un solo tiempo: fuera del selector de origen porque su única
    # fuente por ahora es el backtest (el registro antes del partido recién
    # empieza a guardarlos).
    calib_mitades = load_half_market_calibration()
    if not calib_mitades.empty:
        with st.expander("⏱️ Mercados de un solo tiempo — ¿qué tan bien acierta el modelo?"):
            st.dataframe(
                calib_mitades, width="stretch", hide_index=True,
                column_config={
                    "Acierto": st.column_config.NumberColumn("Acierto real", format="%.0f%%"),
                    "Prometido": st.column_config.NumberColumn("Prob. prometida", format="%.0f%%"),
                },
            )
            st.caption(
                "Backtest de los partidos ya jugados de la temporada (cada uno predicho solo con "
                "partidos anteriores), con el lado más probable de cada mercado. 'Gol' en cada tiempo "
                "sale bien calibrado; '+/-1.5' por tiempo promete más de lo que acierta. Probado en "
                "combinadas: agregarlos NO mejoró el retorno (el margen de la casa es mayor en estos "
                "mercados), así que las combinadas no los usan por ahora."
            )

# ============================== PESTAÑA: EN VIVO ==============================
with tab_vivo:
    st.title("🔴 Partidos en vivo")
    st.caption(
        "Minuto: real en Europa League y selecciones; estimado desde la hora de inicio en las "
        "ligas (puede quedarse pegado si football-data.org tarda en marcar el final). 1X2 en vivo "
        "**ajustado por lo que pasa en la cancha**: el ritmo de gol de cada equipo se actualiza con "
        "sus remates (a puerta ~0,20 de xG, desviados ~0,05), el que va perdiendo ataca un poco más "
        "y se cuenta el tiempo añadido. Debajo, entre paréntesis, lo que daría solo el modelo de "
        "antes del partido. No es una cuota de casa de apuestas."
    )

    live_refresh_button("refresh_vivo")

    live_df = load_live_matches(tuple(config.COMPETITIONS.keys()))

    if live_df.empty:
        st.info(f"No hay partidos en juego ahora mismo en las {len(config.COMPETITIONS)} competiciones.")
    else:
        try:
            cuotas_vivo = load_live_odds(tuple(sorted(set(live_df["competition"]))))
        except Exception:
            cuotas_vivo = []
        if live_data_time(cuotas_vivo):
            st.caption(f"Cuotas BWIN de las {live_data_time(cuotas_vivo)} · se renuevan solas cada minuto "
                       "al interactuar con la app, o al pulsar 'Actualizar en vivo'.")
        for _, row in live_df.iterrows():
            live_1x2 = (
                f"{row.live_home_win:.0f}% / {row.live_draw:.0f}% / {row.live_away_win:.0f}%"
                if pd.notna(row.live_home_win) else "sin datos suficientes"
            )
            row, evento_vivo, stats_vivo = live_event_state(row, cuotas_vivo)
            ajuste, ancla, stats_vivo = live_adjustment(row, stats_vivo)
            if ajuste:
                base_l, base_v = ajuste["base_home_xg"], ajuste["base_away_xg"]
                stats_l = (stats_vivo or {}).get(row.home_team)
                stats_v = (stats_vivo or {}).get(row.away_team)
                record_live_snapshot({
                    "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "competition": row.competition, "home_team": row.home_team, "away_team": row.away_team,
                    "utc_date": row.utc_date, "minute": int(row.minuto_estimado),
                    "home_goals": int(row.home_goals), "away_goals": int(row.away_goals),
                    "pre_home_xg": float(row.pre_home_xg), "pre_away_xg": float(row.pre_away_xg),
                    "base_home_xg": base_l, "base_away_xg": base_v, "market_anchor": ancla is not None,
                    "home_shots": (stats_l or {}).get("remates_totales"),
                    "home_shots_on": (stats_l or {}).get("remates_a_puerta"),
                    "away_shots": (stats_v or {}).get("remates_totales"),
                    "away_shots_on": (stats_v or {}).get("remates_a_puerta"),
                    "pre_home": row.live_home_win / 100 if pd.notna(row.live_home_win) else None,
                    "pre_draw": row.live_draw / 100 if pd.notna(row.live_draw) else None,
                    "pre_away": row.live_away_win / 100 if pd.notna(row.live_away_win) else None,
                    "adj_home": ajuste["home_win"], "adj_draw": ajuste["draw"], "adj_away": ajuste["away_win"],
                })

            # Cuota en vivo de BWIN (365scores); cada cambio queda guardado con
            # la probabilidad del modelo en ese minuto (live_odds_snapshots.csv).
            if evento_vivo:
                if ajuste:
                    evento_vivo = dict(
                        evento_vivo, model_home=round(ajuste["home_win"], 4), model_draw=round(ajuste["draw"], 4),
                        model_away=round(ajuste["away_win"], 4),
                        model_home_xg_rem=round(ajuste["home_xg_remaining"], 3),
                        model_away_xg_rem=round(ajuste["away_xg_remaining"], 3),
                    )
                record_live_odds([evento_vivo])

            def _cuota(clave):
                precio = odds_api.row_price(evento_vivo, clave) if evento_vivo else None
                return precio[0] if precio else None

            c_liga, c_local, c_marcador, c_visitante, c_1x2 = st.columns([1.3, 2, 1, 2, 1.6])
            c_liga.caption(f"{row.competition_name} · {format_bogota(row.utc_date)} · min. {row.minuto_estimado}'")
            c_local.write(row.home_team)
            c_marcador.markdown(f"**{row.home_goals} - {row.away_goals}**")
            c_visitante.write(row.away_team)
            if ajuste:
                c_1x2.markdown(f"**1X2: {ajuste['home_win']:.0%} / {ajuste['draw']:.0%} / {ajuste['away_win']:.0%}**")
                c_1x2.caption(f"(solo pre-partido: {live_1x2})")
                # Desglose como tabla: una fila por dato; columnas local /
                # empate-o-total / visitante. La cuota BWIN va junto a la
                # probabilidad del modelo y su valor (prob × cuota − 1).
                fmt_cuota = lambda c: f"{c:.2f}" if c else "-"  # noqa: E731
                fmt_valor = lambda p, c: f"{p * c - 1:+.0%}" if c else "-"  # noqa: E731
                c_1, c_x, c_2 = _cuota("home"), _cuota("draw"), _cuota("away")
                filas_detalle = [
                    ("Prob. en vivo (modelo)", f"{ajuste['home_win']:.0%}", f"{ajuste['draw']:.0%}",
                     f"{ajuste['away_win']:.0%}"),
                    ("Cuota BWIN en vivo", fmt_cuota(c_1), fmt_cuota(c_x), fmt_cuota(c_2)),
                    ("Valor (prob × cuota − 1)", fmt_valor(ajuste["home_win"], c_1),
                     fmt_valor(ajuste["draw"], c_x), fmt_valor(ajuste["away_win"], c_2)),
                ]
                c_over, c_under = _cuota("over"), _cuota("under")
                if c_over or c_under:
                    filas_detalle.append((
                        "Goles: prob. · cuota · valor", "",
                        f"+2.5: {ajuste['over_2_5']:.0%} · {fmt_cuota(c_over)} · {fmt_valor(ajuste['over_2_5'], c_over)}"
                        f"  |  −2.5: {ajuste['under_2_5']:.0%} · {fmt_cuota(c_under)} · "
                        f"{fmt_valor(ajuste['under_2_5'], c_under)}", ""))
                if ajuste["home_xg_observed"] is not None and ajuste["away_xg_observed"] is not None:
                    s_l, s_v = stats_vivo[row.home_team], stats_vivo[row.away_team]
                    n = lambda v: f"{float(v):.0f}"  # noqa: E731 — 10.0 -> "10"
                    filas_detalle += [
                        ("Remates (a puerta)",
                         f"{n(s_l.get('remates_totales'))} ({n(s_l.get('remates_a_puerta'))})", "",
                         f"{n(s_v.get('remates_totales'))} ({n(s_v.get('remates_a_puerta'))})"),
                        ("xG aprox. por remates", f"{ajuste['home_xg_observed']:.2f}", "",
                         f"{ajuste['away_xg_observed']:.2f}"),
                    ]
                if ancla:
                    filas_detalle.append(("xG base: cuotas previas", f"{ancla['market_home_xg']:.2f}", "",
                                          f"{ancla['market_away_xg']:.2f}"))
                filas_detalle += [
                    ("xG pre-partido: modelo", f"{row.pre_home_xg:.2f}", "", f"{row.pre_away_xg:.2f}"),
                    (f"Goles esperados en lo que falta (~{ajuste['remaining_minutes']:.0f}')",
                     f"{ajuste['home_xg_remaining']:.2f}", "", f"{ajuste['away_xg_remaining']:.2f}"),
                    ("No pierde (doble oportunidad)",
                     f"{ajuste['home_win'] + ajuste['draw']:.0%} · {fmt_cuota(_cuota('dc_1x'))}", "",
                     f"{ajuste['away_win'] + ajuste['draw']:.0%} · {fmt_cuota(_cuota('dc_x2'))}"),
                ]
                st.dataframe(
                    pd.DataFrame(filas_detalle, columns=["Dato", row.home_team, "Empate / total", row.away_team]),
                    hide_index=True, width="stretch",
                )
                notas = []
                if not evento_vivo:
                    notas.append("sin cuota de BWIN para este partido ahora")
                if ajuste["home_xg_observed"] is None or ajuste["away_xg_observed"] is None:
                    notas.append("sin estadísticas de remates todavía: solo marcador, efecto marcador y añadido")
                if ancla:
                    notas.append("base anclada al mercado (cuotas previas al partido)")
                if notas:
                    st.caption(" · ".join(notas))
            else:
                c_1x2.caption(f"1X2: {live_1x2}")
                if evento_vivo:
                    st.caption("💱 BWIN en vivo: " + " · ".join(
                        f"{e} {fmt}" for e, fmt in (("1", _cuota("home")), ("X", _cuota("draw")), ("2", _cuota("away")))
                        if fmt))

            with st.expander("📐 Corners, faltas y tarjetas"):
                # De 365scores (gratis), no de Goal API. NO se guarda en el log:
                # log_match_stats guarda solo la primera foto de cada partido, así
                # que guardar en vivo dejaba datos parciales (ej. minuto 1, 0
                # corners) como si fueran los finales. Las finales las guarda
                # capture_live_odds.py cuando el partido termina.
                if stats_vivo:
                    _render_stats_table(stats_vivo)
                else:
                    st.caption("Sin estadísticas en vivo de 365scores para este partido todavía.")
            st.divider()

    with st.expander("📏 ¿Qué tan bien acierta el ajuste en vivo?"):
        try:
            evaluacion = load_live_evaluation()
        except Exception as e:
            evaluacion = pd.DataFrame()
            st.caption(f"No se pudo calcular ({e}).")
        if evaluacion.empty:
            st.caption(
                "Todavía no hay fotos en vivo de partidos ya terminados. Cada partido que se ve en "
                "esta pestaña deja una foto cada 10 minutos de juego (data/tracking/live_snapshots.csv); "
                "cuando termina, se compara qué método se acercó más al resultado final."
            )
        else:
            st.dataframe(evaluacion, width="stretch", hide_index=True,
                         column_config={"Log-loss": st.column_config.NumberColumn(format="%.3f")})
            st.caption("Log-loss del 1X2: más bajo = mejor. Con pocas fotos (y varias del mismo "
                       "partido) es solo una tendencia.")

# ============================== PESTAÑA: AMISTOSOS ==============================
with tab_amistosos:
    st.title("🌍 Amistosos internacionales")
    st.caption(
        "Amistosos de selecciones absolutas (sin categorías juveniles ni femenino), de ayer a los "
        "próximos 8 días. Sin jornada ni tabla — el modelo no tiene ese concepto acá, así que van "
        "aparte de la pestaña 'Jornada'. Cada selección se predice con SU propio historial (Nations "
        "League + amistosos previos, hasta ~2.5 años atrás) — si ninguna de las dos no tiene ese "
        "historial, el partido se lista igual pero sin predicción."
    )

    # Solo a pedido: Streamlit ejecuta TODAS las pestañas al abrir la app, y esta
    # pedía ~18 consultas a Goal API (10 días de amistosos + Nations League) en
    # cada arranque aunque nadie la mirara. Sin st.rerun(): devolvía la vista a
    # la pestaña "Jornada".
    if not st.session_state.get("amistosos_on"):
        if st.button("🌍 Cargar amistosos", type="primary"):
            st.session_state["amistosos_on"] = True
        else:
            st.caption("Consulta los amistosos solo cuando lo pidas (ahorra consultas a Goal API).")

    fixtures_amistosos = pred_amistosos = pd.DataFrame()
    if st.session_state.get("amistosos_on"):
        if st.button("🔄 Actualizar amistosos"):
            load_friendlies.clear()
            get_friendlies_predictions.clear()
        try:
            fixtures_amistosos = load_friendlies()
            pred_amistosos = get_friendlies_predictions()
        except Exception as e:
            st.error(f"No se pudieron cargar los amistosos: {e}")

    if not st.session_state.get("amistosos_on"):
        pass
    elif fixtures_amistosos.empty:
        st.info("No hay amistosos de selecciones absolutas en esta ventana de días.")
    else:
        st.caption(f"{len(fixtures_amistosos)} amistosos encontrados · "
                   f"{len(pred_amistosos)} con predicción disponible.")
        pred_by_match = (
            {(r.home_team, r.away_team, r.utc_date): r for r in pred_amistosos.itertuples()}
            if not pred_amistosos.empty else {}
        )
        estado_label = {"FINISHED": "✅ Finalizado", "CANCELLED": "🚫 Cancelado", "POSTPONED": "🚫 Aplazado"}
        for s in LIVE_STATUSES:
            estado_label[s] = "🔴 En vivo"

        for _, row in fixtures_amistosos.iterrows():
            pred_row = pred_by_match.get((row["home_team"], row["away_team"], row["utc_date"]))
            with st.container(border=True):
                estado = estado_label.get(row["status"], "")
                cabecera = format_bogota(row["utc_date"])
                if estado:
                    cabecera += f" · {estado}"
                st.caption(cabecera)

                if row["status"] == "FINISHED" and pd.notna(row["home_goals"]):
                    marcador = f"{int(row['home_goals'])}-{int(row['away_goals'])}"
                    st.markdown(f"**{row['home_team']} {marcador} {row['away_team']}**")
                else:
                    st.markdown(f"**{row['home_team']} vs {row['away_team']}**")

                if pred_row is None:
                    st.caption("Sin historial suficiente (Nations League/amistosos) para predecir este cruce.")
                else:
                    st.write(
                        f"xG modelo: {pred_row.home_xg} - {pred_row.away_xg}  ·  "
                        f"1X2: {pred_row.home_win:.0f}% / {pred_row.draw:.0f}% / {pred_row.away_win:.0f}%"
                    )
                    st.caption(
                        f"Over 2.5: {pred_row.over_2_5:.0f}% · BTTS: {pred_row.btts:.0f}% · "
                        f"Marcador top: {pred_row.top_score} ({pred_row.top_score_prob:.0f}%)"
                    )
                    if pred_row.low_confidence:
                        st.caption(f"⚠ {pred_row.confidence_note}")

# ============================== PESTAÑA: TIPS ==============================
with tab_tips:
    st.title("💡 Tips de valor del día")
    st.caption(
        f"Picks de cualquier liga con probabilidad del modelo ≥{MIN_MODEL_PROB:.0%}, solo de partidos "
        "sin baja confianza, comparados contra la mejor cuota disponible. Se sugiere apostar solo "
        "si la cuota paga más de lo que el partido 'vale' — y aun así un pick de 75% falla 1 de cada 4."
    )
    with st.expander("¿Por qué la meta no es doblar la banca cada día?"):
        st.markdown(
            "- Una apuesta que paga **2x** se gana, en el mejor caso, el **50%** de las veces "
            "(cuota justa 2.0 = 50%); con el margen de la casa, menos. Doblar = perder todo lo "
            "apostado más de la mitad de los días.\n"
            "- Combinar picks 'seguros' no lo evita: 3 picks de 78% dan ~2.1x con **47%** de acierto.\n"
            "- Encadenado es peor: doblar 7 días seguidos a 50% tiene **0.8%** de probabilidad.\n"
            "- Lo que sí hacen los apostadores profesionales: apuestas simples con valor, un "
            "**cuarto de Kelly** por apuesta (máx. 5% de la banca) y máx. 20% expuesto por día. "
            "Eso busca crecer unos puntos por semana, no 100% por día.\n"
            "- Los días sin picks con valor, lo correcto es **no apostar** — ese día no cuenta."
        )

    col_banca, col_presu, col_fecha, col_casas = st.columns([1, 1, 1, 1.3])
    banca = col_banca.number_input("Banca (COP)", min_value=10_000, value=50_000, step=10_000, key="tips_banca")
    presupuesto = col_presu.number_input(
        "Presupuesto del día (COP)", min_value=5_000, value=50_000, step=5_000, key="tips_presupuesto",
        help="Lo máximo que quieres apostar hoy, repartido en hasta 3 boletos (ver 'Plan del día').")
    hoy_col = datetime.now(BOGOTA_TZ).date()
    fecha_tips = col_fecha.date_input("Día", value=hoy_col, key="tips_fecha")
    expuesto_prepartido = 0.0  # fracción de la banca ya sugerida antes del inicio (tope diario compartido)
    solo_colombia = col_casas.toggle(
        "Solo casas que operan en Colombia", value=True, key="tips_solo_co",
        help="Betsson, Codere, 1xBet, bwin, Betway, Stake... — cuotas de sus sitios europeos, "
             "parecidas pero no idénticas a las de Colombia.",
    )

    # Reunir todas las ligas del día es pesado la primera vez (varias APIs con
    # límite de peticiones), y Streamlit ejecuta todas las pestañas en cada
    # interacción — por eso solo arranca a pedido.
    # Sin st.rerun(): el análisis se dibuja en la misma ejecución del clic —
    # un rerun forzado devolvía la vista a la pestaña "Jornada".
    if not st.session_state.get("tips_on"):
        if st.button("🔎 Analizar partidos del día", type="primary"):
            st.session_state["tips_on"] = True
        else:
            st.caption("La primera vez puede tardar 1-3 minutos (junta todas las ligas); luego queda en caché.")
    if st.session_state.get("tips_on"):
        try:
            dia = load_day_predictions(fecha_tips.isoformat())
        except Exception as e:
            st.error(f"No se pudieron reunir los partidos del día: {e}")
            dia = pd.DataFrame()

        candidatos = candidate_picks(dia) if not dia.empty else pd.DataFrame()

        # --- Cuotas del día (365scores; The Odds API de respaldo si hay clave) ---
        # Para TODAS las competiciones del día (no solo las de los candidatos):
        # corners/tarjetas necesitan la línea de la casa para armar su pick.
        inicio_local = datetime.combine(fecha_tips, datetime.min.time(), tzinfo=BOGOTA_TZ)
        desde = inicio_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        hasta = (inicio_local + pd.Timedelta(days=1)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        eventos_por_comp, restante = {}, None
        for code in (dia["competition"].unique() if not dia.empty else []):
            if code not in odds_api.SPORT_KEYS:
                continue  # amistosos: sin competición mapeada -> cuota manual
            eventos, rest, error = load_odds_table(code, desde, hasta, bool(ODDS_KEY))
            restante = rest if rest is not None else restante
            if error:
                st.caption(f"⚠ {config.COMPETITIONS.get(code, code)}: {error}")
                continue
            eventos_por_comp[code] = eventos

        def _evento(r):
            return odds_api.find_event(eventos_por_comp.get(r["competition"]) or [], r["home_team"], r["away_team"],
                                       r["utc_date"], r.get("home_team_short") or "", r.get("away_team_short") or "")

        if not candidatos.empty:
            candidatos["cuota_auto"] = None
            candidatos["casa"] = ""
            for idx, r in candidatos.iterrows():
                evento = _evento(r)
                precio = odds_api.row_price(evento, r.odds_key, only_colombia=solo_colombia) if evento else None
                if precio:
                    candidatos.at[idx, "cuota_auto"] = precio[0]
                    candidatos.at[idx, "casa"] = precio[1]
        # Corners y tarjetas (experimental, ver value_tips.TOTALS_MODEL_WEIGHT):
        # solo en la línea principal de la casa y con su cuota real.
        totales = totals_candidates(dia[~dia.get("low_confidence", False).astype(bool)] if not dia.empty else dia,
                                    _evento)
        if not totales.empty:
            candidatos = pd.concat([candidatos, totales], ignore_index=True)

        st.caption(f"{len(dia)} partidos por jugar ese día · {len(candidatos)} picks con ≥{MIN_MODEL_PROB:.0%} "
                   "en partidos sin baja confianza"
                   + (f" (incluye {len(totales)} de corners/tarjetas)" if not totales.empty else "") + ".")

        if dia.empty:
            st.info("No hay partidos por jugar ese día en las competiciones que sigue el proyecto.")
        elif candidatos.empty:
            st.warning(
                "🚫 **Hoy no hay opciones de muy bajo riesgo.** Ningún pick llega al "
                f"{MIN_MODEL_PROB:.0%} en partidos con datos suficientes. Recomendación: no apostar — "
                "este día no cuenta para la meta."
            )
        else:
            if restante is not None:
                st.caption(f"Créditos de The Odds API restantes este mes: {restante}")

            # --- Tabla editable: la cuota manual manda sobre la automática ---
            editor = pd.DataFrame({
                "Partido": candidatos["partido"],
                "Liga": candidatos["competition_name"],
                "Hora": candidatos["utc_date"].map(format_bogota),
                "Mercado": candidatos["mercado"],
                "Pick": candidatos["pick"],
                "Prob. modelo": (candidatos["prob_modelo"] * 100).round(0),
                "Cuota justa": candidatos["cuota_justa"],
                "Mejor cuota": pd.to_numeric(candidatos["cuota_auto"], errors="coerce").astype(float),
                "Casa": candidatos["casa"],
                "Cuota manual": pd.Series([None] * len(candidatos), index=candidatos.index, dtype="float"),
            })
            st.markdown("**Picks candidatos** — corrige 'Cuota manual' con la de tu casa antes de apostar "
                        "(decimales con coma, ej. 1,45):")
            editado = st.data_editor(
                editor, hide_index=True, width="stretch",
                key=f"tips_editor_{fecha_tips.isoformat()}",
                disabled=[c for c in editor.columns if c != "Cuota manual"],
                column_config={
                    "Partido": st.column_config.Column(pinned=True),
                    "Prob. modelo": st.column_config.NumberColumn(format="%.0f%%"),
                    "Mejor cuota": st.column_config.NumberColumn(format="%.2f"),
                    # Sin max_value a propósito: recortaría un error de tipeo
                    # (136) a un valor que parece real (20); value_tips lo marca.
                    "Cuota manual": st.column_config.NumberColumn(
                        min_value=1.01, step=0.01, format="%.2f",
                        help="Cuota decimal de tu casa, ej. 1,45 (si tu navegador está en "
                             "español usa coma: con punto puede guardarse como 145).",
                    ),
                },
            )

            manual = pd.to_numeric(editado["Cuota manual"], errors="coerce")
            auto = pd.to_numeric(candidatos["cuota_auto"], errors="coerce")
            # Una cuota manual fuera de rango NO cae a la automática en silencio:
            # queda marcada como inválida para que se corrija a la vista.
            candidatos["cuota"] = manual.where(manual.notna(), auto)
            evaluado = evaluate(candidatos, float(banca))
            resumen = day_summary(evaluado, float(banca))
            expuesto_prepartido = float(evaluado["fraccion"].sum())

            st.divider()
            if resumen["n"] == 0:
                evaluados = evaluado["estado"].isin(["Sin valor", "Otro pick del mismo partido"]).sum()
                pendientes = len(evaluado) - evaluados
                if evaluados == 0:
                    st.info(
                        f"✍️ Falta la cuota de {pendientes} pick(s) para poder evaluarlos: escríbela en "
                        "'Cuota manual' (o configura ODDS_API_KEY para traerla sola)."
                    )
                else:
                    st.warning(
                        "🚫 **Hoy no hay tips con valor.** Con las cuotas disponibles, ninguna apuesta "
                        "paga lo suficiente para su riesgo"
                        + (f" ({pendientes} pick(s) siguen sin cuota válida)" if pendientes else "")
                        + ". Recomendación: no apostar — este día no cuenta para la meta."
                    )
            else:
                # Solo se guarda en el historial lo de hoy o días futuros (un día
                # pasado ya no tiene partidos por jugar que recomendar).
                render_daily_plan(evaluado, float(presupuesto), float(banca),
                                  fecha=fecha_tips.isoformat() if fecha_tips >= hoy_col else None)
                with st.expander("Reparto por ¼ de Kelly sobre la banca (método anterior, montos chicos)"):
                    st.subheader(f"✅ {resumen['n']} apuesta(s) SIMPLE(S) sugerida(s)")
                    st.caption("Cada apuesta va por separado, con su propio monto — no es una combinada: "
                               "si una falla, las otras se cobran igual.")
                    # Todo en GANANCIA/PÉRDIDA NETA (antes "Si todo acierta" mostraba
                    # la banca final, $62.066, y se confundía con lo que se gana).
                    neta_todo = resumen["si_todo_acierta"] - banca
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total a apostar", f"${resumen['apostado']:,.0f}",
                              f"{resumen['apostado'] / banca:.0%} de la banca", delta_color="off")
                    m2.metric("Ganancia neta si todas aciertan", f"+${neta_todo:,.0f}",
                              f"prob. {resumen['prob_todo_acierta']:.0%}", delta_color="off")
                    m3.metric("Pérdida si todas fallan", f"-${resumen['apostado']:,.0f}")
                    m4.metric("Ganancia esperada", f"+${resumen['esperado']:,.0f}",
                              help="Promedio a la larga si se repitieran muchas veces apuestas como estas: "
                                   "unos días se gana más, otros se pierde. Es la medida de si conviene apostar.")
                    st.caption(f"Tu banca: ${banca:,.0f} → ${resumen['si_todo_acierta']:,.0f} si todas aciertan · "
                               f"${resumen['si_todo_falla']:,.0f} si todas fallan.")

                    apuestas = evaluado[evaluado["monto"] > 0]
                    st.dataframe(
                        pd.DataFrame({
                            "Partido": apuestas["partido"],
                            "Hora": apuestas["utc_date"].map(format_bogota),
                            "Mercado": apuestas["mercado"],
                            "Pick": apuestas["pick"],
                            "Cuota": apuestas["cuota"].astype(float).round(2),
                            "Casa": apuestas["casa"].where(manual.reindex(apuestas.index).isna(), "Tu casa (manual)"),
                            "Valor": (apuestas["valor"] * 100).round(1),
                            "Apostar": apuestas["monto"],
                            "Recibes si acierta": (apuestas["monto"] * apuestas["cuota"].astype(float)).round(0),
                            "Ganancia neta": apuestas["ganancia"],
                        }),
                        hide_index=True, width="stretch",
                        column_config={
                            "Partido": st.column_config.Column(pinned=True),
                            "Valor": st.column_config.NumberColumn(format="+%.1f%%"),
                            "Apostar": st.column_config.NumberColumn(format="$%d"),
                            "Recibes si acierta": st.column_config.NumberColumn(
                                format="$%d", help="Monto × cuota: lo que paga la casa, incluida tu apuesta."),
                            "Ganancia neta": st.column_config.NumberColumn(
                                format="$%d", help="Lo que recibes menos lo apostado."),
                        },
                    )
                    if len(apuestas) >= 2:
                        # Comparación con armarlas como UNA combinada — la cuenta que
                        # suele hacerse a mano (cuota = producto de las cuotas).
                        cuota_comb = float(apuestas["cuota"].astype(float).prod())
                        prob_comb = float(apuestas["prob_ajustada"].prod())
                        with st.expander("¿Y si las combino en una sola apuesta?"):
                            st.markdown(
                                f"- Cuota combinada: **{cuota_comb:.2f}** (= "
                                + " × ".join(f"{c:.2f}" for c in apuestas["cuota"].astype(float)) + ")\n"
                                f"- Apostando los mismos **${resumen['apostado']:,.0f}**: recibes "
                                f"**${resumen['apostado'] * cuota_comb:,.0f}** (ganancia neta "
                                f"${resumen['apostado'] * (cuota_comb - 1):,.0f}) — pero **solo si aciertan las "
                                f"{len(apuestas)}**: probabilidad **{prob_comb:.0%}**. Si falla una sola, pierdes todo.\n"
                                f"- Por cada $10.000 en la combinada recibes ${10_000 * cuota_comb:,.0f} si acierta.\n"
                                f"- Ganancia esperada de la combinada: **{prob_comb * cuota_comb - 1:+.1%}** por peso "
                                f"apostado, contra **{resumen['esperado'] / resumen['apostado']:+.1%}** de las simples."
                            )
                            st.caption(
                                f"Si cada pick de verdad tiene valor, la combinada lo multiplica — pero gana solo "
                                f"{prob_comb:.0%} de las veces, y si el modelo se equivoca en UN pick, el error "
                                "también se multiplica. Por eso la base son las simples (¼ de Kelly); si juegas la "
                                "combinada, que sea con un monto chico (~1% de la banca, como en '🎯 Combinadas')."
                            )
                    st.caption(
                        "Apuestas simples, no combinadas. 'Valor' = ganancia esperada por cada peso apostado, "
                        "ya descontando la mitad de la ventaja que ve el modelo (un modelo simple se equivoca "
                        "más que el mercado). Montos: ¼ de Kelly, máx. 5% por apuesta y 20% por día."
                    )

            descartes = evaluado[evaluado["estado"] != "Apostar"]
            if not descartes.empty:
                with st.expander(f"Picks descartados ({len(descartes)})"):
                    st.dataframe(
                        pd.DataFrame({
                            "Partido": descartes["partido"],
                            "Pick": descartes["mercado"] + ": " + descartes["pick"],
                            "Cuota": pd.to_numeric(descartes["cuota"], errors="coerce").round(2),
                            "Motivo": descartes["estado"],
                        }),
                        hide_index=True, width="stretch",
                    )

        # Combinadas entre TODAS las ligas del día (la de "Jornada" solo combina
        # partidos de una misma liga y jornada).
        if not dia.empty:
            st.divider()
            st.subheader("🎯 Combinadas del día — todas las ligas")
            combinadas_dia, avisos_dia = parlays_with_real_odds(dia, only_colombia=solo_colombia)
            render_parlays(combinadas_dia, bankroll=float(banca), avisos=avisos_dia)
            if fecha_tips >= hoy_col:
                for nivel in sorted({c.risk for c in combinadas_dia}):
                    tips_log.record(fecha_tips.isoformat(), f"Combinada del día · {nivel}",
                                    tips_log.tickets_from_parlays([c for c in combinadas_dia if c.risk == nivel],
                                                                  dia, float(banca)), "app")

    # ------------------------- Tips EN VIVO (BWIN) -------------------------
    # Misma regla de valor, con la probabilidad ajustada en vivo (la misma de
    # la pestaña "En vivo") y la cuota en vivo de BWIN vía 365scores. No
    # depende del botón de arriba: es liviano (reutiliza las cachés de "En vivo").
    st.divider()
    st.subheader("🔴 Tips en vivo (BWIN)")
    live_refresh_button("refresh_tips_vivo")
    try:
        vivos = load_live_matches(tuple(config.COMPETITIONS.keys()))
    except Exception:
        vivos = pd.DataFrame()
    if vivos.empty:
        st.caption("No hay partidos en juego ahora mismo.")
    else:
        try:
            cuotas_en_vivo = load_live_odds(tuple(sorted(set(vivos["competition"]))))
        except Exception:
            cuotas_en_vivo = []
        if live_data_time(cuotas_en_vivo):
            st.caption(f"Cuotas BWIN de las {live_data_time(cuotas_en_vivo)}.")
        entradas, sin_ancla, sin_cuota = [], 0, 0
        for _, row in vivos.iterrows():
            row, evento, stats_365 = live_event_state(row, cuotas_en_vivo)
            ajuste, ancla, _ = live_adjustment(row, stats_365)
            if not ajuste:
                continue
            if not ancla:
                sin_ancla += 1
                continue
            if not evento:
                sin_cuota += 1
                continue
            entradas.append({
                "competition": row.competition, "competition_name": row.competition_name,
                "utc_date": row.utc_date, "home_team": row.home_team, "away_team": row.away_team,
                "en_vivo": f"{row.minuto_estimado}' · {row.home_goals}-{row.away_goals}",
                "probs": {
                    "home_win": ajuste["home_win"], "draw": ajuste["draw"], "away_win": ajuste["away_win"],
                    "over_2_5": ajuste["over_2_5"], "under_2_5": ajuste["under_2_5"],
                    "dc_1x": ajuste["home_win"] + ajuste["draw"], "dc_x2": ajuste["away_win"] + ajuste["draw"],
                },
                "odds": {k: precio[0] for k in odds_api.PRICE_KEYS if (precio := odds_api.row_price(evento, k))},
            })
        candidatos_vivo = live_candidate_picks(entradas)
        avisos_vivo = []
        if sin_ancla:
            avisos_vivo.append(f"{sin_ancla} partido(s) sin cuota previa guardada (el ajuste no está anclado "
                               "al mercado): no se recomiendan")
        if sin_cuota:
            avisos_vivo.append(f"{sin_cuota} partido(s) sin cuota en vivo de BWIN ahora")
        if candidatos_vivo.empty:
            st.info(f"Ningún pick en vivo llega al {MIN_MODEL_PROB:.0%} de probabilidad en este momento "
                    f"({len(vivos)} partido(s) en juego).")
        else:
            candidatos_vivo["cuota"] = pd.to_numeric(candidatos_vivo["cuota_auto"], errors="coerce")
            evaluado_vivo = evaluate(candidatos_vivo, float(banca),
                                     max_exposure=MAX_DAILY_EXPOSURE - expuesto_prepartido,
                                     model_weight=LIVE_MODEL_WEIGHT)
            tips_log.record(datetime.now(BOGOTA_TZ).date().isoformat(), "En vivo",
                            tips_log.tickets_from_singles(evaluado_vivo), "app")
            orden = {"Apostar": 0, "Sin valor": 1, "Otro pick del mismo partido": 2}
            evaluado_vivo = evaluado_vivo.assign(_o=evaluado_vivo["estado"].map(orden).fillna(3))                 .sort_values(["_o", "valor"], ascending=[True, False])
            n_apostar = int((evaluado_vivo["monto"] > 0).sum())
            if n_apostar:
                st.success(f"✅ {n_apostar} apuesta(s) en vivo con valor ahora mismo")
            else:
                st.warning("🚫 Ningún pick en vivo paga lo suficiente para su riesgo ahora: no apostar.")
            st.dataframe(
                pd.DataFrame({
                    "Partido": evaluado_vivo["partido"],
                    "En vivo": evaluado_vivo["en_vivo"],
                    "Mercado": evaluado_vivo["mercado"],
                    "Pick": evaluado_vivo["pick"],
                    "Prob. en vivo": (evaluado_vivo["prob_modelo"] * 100).round(0),
                    "Cuota justa": evaluado_vivo["cuota_justa"],
                    "Cuota BWIN": evaluado_vivo["cuota"].round(2),
                    "Valor": (evaluado_vivo["valor"] * 100).round(1),
                    "Acción": evaluado_vivo["estado"],
                    "Apostar": evaluado_vivo["monto"],
                    "Recibes si acierta": (evaluado_vivo["monto"] * evaluado_vivo["cuota"].fillna(0)).round(0),
                    "Ganancia neta": evaluado_vivo["ganancia"],
                }),
                hide_index=True, width="stretch",
                column_config={
                    "Partido": st.column_config.Column(pinned=True),
                    "Prob. en vivo": st.column_config.NumberColumn(format="%.0f%%"),
                    "Valor": st.column_config.NumberColumn(format="%+.1f%%"),
                    "Apostar": st.column_config.NumberColumn(format="$%d"),
                    "Recibes si acierta": st.column_config.NumberColumn(format="$%d"),
                    "Ganancia neta": st.column_config.NumberColumn(format="$%d"),
                },
            )
            st.caption(
                "🧪 Experimental: el ajuste en vivo todavía no se validó contra el mercado, así que acá el "
                f"modelo pesa solo {LIVE_MODEL_WEIGHT:.0%} y la cuota {1 - LIVE_MODEL_WEIGHT:.0%} al calcular el "
                "valor. La cuota en vivo cambia en segundos: confirma en BWIN que siga igual o mejor antes de "
                "apostar. 'Cuota justa' = la mínima para no perder plata a la larga según el modelo. "
                "El tope de 20% diario se comparte con los tips prepartido."
            )
        if avisos_vivo:
            st.caption(" · ".join(avisos_vivo))


# ============================== PESTAÑA: REPORTES ==============================
# Colores (guía de visualización del proyecto): azul = acierto real /
# fortaleza, naranja = lo esperado por el modelo, rojo = debilidad, gris neutro
# en el punto medio de la escala divergente.
COLOR_REAL, COLOR_ESPERADO = "#3987e5", "#eb6834"
COLOR_FUERTE, COLOR_NEUTRO, COLOR_DEBIL = "#3987e5", "#8a8a86", "#e66767"


@st.cache_data(ttl=600, show_spinner="Actualizando resultados de los tips...")
def load_reports_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(log de predicciones, log de tips). Antes resuelve los tips pendientes
    con los marcadores finales de 365scores (gratis), como mucho cada 10 min."""
    tips_log.resolve()
    same_day.resolve_finished()  # las del día, sin esperar a la tarea de los lunes
    pred = config.ROOT_DIR / "data" / "tracking" / "predictions_log.csv"
    return (pd.read_csv(pred) if pred.exists() else pd.DataFrame()), tips_log.load()


def _bars_vs_expected(tabla: pd.DataFrame, categoria: str, titulo: str, horizontal: bool = True):
    """Barras de % de acierto real con una marca del % esperado por el modelo."""
    base = alt.Chart(tabla)
    tooltip = [alt.Tooltip(f"{categoria}:{'T' if not horizontal else 'N'}", title=titulo),
               alt.Tooltip("predicciones:Q", title="Predicciones"),
               alt.Tooltip("acierto_pct:Q", title="Acierto real %", format=".0f"),
               alt.Tooltip("esperado_pct:Q", title="Esperado %", format=".0f")]
    if horizontal:
        orden = alt.EncodingSortField(field="acierto_pct", order="descending")
        barras = base.mark_bar(color=COLOR_REAL, cornerRadiusEnd=4, height=14).encode(
            y=alt.Y(f"{categoria}:N", sort=orden, title=None),
            x=alt.X("acierto_pct:Q", title="% de acierto", scale=alt.Scale(domain=[0, 100])), tooltip=tooltip)
        marca = base.mark_tick(color=COLOR_ESPERADO, thickness=3, size=18).encode(
            y=alt.Y(f"{categoria}:N", sort=orden), x="esperado_pct:Q", tooltip=tooltip)
        alto = alt.Step(30)  # alto por fila: un alto fijo se ignoraba y las barras se encimaban
    else:
        # Eje ordinal (solo los días con partidos): con eje de tiempo, los días
        # sin partidos dejaban huecos y las barras quedaban como líneas.
        barras = base.mark_bar(color=COLOR_REAL, cornerRadiusEnd=4).encode(
            x=alt.X("etiqueta_x:O", sort="ascending", title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("acierto_pct:Q", title="% de acierto", scale=alt.Scale(domain=[0, 100])), tooltip=tooltip)
        marca = base.mark_tick(color=COLOR_ESPERADO, thickness=3).encode(
            x=alt.X("etiqueta_x:O", sort="ascending"), y="esperado_pct:Q", tooltip=tooltip)
        alto = 260
    return (barras + marca).properties(height=alto)


MAX_TEAMS_CHART = 30
# Aclaración pedida tras una duda real (Deportivo 1-1 Levante, oct-2026): el
# mercado mide si ACERTÓ EL PICK del modelo, que puede ser Sí o No / más o menos.
PICK_NOTE = ("Cada mercado mide si acertó el **pick del modelo**, no si pasó el evento: en Ambos anotan el "
             "pick puede ser Sí o No (y en goles, más o menos de 2.5). Ej.: si el modelo dijo 'No' y quedó 1-1, "
             "es un fallo. El pick de cada partido se ve en 'Partidos del día'.")


def _diverging_bars(tabla: pd.DataFrame, categoria: str, titulo: str):
    """Barras de (real − esperado) alrededor de 0: azul = fortaleza, rojo = debilidad."""
    orden = alt.EncodingSortField(field="diferencia", order="descending")
    tope = max(10.0, float(tabla["diferencia"].abs().max()))
    barras = alt.Chart(tabla).mark_bar(cornerRadiusEnd=4, height=16).encode(
        y=alt.Y(f"{categoria}:N", sort=orden, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("diferencia:Q", title="Real − esperado (pts)", scale=alt.Scale(domain=[-tope, tope])),
        color=alt.condition("datum.diferencia >= 0", alt.value(COLOR_FUERTE), alt.value(COLOR_DEBIL)),
        tooltip=[alt.Tooltip(f"{categoria}:N", title=titulo), alt.Tooltip("predicciones:Q", title="Predicciones"),
                 alt.Tooltip("acierto_pct:Q", title="Acierto real %", format=".0f"),
                 alt.Tooltip("esperado_pct:Q", title="Esperado %", format=".0f"),
                 alt.Tooltip("diferencia:Q", title="Real − esperado (pts)", format="+.0f")],
    )
    cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=COLOR_NEUTRO, strokeWidth=2).encode(x="x:Q")
    return (barras + cero).properties(height=alt.Step(24))


def _summary_table(tabla: pd.DataFrame, categoria: str, titulo: str) -> None:
    st.dataframe(
        tabla.rename(columns={categoria: titulo}), hide_index=True, width="stretch",
        column_config={
            "predicciones": st.column_config.NumberColumn("Predicciones"),
            "aciertos": st.column_config.NumberColumn("Aciertos"),
            "acierto_pct": st.column_config.NumberColumn("% acierto", format="%.0f%%"),
            "esperado_pct": st.column_config.NumberColumn("% esperado", format="%.0f%%",
                                                          help="Probabilidad que el modelo le daba a sus picks."),
            "diferencia": st.column_config.NumberColumn("Real − esperado", format="%+.0f pts",
                                                        help="Positivo = fortaleza; negativo = debilidad."),
        },
    )


def _render_detail(long: pd.DataFrame, caption: str | None = None) -> None:
    """Tabla de predicciones con el pick del modelo (para entender cada ✔/✘)."""
    detalle = reports.prediction_detail(long)
    if detalle.empty:
        st.caption("Sin predicciones para ese detalle.")
        return
    detalle["Liga"] = detalle["Liga"].map(lambda c: config.COMPETITIONS.get(c, c))
    st.dataframe(detalle, hide_index=True, width="stretch", column_config={
        "Prob.": st.column_config.NumberColumn("Prob. del pick", format="%.0f%%",
                                               help="Probabilidad que el modelo le daba a su pick."),
        "Pick del modelo": st.column_config.Column(
            help="Lo que predijo el modelo: Local/Empate/Visitante, 1X/X2, +2.5/-2.5, Sí/No, más/menos de la línea."),
    })
    if caption:
        st.caption(caption)


def _league_multiselect(container, label: str, opciones: list, key: str) -> list:
    """Multiselect de ligas que agrega solas las ligas NUEVAS a la selección:
    Streamlit recuerda la selección anterior de la sesión, y una liga agregada
    después (ej. la Liga BetPlay, oct-2026) quedaba sin marcar y no aparecía."""
    vistas_key = f"{key}__vistas"
    vistas = set(st.session_state.get(vistas_key, []))
    if key in st.session_state:
        nuevas = [c for c in opciones if c not in vistas]
        st.session_state[key] = [c for c in st.session_state[key] if c in opciones] + nuevas
    st.session_state[vistas_key] = list(opciones)
    return container.multiselect(label, opciones, default=None if key in st.session_state else opciones, key=key,
                                 format_func=lambda c: config.COMPETITIONS.get(c, c))


def _min_sample_slider(label: str, conteos: pd.Series, defecto: int, key: str, help: str | None = None,
                       container=None) -> int:
    """Slider de muestra mínima con el rango que de verdad hay en el filtro: de
    la menor a la mayor cantidad de predicciones (antes era fijo 1-30: con pocos
    días filtrados, el valor por defecto superaba a todos y la tabla quedaba vacía)."""
    piso = int(conteos.min()) if len(conteos) else 1
    tope = int(conteos.max()) if len(conteos) else 1
    if tope <= piso:
        return piso
    # Si el filtro cambió y el valor guardado quedó fuera del nuevo rango, se ajusta.
    if key in st.session_state:
        st.session_state[key] = min(max(st.session_state[key], piso), tope)
    return (container or st).slider(label, piso, tope, min(max(defecto, piso), tope), key=key, help=help)


def _render_strengths_map(datos: pd.DataFrame) -> None:
    st.markdown("**Mapa de fortalezas y debilidades — liga × mercado**")
    mapa = reports.summarize(datos, ["competition", "mercado"])
    # Por defecto 1 = todas las celdas con lo que haya en el rango (pedido del
    # usuario); el slider queda para ocultar las de muestra chica.
    # Siempre arranca en el mínimo real (pedido del usuario): clave nueva para que
    # una sesión que guardaba el valor viejo (8) no lo arrastre.
    min_celda = _min_sample_slider("Mínimo de predicciones por celda",
                                   mapa["predicciones"] if not mapa.empty else pd.Series(dtype=int), 1,
                                   "rep_min_celda_v2",
                                   help="Súbelo para ocultar celdas con pocas predicciones: ahí la diferencia "
                                        "es casi puro azar.")
    mapa = mapa[mapa["predicciones"] >= min_celda] if not mapa.empty else mapa
    if mapa.empty:
        st.caption("Ninguna combinación llega a ese mínimo en el filtro.")
        return
    mapa["liga"] = mapa["competition"].map(lambda c: config.COMPETITIONS.get(c, c))
    mapa["etiqueta"] = [f"{a:.0f}% · {n}" for a, n in zip(mapa["acierto_pct"], mapa["predicciones"])]
    tope = max(15.0, float(mapa["diferencia"].abs().max()))
    base = alt.Chart(mapa).encode(
        x=alt.X("mercado:N", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("liga:N", title=None, axis=alt.Axis(labelLimit=240, labelOverlap=False)),
        tooltip=[alt.Tooltip("liga:N", title="Liga"), alt.Tooltip("mercado:N", title="Mercado"),
                 alt.Tooltip("predicciones:Q", title="Predicciones"),
                 alt.Tooltip("acierto_pct:Q", title="Acierto real %", format=".0f"),
                 alt.Tooltip("esperado_pct:Q", title="Esperado %", format=".0f"),
                 alt.Tooltip("diferencia:Q", title="Real − esperado (pts)", format="+.0f")],
    )
    celdas = base.mark_rect(cornerRadius=4, stroke="#1a1a19", strokeWidth=2).encode(
        color=alt.Color("diferencia:Q", title="Real − esperado (pts)",
                        scale=alt.Scale(domain=[-tope, 0, tope], range=[COLOR_DEBIL, COLOR_NEUTRO, COLOR_FUERTE])))
    texto = base.mark_text(fontSize=11, color="#ffffff").encode(text="etiqueta:N")
    # Alto por fila: con un alto fijo, Vega ocultaba las etiquetas de ligas que no cabían.
    st.altair_chart((celdas + texto).properties(height=alt.Step(40)), use_container_width=True)
    st.caption("Cada celda: % de acierto real · cantidad de predicciones. Azul = acierta más de lo que el "
               "modelo esperaba (fortaleza); rojo = menos (debilidad); gris = según lo esperado.")

    celdas_det = [f"{l} · {m}" for l, m in zip(mapa["liga"], mapa["mercado"])]
    celda = st.selectbox("Ver el detalle de predicciones de la celda", celdas_det, key="rep_celda_detalle")
    fila = mapa.iloc[celdas_det.index(celda)]
    _render_detail(datos[(datos["competition"] == fila["competition"]) & (datos["mercado"] == fila["mercado"])],
                   PICK_NOTE)

    st.markdown("**Real vs esperado por mercado**")
    por_mercado = reports.summarize(datos, ["mercado"])
    st.altair_chart(_bars_vs_expected(por_mercado, "mercado", "Mercado"), use_container_width=True)
    st.caption("Barra azul: % de acierto real · marca naranja: % que el modelo esperaba. Barra más allá de "
               "la marca = el modelo es más fuerte de lo que cree en ese mercado.")


def _render_tips_report(boletos: pd.DataFrame, cerrados: pd.DataFrame, mostrar_planes: bool = True) -> None:
    if boletos.empty:
        st.info("Todavía no hay boletos guardados. Se guardan solos cada vez que la pestaña Tips arma el Plan "
                "del día, las combinadas del día o un tip en vivo (y el resultado se completa cuando terminan "
                "los partidos).")
        return
    por_plan = reports.tips_by_plan(boletos)
    if not por_plan.empty:
        st.markdown("**Resultado por plan**")
        st.dataframe(por_plan, hide_index=True, width="stretch", column_config={
            "plan": "Plan", "boletos": "Boletos resueltos", "ganados": "Ganados",
            "ganados_pct": st.column_config.NumberColumn("% ganados", format="%.0f%%"),
            "apostado": st.column_config.NumberColumn("Apostado", format="$%d"),
            "ganancia": st.column_config.NumberColumn("Ganancia neta", format="$%d"),
            "roi_pct": st.column_config.NumberColumn("Retorno", format="%+.0f%%", help="Ganancia neta / apostado."),
        })
        diario = (cerrados.assign(ganancia=pd.to_numeric(cerrados["ganancia"], errors="coerce"))
                  .groupby("fecha", as_index=False)["ganancia"].sum())
        diario["fecha"] = pd.to_datetime(diario["fecha"])
        st.altair_chart(alt.Chart(diario).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("fecha:T", title=None), y=alt.Y("ganancia:Q", title="Ganancia neta ($)"),
            color=alt.condition("datum.ganancia >= 0", alt.value(COLOR_FUERTE), alt.value(COLOR_DEBIL)),
            tooltip=[alt.Tooltip("fecha:T", title="Día"), alt.Tooltip("ganancia:Q", title="Ganancia neta", format="$,.0f")],
        ).properties(height=220), use_container_width=True)
        st.caption("Ganancia neta por día de todos los boletos resueltos (con los montos sugeridos).")
    planes_disp = sorted(boletos["plan"].unique())
    plan_sel = (st.multiselect("Planes", planes_disp, default=planes_disp, key="rep_planes")
                if mostrar_planes and len(planes_disp) > 1 else planes_disp)
    vista = boletos[boletos["plan"].isin(plan_sel)].sort_values("fecha", ascending=False).head(80)
    st.markdown(f"**Boletos** ({len(vista)} más recientes) — ✔ acertado · ✘ fallado · · pendiente")
    pesos = lambda v: f"${v:,.0f}" if pd.notna(v) else "-"  # noqa: E731
    # Índice único (el Styler no admite repetidos: varios boletos del mismo día).
    etiquetas, vistos = [], {}
    for f, pl, b in zip(vista["fecha"], vista["plan"], vista["boleto"]):
        base = f"{f} · {pl}" + (f" #{int(b)}" if pd.notna(b) else "")
        vistos[base] = vistos.get(base, 0) + 1
        etiquetas.append(base if vistos[base] == 1 else f"{base} ({vistos[base]})")
    st.table(
        vista[["tipo", "picks", "cuota", "monto", "resultado", "ganancia"]]
        .rename(columns={"tipo": "Tipo", "picks": "Picks", "cuota": "Cuota", "monto": "Monto",
                         "resultado": "Resultado", "ganancia": "Ganancia"})
        .set_axis(pd.Index(etiquetas, name="Boleto"), axis=0).style
        .format({"Cuota": lambda v: f"{v:.2f}" if pd.notna(v) else "-", "Monto": pesos, "Ganancia": pesos})
        .set_properties(subset=["Picks"], **{"white-space": "pre-wrap", "min-width": "18em"})
        .set_properties(subset=["Cuota", "Monto", "Ganancia"], **{"white-space": "nowrap"})
    )


# Colores de estado (paleta de estados de la guía, siempre con ícono + texto).
COLOR_ACERTADO, COLOR_FALLADO, COLOR_PENDIENTE = "#0ca30c", "#d03b3b", "#5f5f5b"
DAY_MARKETS = ["1X2", "Doble oportunidad", "Goles 2.5", "Ambos anotan", "Corners", "Faltas", "Tarjetas"]


def _render_day_matches(log_pred: pd.DataFrame, log_tips: pd.DataFrame, ligas: list, equipos: list,
                        mercados: list, incluir_bt: bool, dia_defecto, tipo: str = reports.TIPO_MODELO) -> None:
    """Todos los partidos de un día, con el pick de cada mercado y si se ganó."""
    hoy = datetime.now(BOGOTA_TZ).date()
    c_dia, c_merc = st.columns([1, 2.5])
    dia = c_dia.date_input("Día", value=min(dia_defecto, hoy), max_value=hoy, key="rep_dia")
    log = log_pred
    if not incluir_bt and not log.empty:
        log = log[~log["source"].astype(str).str.startswith("backtest")]
    picks = reports.day_picks(log, log_tips, dia, tipo)
    if not picks.empty:
        picks = picks[picks["competition"].isin(ligas)]
        if equipos:
            picks = picks[picks["partido"].map(lambda t: any(e in t for e in equipos))]
        # Filtro de mercado propio de esta subpestaña (pedido del usuario): las
        # columnas del gráfico. Arranca con los mercados elegidos arriba; un
        # mercado que aparece por primera vez entra solo si está elegido arriba.
        largo = {"1X2": "Resultado (1X2)", "Goles 2.5": "Goles +/-2.5"}
        presentes = set(picks["mercado"])
        opciones = [m for m in DAY_MARKETS if m in presentes] + sorted(presentes - set(DAY_MARKETS))
        key, vistos_key = "rep_partidos_mercados", "rep_partidos_mercados__vistos"
        de_arriba = [m for m in opciones if largo.get(m, m) in mercados]
        if key in st.session_state:
            vistos = set(st.session_state.get(vistos_key, []))
            st.session_state[key] = ([m for m in st.session_state[key] if m in opciones]
                                     + [m for m in de_arriba if m not in vistos and m not in st.session_state[key]])
        st.session_state[vistos_key] = sorted(set(st.session_state.get(vistos_key, [])) | set(opciones))
        elegidos = c_merc.multiselect("Mercados", opciones, key=key,
                                      default=None if key in st.session_state else de_arriba)
        picks = picks[picks["mercado"].isin(elegidos)]
    if picks.empty:
        st.info("No hay predicciones registradas para ese día con los filtros elegidos."
                + ("" if tipo == reports.TIPO_MODELO else
                   f" Los boletos de Tips se guardan desde el 16-oct: revisa días posteriores o el tipo 'Modelo'."))
        return

    resueltos = picks[picks["estado"] != "Pendiente"]
    ganados = int((resueltos["estado"] == "Acertado").sum())
    st.markdown(f"**{picks['partido'].nunique()} partido(s) · {ganados} de {len(resueltos)} picks resueltos "
                f"acertados**" + (f" · {len(picks) - len(resueltos)} pendientes" if len(resueltos) < len(picks) else ""))

    icono = {"Acertado": "✔", "Fallado": "✘", "Pendiente": "·"}
    picks = picks.assign(
        etiqueta=[f"{icono[e]} {p}" + (f" · {pr:.0f}%" if pd.notna(pr) else "")
                  for e, p, pr in zip(picks["estado"], picks["pick"], picks["prob"])],
        orden=pd.to_datetime(picks["utc_date"], utc=True, errors="coerce"),
    )
    mercados = [m for m in DAY_MARKETS if m in set(picks["mercado"])]
    mercados += sorted(m for m in set(picks["mercado"]) if m not in mercados)
    partidos = picks.sort_values(["orden", "partido"])["partido"].drop_duplicates().tolist()
    base = alt.Chart(picks).encode(
        x=alt.X("mercado:N", sort=mercados, title=None, axis=alt.Axis(orient="top", labelAngle=0, labelLimit=140)),
        y=alt.Y("partido:N", sort=partidos, title=None, axis=alt.Axis(labelLimit=320)),
        tooltip=[alt.Tooltip("partido:N", title="Partido"), alt.Tooltip("mercado:N", title="Mercado"),
                 alt.Tooltip("pick:N", title="Pick"), alt.Tooltip("prob:Q", title="Probabilidad %", format=".0f"),
                 alt.Tooltip("estado:N", title="Estado")],
    )
    celdas = base.mark_rect(cornerRadius=4, stroke="#1a1a19", strokeWidth=2).encode(
        color=alt.Color("estado:N", title=None,
                        scale=alt.Scale(domain=["Acertado", "Fallado", "Pendiente"],
                                        range=[COLOR_ACERTADO, COLOR_FALLADO, COLOR_PENDIENTE]),
                        legend=alt.Legend(orient="top")))
    texto = base.mark_text(fontSize=11, color="#ffffff").encode(text="etiqueta:N")
    st.altair_chart((celdas + texto).properties(height=alt.Step(38)), use_container_width=True)
    st.caption(("Cada celda es el pick del modelo para ese mercado" if tipo == reports.TIPO_MODELO
                else f"Cada celda es un pick que entró en boletos del plan '{tipo}' de Tips")
               + " (con su probabilidad): ✔ acertado · ✘ fallado · · pendiente. Doble oportunidad = la más "
               "probable entre 1X y X2. Corners/faltas/tarjetas: más o menos que la línea (se resuelven cuando "
               "hay estadísticas finales).")


with tab_reportes:
    st.title("📊 Reportes")
    st.caption(
        "Qué tan bien acertaron las predicciones del modelo y los boletos de Tips, por día, liga, equipo y "
        "mercado. 'Esperado' es la probabilidad que el modelo le daba a su pick: si el acierto real queda por "
        "encima, es una fortaleza; por debajo, una debilidad. Corners, faltas y tarjetas se comparan contra "
        "50% (su pick es más/menos que una línea puesta en el promedio)."
    )
    try:
        log_pred, log_tips = load_reports_data()
    except Exception as e:
        st.error(f"No se pudieron leer los historiales: {e}")
        log_pred, log_tips = pd.DataFrame(), pd.DataFrame()

    # Una sola barra de filtros para TODAS las subpestañas (antes ligas, equipos y
    # mercados se repetían dentro de varias subpestañas, cada una con su propia
    # selección). Dentro de cada subpestaña solo queda lo que es propio de ella.
    tipo = st.radio("Tipo de predicción", reports.TIPOS, horizontal=True, key="rep_tipo",
                    help="Modelo = todas las predicciones del modelo. Conservador/Equilibrado/Agresivo = picks de "
                         "los planes del día de Tips; Combinadas del día y En vivo = sus boletos. Una sola opción "
                         "a la vez: un mismo pick puede ser del modelo y de un boleto (se contaría dos veces).")
    f1, f1b, f4 = st.columns([1, 1, 1])
    incluir_bt = f4.toggle("Incluir backtests", value=True, key="rep_backtest",
                           help="Partidos ya jugados predichos después, solo con datos anteriores a cada uno "
                                "(sin trampa). Suman muestra; apágalo para ver solo lo predicho en su momento.")
    largo = reports.predictions_long(log_pred, include_backtest=incluir_bt)
    largo_tips = reports.tips_legs_long(log_tips)
    todo = pd.concat([largo, largo_tips], ignore_index=True) if not largo_tips.empty else largo

    if todo.empty:
        st.info("Todavía no hay predicciones resueltas para reportar.")
    else:
        fechas = pd.to_datetime(todo["fecha"]).dt.date
        hoy_rep = datetime.now(BOGOTA_TZ).date()
        inicio_def = max(fechas.min(), hoy_rep - pd.Timedelta(days=90))
        # Inicio y fin por separado (pedido del usuario); el fin nunca pasa de hoy.
        desde = f1.date_input("Fecha inicio", value=inicio_def, min_value=fechas.min(), max_value=hoy_rep,
                              key="rep_desde")
        hasta = f1b.date_input("Fecha fin", value=hoy_rep, min_value=fechas.min(), max_value=hoy_rep,
                               key="rep_hasta")
        if hasta < desde:
            st.warning("La fecha fin es anterior a la de inicio: se intercambian.")
            desde, hasta = hasta, desde
        datos = todo[(fechas >= desde) & (fechas <= hasta) & (todo["tipo"] == tipo)]

        g_liga, g_eq, g_merc = st.columns(3)
        ligas = _league_multiselect(g_liga, "Ligas", sorted(todo["competition"].dropna().unique()), "rep_ligas")
        datos = datos[datos["competition"].isin(ligas)]
        # Equipos: solo los de las ligas elegidas; vacío = todos.
        equipos_opc = sorted(set(datos["home_team"].dropna()) | set(datos["away_team"].dropna()))
        if "rep_equipos" in st.session_state:
            st.session_state["rep_equipos"] = [e for e in st.session_state["rep_equipos"] if e in equipos_opc]
        equipos = g_eq.multiselect("Equipos", equipos_opc, key="rep_equipos", placeholder="Todos")
        if equipos:
            datos = datos[datos["home_team"].isin(equipos) | datos["away_team"].isin(equipos)]
        mercados_disp = sorted(todo["mercado"].dropna().unique())
        if "rep_mercados" in st.session_state:  # mercados nuevos entran solos, como las ligas
            vistos = set(st.session_state.get("rep_mercados__vistos", []))
            st.session_state["rep_mercados"] = ([m for m in st.session_state["rep_mercados"] if m in mercados_disp]
                                                + [m for m in mercados_disp if m not in vistos])
        st.session_state["rep_mercados__vistos"] = mercados_disp
        mercados = g_merc.multiselect("Mercados", mercados_disp, key="rep_mercados",
                                      default=None if "rep_mercados" in st.session_state else mercados_disp)
        datos = datos[datos["mercado"].isin(mercados)]
        modelo = datos

        boletos = reports.tips_tickets(log_tips)
        if not boletos.empty:
            fb = pd.to_datetime(boletos["fecha"]).dt.date
            boletos = boletos[(fb >= desde) & (fb <= hasta)]
            if tipo != reports.TIPO_MODELO:
                boletos = boletos[boletos["plan"].map(reports.plan_tipo) == tipo]
            boletos = boletos[[any(l in ligas for l in str(ls).split(", ")) for ls in boletos["ligas"]]]
            if equipos:
                boletos = boletos[boletos["picks"].map(lambda t: any(e in t for e in equipos))]
        cerrados = boletos[boletos["resultado"].isin(["ganada", "perdida"])] if not boletos.empty else boletos

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Predicciones resueltas" if tipo == reports.TIPO_MODELO else f"Picks resueltos · {tipo}",
                  f"{len(modelo):,}")
        k2.metric("Acierto", f"{modelo['acierto'].mean():.0%}" if len(modelo) else "-",
                  f"esperado {modelo['esperado'].mean():.0%}" if len(modelo) else None, delta_color="off")
        k3.metric("Boletos de Tips resueltos", f"{len(cerrados)}",
                  f"{(cerrados['resultado'] == 'ganada').mean():.0%} ganados" if len(cerrados) else None,
                  delta_color="off")
        ganancia_tips = pd.to_numeric(cerrados["ganancia"], errors="coerce").sum() if len(cerrados) else 0
        k4.metric("Ganancia neta de Tips", f"${ganancia_tips:+,.0f}")

        t_partidos, t_dia, t_liga, t_equipo, t_mercado, t_fd, t_tips = st.tabs(
            ["Partidos del día", "Por día", "Por liga", "Por equipo", "Por mercado", "Fortalezas y debilidades",
             "Boletos de Tips"])

        with t_partidos:
            _render_day_matches(log_pred, log_tips, ligas, equipos, mercados, incluir_bt, hasta, tipo)

        with t_dia:
            tabla = reports.summarize(datos, ["fecha"])
            if tabla.empty:
                st.caption("Sin datos en el filtro.")
            else:
                tabla = tabla.sort_values("fecha")
                grafico = tabla.assign(fecha=pd.to_datetime(tabla["fecha"]))
                grafico["etiqueta_x"] = grafico["fecha"].dt.strftime("%Y-%m-%d")  # texto que ordena bien
                st.altair_chart(_bars_vs_expected(grafico, "fecha", "Día", horizontal=False), use_container_width=True)
                st.caption("Barras azules: % de acierto real · marca naranja: % esperado por el modelo.")
                _summary_table(tabla.sort_values("fecha", ascending=False), "fecha", "Día")
                dias_det = sorted(tabla["fecha"].unique(), reverse=True)
                dia_det = st.selectbox("Ver el detalle de predicciones del día", dias_det, key="rep_dia_detalle")
                _render_detail(datos[pd.to_datetime(datos["fecha"]).dt.date == dia_det], PICK_NOTE)

        with t_liga:
            tabla = reports.summarize(datos, ["competition"])
            if not tabla.empty:
                tabla["competition"] = tabla["competition"].map(lambda c: config.COMPETITIONS.get(c, c))
                st.altair_chart(_bars_vs_expected(tabla, "competition", "Liga"), use_container_width=True)
                _summary_table(tabla.sort_values("acierto_pct", ascending=False), "competition", "Liga")
                ligas_det = sorted(datos["competition"].dropna().unique())
                liga_det = st.selectbox("Ver el detalle de predicciones de la liga", ligas_det, key="rep_liga_detalle",
                                        format_func=lambda c: config.COMPETITIONS.get(c, c))
                _render_detail(datos[datos["competition"] == liga_det], PICK_NOTE)

        with t_equipo:
            tabla = reports.summarize(reports.by_team(datos), ["equipo"])
            if equipos and not tabla.empty:
                tabla = tabla[tabla["equipo"].isin(equipos)]  # by_team suma también a los rivales
            # Sin equipos elegidos arriba, el mínimo arranca en el máximo posible
            # (primero los de más muestra). Con equipos elegidos no hay mínimo: se
            # ven todos los elegidos (un slider guardado ocultaba alguno al cambiar
            # el mercado).
            min_n = 1 if equipos else _min_sample_slider(
                "Mínimo de predicciones por equipo",
                tabla["predicciones"] if not tabla.empty else pd.Series(dtype=int), 10 ** 6, "rep_min_equipo")
            tabla = tabla[tabla["predicciones"] >= min_n] if not tabla.empty else tabla
            if tabla.empty:
                st.caption("Ningún equipo llega a ese mínimo en el filtro.")
            else:
                tabla = tabla.sort_values("diferencia", ascending=False)
                # Un solo gráfico divergente; con muchos equipos, los MAX_TEAMS_CHART
                # de diferencia más marcada.
                grafico = tabla.reindex(tabla["diferencia"].abs().sort_values(ascending=False).index)
                grafico = grafico.head(MAX_TEAMS_CHART).sort_values("diferencia", ascending=False)
                st.markdown("**Acierto real − esperado por equipo** (puntos porcentuales)")
                st.altair_chart(_diverging_bars(grafico, "equipo", "Equipo"), use_container_width=True)
                st.caption("Azul a la derecha: con ese equipo el modelo acierta más de lo que esperaba (fortaleza). "
                           "Rojo a la izquierda: acierta menos (debilidad)."
                           + (f" Se muestran los {MAX_TEAMS_CHART} con la diferencia más marcada de {len(tabla)}; "
                              "la tabla de abajo tiene todos." if len(tabla) > MAX_TEAMS_CHART else ""))
                _summary_table(tabla, "equipo", "Equipo")
                st.caption(PICK_NOTE + " Cada predicción cuenta para los dos equipos del partido.")
                opciones = (([e for e in equipos if e in set(tabla["equipo"])]) if equipos else
                            grafico["equipo"].tolist() + [e for e in tabla["equipo"] if e not in set(grafico["equipo"])])
                elegido = st.selectbox("Ver el detalle de predicciones de", opciones, key="rep_equipo_detalle")
                _render_detail(datos[(datos["home_team"] == elegido) | (datos["away_team"] == elegido)])

        with t_mercado:
            tabla = reports.summarize(datos, ["mercado"])
            if not tabla.empty:
                st.altair_chart(_bars_vs_expected(tabla, "mercado", "Mercado"), use_container_width=True)
                _summary_table(tabla.sort_values("acierto_pct", ascending=False), "mercado", "Mercado")
                st.caption(PICK_NOTE)
                merc_det = st.selectbox("Ver el detalle de predicciones del mercado",
                                        tabla.sort_values("acierto_pct", ascending=False)["mercado"].tolist(),
                                        key="rep_mercado_detalle")
                _render_detail(datos[datos["mercado"] == merc_det])

        with t_fd:
            _render_strengths_map(datos)

        with t_tips:
            _render_tips_report(boletos, cerrados, mostrar_planes=tipo == reports.TIPO_MODELO)
