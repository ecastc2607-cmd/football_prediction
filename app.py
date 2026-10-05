"""
Dashboard de Streamlit: conecta en vivo con el pipeline (src/) para mostrar
predicciones reales de la jornada y la calibración histórica del proyecto.

Correr local:
    streamlit run app.py

Desplegado en Streamlit Community Cloud: configura FOOTBALL_DATA_API_KEY en
"Secrets" (no se sube el .env al repo).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import streamlit as st

from src import config, national_teams, odds_api
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
from src.timezones import BOGOTA_TZ, format_bogota, to_bogota
from src.value_tips import MIN_MODEL_PROB, candidate_picks, day_summary, evaluate

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
    client = FootballDataClient(api_key=API_KEY)
    return fetch_competition(client, code, season)


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


def _prepare_competition(code: str, season: int) -> None:
    # Asegura que los datos de esta y la temporada anterior estén descargados.
    load_competition(code, season)
    load_competition(code, season - 1)
    if code in config.CUP_STYLE_COMPETITIONS:
        load_support_leagues(season)


@st.cache_data(ttl=1800, show_spinner="Calculando predicciones...")
def get_predictions(code: str, season: int, matchday: int) -> pd.DataFrame:
    _prepare_competition(code, season)
    return matchday_predictions_df(code, season, matchday, seasons_back=1)


@st.cache_data(ttl=300, show_spinner="Calculando predicciones...")
def get_predictions_con_jugados(code: str, season: int, matchday: int) -> pd.DataFrame:
    """Igual que get_predictions, pero también incluye los partidos ya en vivo
    o terminados de la jornada (get_predictions se queda solo con lo pendiente,
    a propósito, porque otros consumidores como log_predictions.py no deben
    mezclar partidos ya jugados). Usada por "Detalle por partido" para no
    hacer desaparecer un partido de la tabla apenas arranca o termina — TTL
    más corto porque el estado en vivo cambia rápido."""
    _prepare_competition(code, season)
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


@st.cache_data(ttl=3600, show_spinner="Consultando cuotas de las casas...")
def load_odds_table(code: str, desde_utc: str, hasta_utc: str) -> tuple:
    """(eventos, créditos_restantes, error). Pasa por el historial versionado
    (odds_api.get_odds_table): una ventana consultada hace menos de
    SNAPSHOT_MAX_AGE_HOURS no gasta créditos, ni siquiera tras un reinicio."""
    return odds_api.get_odds_table(code, desde_utc, hasta_utc, api_key=ODDS_KEY)


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


def parlays_with_real_odds(df: pd.DataFrame, only_colombia: bool = True) -> tuple[list, list[str]]:
    """Arma las combinadas, y solo para las competiciones que aparecen en ellas
    trae cuotas reales (2 créditos por competición, o 0 si están en el
    historial reciente) y las vuelve a armar ya con esas cuotas.
    Devuelve (combinadas, avisos)."""
    calibracion = load_calibrator()
    base = build_parlays(df, calibracion)
    if not ODDS_KEY or not base:
        return base, ([] if ODDS_KEY else ["Sin ODDS_API_KEY: cuotas estimadas, no reales."])
    comp_por_partido = {(r.home_team, r.away_team): r.competition for r in df.itertuples()}
    comps = {comp_por_partido.get((l.home_team, l.away_team)) for p in base for l in p.legs}
    tablas, avisos, restante = {}, [], None
    for code in sorted(c for c in comps if c in odds_api.SPORT_KEYS):
        desde, hasta = odds_window(df[df["competition"] == code])
        eventos, rest, error = load_odds_table(code, desde, hasta)
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
    return Calibrator.from_log(load_calibration_log())


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

comp_code = st.sidebar.selectbox(
    "Competición",
    options=list(config.COMPETITIONS.keys()),
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

tab_jornada, tab_vivo, tab_amistosos, tab_tips = st.tabs(
    ["📅 Jornada", "🔴 En vivo", "🌍 Amistosos", "💡 Tips"]
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
                           "over_2_5", "btts", "top_score"]
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
    # Cuotas reales solo a pedido: cada liga consultada gasta ~2 de los 500
    # créditos/mes (0 si se consultó hace poco: queda en el historial).
    clave_cuotas = f"cuotas_reales_{comp_code}_{int(season)}_{int(matchday)}"
    df_combinadas = get_predictions(comp_code, int(season), int(matchday))
    if (ODDS_KEY and comp_code in odds_api.SPORT_KEYS and not df_combinadas.empty
            and not st.session_state.get(clave_cuotas)):
        if st.button("💱 Traer cuotas reales para estas combinadas (~2 créditos)", key=f"btn_{clave_cuotas}"):
            st.session_state[clave_cuotas] = True
    if st.session_state.get(clave_cuotas) and not df_combinadas.empty:
        combinadas, avisos_cuotas = parlays_with_real_odds(df_combinadas)
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

    if st.button("🔄 Actualizar en vivo"):
        load_live_matches.clear()
        st.rerun()

    live_df = load_live_matches(tuple(config.COMPETITIONS.keys()))

    if live_df.empty:
        st.info(f"No hay partidos en juego ahora mismo en las {len(config.COMPETITIONS)} competiciones.")
    else:
        for _, row in live_df.iterrows():
            live_1x2 = (
                f"{row.live_home_win:.0f}% / {row.live_draw:.0f}% / {row.live_away_win:.0f}%"
                if pd.notna(row.live_home_win) else "sin datos suficientes"
            )
            # Las mismas estadísticas del desplegable de abajo (caché de 5 min:
            # pedirlas acá no gasta una consulta extra).
            stats_vivo = None
            if STATS_KEY_PRESENT:
                try:
                    stats_vivo = load_match_stats(row.home_team, row.away_team, row.utc_date, row.competition)
                except Exception:
                    stats_vivo = None
            ajuste = ancla = None
            if pd.notna(row.get("pre_home_xg")) and pd.notna(row.get("pre_away_xg")):
                # Base pre-partido anclada al mercado si hay cuotas previas
                # guardadas (sin gastar créditos); si no, la del modelo.
                base_l, base_v, ancla = anchored_prematch_xg(
                    float(row.pre_home_xg), float(row.pre_away_xg), row.competition,
                    row.home_team, row.away_team, row.utc_date,
                )
                stats_l = (stats_vivo or {}).get(row.home_team)
                stats_v = (stats_vivo or {}).get(row.away_team)
                ajuste = adjusted_live_probabilities(
                    int(row.home_goals), int(row.away_goals), int(row.minuto_estimado),
                    base_l, base_v, stats_l, stats_v,
                )
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

            c_liga, c_local, c_marcador, c_visitante, c_1x2 = st.columns([1.3, 2, 1, 2, 1.6])
            c_liga.caption(f"{row.competition_name} · {format_bogota(row.utc_date)} · min. {row.minuto_estimado}'")
            c_local.write(row.home_team)
            c_marcador.markdown(f"**{row.home_goals} - {row.away_goals}**")
            c_visitante.write(row.away_team)
            if ajuste:
                c_1x2.markdown(f"**1X2: {ajuste['home_win']:.0%} / {ajuste['draw']:.0%} / {ajuste['away_win']:.0%}**")
                c_1x2.caption(f"(solo pre-partido: {live_1x2})")
                detalle = []
                if ajuste["home_xg_observed"] is not None and ajuste["away_xg_observed"] is not None:
                    s_l, s_v = stats_vivo[row.home_team], stats_vivo[row.away_team]
                    n = lambda v: f"{float(v):.0f}"  # noqa: E731 — 10.0 -> "10"
                    detalle.append(
                        f"remates {n(s_l.get('remates_totales'))} ({n(s_l.get('remates_a_puerta'))} a puerta) – "
                        f"{n(s_v.get('remates_totales'))} ({n(s_v.get('remates_a_puerta'))}) → xG aprox. "
                        f"{ajuste['home_xg_observed']:.2f} – {ajuste['away_xg_observed']:.2f}"
                    )
                else:
                    detalle.append("sin estadísticas de remates todavía: solo marcador, efecto marcador y añadido")
                if ancla:
                    detalle.append(
                        f"base anclada al mercado (xG cuotas previas {ancla['market_home_xg']:.2f} – "
                        f"{ancla['market_away_xg']:.2f}, modelo {row.pre_home_xg:.2f} – {row.pre_away_xg:.2f})"
                    )
                detalle.append(
                    f"goles esperados en lo que falta (~{ajuste['remaining_minutes']:.0f}'): "
                    f"{ajuste['home_xg_remaining']:.2f} – {ajuste['away_xg_remaining']:.2f}"
                )
                detalle.append(f"no pierde {row.home_team}: {ajuste['home_win'] + ajuste['draw']:.0%} · "
                               f"no pierde {row.away_team}: {ajuste['away_win'] + ajuste['draw']:.0%}")
                st.caption(" · ".join(detalle))
            else:
                c_1x2.caption(f"1X2: {live_1x2}")

            if STATS_KEY_PRESENT:
                with st.expander("📐 Corners, faltas y tarjetas"):
                    # jornada/temporada quedan en 0: no siempre se conocen con certeza
                    # cuando se mezclan todas las competiciones a la vez en esta pestaña.
                    render_stats_block(row.home_team, row.away_team, row.utc_date, row.competition)
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

    if st.button("🔄 Actualizar amistosos"):
        load_friendlies.clear()
        get_friendlies_predictions.clear()
        st.rerun()

    try:
        fixtures_amistosos = load_friendlies()
        pred_amistosos = get_friendlies_predictions()
    except Exception as e:
        st.error(f"No se pudieron cargar los amistosos: {e}")
        fixtures_amistosos, pred_amistosos = pd.DataFrame(), pd.DataFrame()

    if fixtures_amistosos.empty:
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

    col_banca, col_fecha, col_casas = st.columns([1, 1, 1.3])
    banca = col_banca.number_input("Banca (COP)", min_value=10_000, value=50_000, step=10_000, key="tips_banca")
    hoy_col = datetime.now(BOGOTA_TZ).date()
    fecha_tips = col_fecha.date_input("Día", value=hoy_col, key="tips_fecha")
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
        st.caption(f"{len(dia)} partidos por jugar ese día · {len(candidatos)} picks con ≥{MIN_MODEL_PROB:.0%} "
                   "en partidos sin baja confianza.")

        if dia.empty:
            st.info("No hay partidos por jugar ese día en las competiciones que sigue el proyecto.")
        elif candidatos.empty:
            st.warning(
                "🚫 **Hoy no hay opciones de muy bajo riesgo.** Ningún pick llega al "
                f"{MIN_MODEL_PROB:.0%} en partidos con datos suficientes. Recomendación: no apostar — "
                "este día no cuenta para la meta."
            )
        else:
            # --- Mejor cuota automática (The Odds API) ---
            candidatos["cuota_auto"] = None
            candidatos["casa"] = ""
            if not ODDS_KEY:
                st.info(
                    "Sin cuotas automáticas: falta `ODDS_API_KEY` (gratis en the-odds-api.com). "
                    "Mientras tanto, escribe la cuota de tu casa en la columna 'Cuota manual'."
                )
            else:
                inicio_local = datetime.combine(fecha_tips, datetime.min.time(), tzinfo=BOGOTA_TZ)
                desde = inicio_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                hasta = (inicio_local + pd.Timedelta(days=1)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                restante = None
                for code in candidatos["competition"].unique():
                    if code not in odds_api.SPORT_KEYS:
                        continue  # amistosos: sin clave verificada en The Odds API -> cuota manual
                    eventos, rest, error = load_odds_table(code, desde, hasta)
                    restante = rest if rest is not None else restante
                    if error:
                        st.caption(f"⚠ {config.COMPETITIONS.get(code, code)}: {error}")
                        continue
                    for idx, r in candidatos[candidatos["competition"] == code].iterrows():
                        evento = odds_api.find_event(
                            eventos, r.home_team, r.away_team, r.utc_date,
                            r.home_team_short, r.away_team_short,
                        )
                        if evento is None:
                            continue
                        precio = odds_api.row_price(evento, r.odds_key, only_colombia=solo_colombia)
                        if precio:
                            candidatos.at[idx, "cuota_auto"] = precio[0]
                            candidatos.at[idx, "casa"] = precio[1]
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
                st.subheader(f"✅ {resumen['n']} apuesta(s) sugerida(s)")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total a apostar", f"${resumen['apostado']:,.0f}",
                          f"{resumen['apostado'] / banca:.0%} de la banca", delta_color="off")
                m2.metric("Ganancia esperada", f"${resumen['esperado']:,.0f}")
                m3.metric("Si todo acierta", f"${resumen['si_todo_acierta']:,.0f}",
                          f"prob. {resumen['prob_todo_acierta']:.0%}", delta_color="off")
                m4.metric("Si todo falla", f"${resumen['si_todo_falla']:,.0f}")

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
                        "Ganancia si acierta": apuestas["ganancia"],
                    }),
                    hide_index=True, width="stretch",
                    column_config={
                        "Partido": st.column_config.Column(pinned=True),
                        "Valor": st.column_config.NumberColumn(format="+%.1f%%"),
                        "Apostar": st.column_config.NumberColumn(format="$%d"),
                        "Ganancia si acierta": st.column_config.NumberColumn(format="$%d"),
                    },
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
