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

from src import config, europa_league
from src.competition_status import resolve_current_season_and_matchday
from src.fetch_football_data import FootballDataClient, fetch_competition
from src.goal_api_client import (
    GOAL_API_KEY,
    GoalApiClient,
    GoalApiRateLimited,
    get_match_stats as goal_get_match_stats,
)
from src.highlightly_client import (
    HIGHLIGHTLY_API_KEY,
    HighlightlyRateLimited,
    get_match_stats as highlightly_get_match_stats,
)
from src.live_matches import get_live_matches
from src.match_stats_log import get_cached_stats, log_match_stats
from src.match_context import get_standings_map, rivalry_label
from src.match_tendencies import load_team_averages, pick_tendencies
from src.matchday_predictions import matchday_predictions_df
from src.parlay_builder import build_parlays
from src.predict_matchday import LIVE_STATUSES, finished_fixtures
from src.prediction_log import MARKETS
from src.timezones import format_bogota

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


@st.cache_data(ttl=1800, show_spinner="Descargando datos de la competición...")
def load_competition(code: str, season: int) -> pd.DataFrame:
    # La Europa League no está en el plan gratis de football-data.org (verificado
    # contra su API y su tabla de cobertura), así que corre sobre Goal API en vez
    # de football-data.org — ver src/europa_league.py. Si esto falla, la excepción
    # sube tal cual y la queda contenida por el propio try/except de la pestaña
    # Jornada (por competición, no afecta a las otras 6).
    if code in config.GOAL_API_COMPETITIONS:
        return europa_league.fetch_competition(season)
    client = FootballDataClient(api_key=API_KEY)
    return fetch_competition(client, code, season)


@st.cache_data(ttl=1800, show_spinner="Calculando predicciones...")
def get_predictions(code: str, season: int, matchday: int) -> pd.DataFrame:
    # Asegura que los datos de esta y la temporada anterior estén descargados.
    load_competition(code, season)
    load_competition(code, season - 1)
    return matchday_predictions_df(code, season, matchday, seasons_back=1)


@st.cache_data(ttl=300, show_spinner="Calculando predicciones...")
def get_predictions_con_jugados(code: str, season: int, matchday: int) -> pd.DataFrame:
    """Igual que get_predictions, pero también incluye los partidos ya en vivo
    o terminados de la jornada (get_predictions se queda solo con lo pendiente,
    a propósito, porque otros consumidores como log_predictions.py no deben
    mezclar partidos ya jugados). Usada por "Detalle por partido" para no
    hacer desaparecer un partido de la tabla apenas arranca o termina — TTL
    más corto porque el estado en vivo cambia rápido."""
    load_competition(code, season)
    load_competition(code, season - 1)
    return matchday_predictions_df(code, season, matchday, seasons_back=1, include_played=True)


@st.cache_data(ttl=1800, show_spinner="Detectando jornada actual...")
def auto_status(code: str):
    if code in config.GOAL_API_COMPETITIONS:
        return europa_league.resolve_current_season_and_matchday()
    client = FootballDataClient(api_key=API_KEY)
    return resolve_current_season_and_matchday(client, code)


@st.cache_data(ttl=180, show_spinner="Buscando partidos en vivo...")
def load_live_matches(codes: tuple[str, ...]) -> pd.DataFrame:
    client = FootballDataClient(api_key=API_KEY)
    football_data_codes = [c for c in codes if c not in config.GOAL_API_COMPETITIONS]
    # Reutiliza el caché de 30 min de load_competition en vez de re-descargar
    # la liga si la vista principal ya la trajo hace un momento.
    live_df = get_live_matches(client, football_data_codes, ensure_data=load_competition)

    # La Europa League corre aparte, sobre Goal API (ver src/europa_league.py).
    # Aislado a propósito: si esto falla, se ignora y las otras 6 ligas siguen
    # mostrando su "en vivo" con total normalidad — nunca debe tumbar la pestaña.
    if any(c in config.GOAL_API_COMPETITIONS for c in codes):
        try:
            el_live = europa_league.get_live_matches(ensure_data=load_competition)
        except Exception:
            el_live = pd.DataFrame()
        if not el_live.empty:
            live_df = pd.concat([live_df, el_live], ignore_index=True)

    return live_df


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
            return europa_league.get_standings_map(season)
        except Exception:
            return {}  # la columna "Posiciones" ya sabe mostrar "-" si esto viene vacío
    client = FootballDataClient(api_key=API_KEY)
    return get_standings_map(client, code, season)


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
    return build_parlays(df) if not df.empty else []


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
    get_predictions.clear()
    get_predictions_con_jugados.clear()
    auto_status.clear()
    load_live_matches.clear()
    load_tendency_averages.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    "Los equipos marcados de baja confianza tienen muy pocos partidos propios en "
    "los datos cargados (casi siempre recién ascendidos): el modelo se apoya "
    "sobre todo en el promedio de liga para ellos."
)

tab_jornada, tab_vivo = st.tabs(["📅 Jornada", "🔴 En vivo"])

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
    st.caption(
        "Cuota propia del modelo (1/probabilidad, sin margen de casa de apuestas). Cada partido "
        "aporta como mucho un pick — 1X2, Más/Menos de 2.5 goles, o Ambos anotan, el que mejor "
        "calce — nunca dos mercados del mismo partido juntos (estarían correlacionados). Solo se "
        "muestran combinadas entre 6x y 30x, máximo 6 partidos. El riesgo se calcula por la "
        "probabilidad combinada real, no por cantidad de selecciones — esto es estadística sobre "
        "datos históricos, no una garantía de resultado."
    )
    parlays = load_parlays(comp_code, int(season), int(matchday))
    if not parlays:
        st.caption("No se armó ninguna combinada ni siquiera bajando la cuota mínima — muy pocos partidos disponibles.")
    else:
        if parlays[0].reduced_quota:
            st.warning(
                "⚠ Quedan pocos partidos por jugar en esta jornada y no alcanzan para llegar a 6x "
                "combinando lo disponible — estas combinadas se armaron con una cuota mínima más "
                "baja para no dejar la sección vacía."
            )
        risk_meta = {
            "Bajo": ("🟢", "Riesgo bajo"),
            "Medio": ("🟡", "Riesgo medio"),
            "Alto": ("🔴", "Riesgo alto"),
        }
        for risk in ("Bajo", "Medio", "Alto"):
            tier = [p for p in parlays if p.risk == risk]
            if not tier:
                continue
            emoji, titulo = risk_meta[risk]
            st.markdown(f"**{emoji} {titulo}** · {len(tier)} combinada(s)")
            # Cada combinada es su propio "cupón": un contenedor con borde con
            # la cuota/probabilidad total arriba, y abajo una fila por partido
            # (Partido | Mercado | Pick) — como un tiquete de apuesta, no una
            # sola celda de texto con todo junto.
            for p in tier:
                with st.container(border=True):
                    st.markdown(f"`{p.combined_odds}x` · prob. combinada **{p.combined_probability:.1%}**")
                    tabla = pd.DataFrame([
                        {"Partido": l.match, "Mercado": l.market, "Pick": l.pick}
                        for l in p.legs
                    ])
                    st.dataframe(tabla, width="stretch", hide_index=True)

    # --- Calibración histórica ---
    st.divider()
    st.subheader("📊 Calibración del proyecto")
    log = load_calibration_log()
    resolved = log[log["status"] == "resolved"] if not log.empty else pd.DataFrame()

    if resolved.empty:
        st.caption("Todavía no hay predicciones resueltas en el log.")
    else:
        resolved = resolved.copy()
        resolved["hit"] = _to_bool_series(resolved["hit"])
        overall = resolved["hit"].mean()
        st.metric("Acierto histórico (todas las competiciones)", f"{overall:.0%}",
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
        "Minuto estimado a partir de la hora de inicio (la API gratuita no da el minuto real ni "
        "tiempo añadido) — puede quedarse pegado si football-data.org tarda en marcar un partido "
        "como finalizado. 1X2 recalculado en vivo: marcador actual + goles esperados restantes "
        "del modelo, no una cuota de casa de apuestas."
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
            c_liga, c_local, c_marcador, c_visitante, c_1x2 = st.columns([1.3, 2, 1, 2, 1.6])
            c_liga.caption(f"{row.competition_name} · {format_bogota(row.utc_date)} · min. {row.minuto_estimado}'")
            c_local.write(row.home_team)
            c_marcador.markdown(f"**{row.home_goals} - {row.away_goals}**")
            c_visitante.write(row.away_team)
            c_1x2.caption(f"1X2: {live_1x2}")

            if STATS_KEY_PRESENT:
                with st.expander("📐 Corners, faltas y tarjetas"):
                    # jornada/temporada quedan en 0: no siempre se conocen con certeza
                    # cuando se mezclan todas las competiciones a la vez en esta pestaña.
                    render_stats_block(row.home_team, row.away_team, row.utc_date, row.competition)
            st.divider()
