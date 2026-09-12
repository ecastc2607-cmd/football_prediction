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

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from src.competition_status import resolve_current_season_and_matchday
from src.fetch_football_data import FootballDataClient, fetch_competition
from src.highlightly_client import HIGHLIGHTLY_API_KEY, get_match_stats
from src.live_matches import get_live_matches
from src.matchday_predictions import matchday_predictions_df
from src.parlay_builder import build_parlays
from src.predict_matchday import finished_fixtures

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

# Misma lógica para la key de Highlightly (corners/faltas/tarjetas) — es opcional,
# así que si falta simplemente se desactiva ese bloque más abajo, sin error.
HIGHLIGHTLY_KEY = os.getenv("HIGHLIGHTLY_API_KEY") or HIGHLIGHTLY_API_KEY
if not HIGHLIGHTLY_KEY and "HIGHLIGHTLY_API_KEY" in st.secrets:
    HIGHLIGHTLY_KEY = st.secrets["HIGHLIGHTLY_API_KEY"]


@st.cache_data(ttl=1800, show_spinner="Descargando datos de football-data.org...")
def load_competition(code: str, season: int) -> pd.DataFrame:
    client = FootballDataClient(api_key=API_KEY)
    return fetch_competition(client, code, season)


@st.cache_data(ttl=1800, show_spinner="Calculando predicciones...")
def get_predictions(code: str, season: int, matchday: int) -> pd.DataFrame:
    # Asegura que los datos de esta y la temporada anterior estén descargados.
    load_competition(code, season)
    load_competition(code, season - 1)
    return matchday_predictions_df(code, season, matchday, seasons_back=1)


@st.cache_data(ttl=1800, show_spinner="Detectando jornada actual...")
def auto_status(code: str):
    client = FootballDataClient(api_key=API_KEY)
    return resolve_current_season_and_matchday(client, code)


@st.cache_data(ttl=180, show_spinner="Buscando partidos en vivo...")
def load_live_matches(codes: tuple[str, ...]) -> pd.DataFrame:
    client = FootballDataClient(api_key=API_KEY)
    # Reutiliza el caché de 30 min de load_competition en vez de re-descargar
    # la liga si la vista principal ya la trajo hace un momento.
    return get_live_matches(client, list(codes), ensure_data=load_competition)


@st.cache_data(ttl=3600, show_spinner=False)
def load_match_stats(home: str, away: str, date_iso: str, competition_code: str):
    return get_match_stats(home, away, date_iso, competition_code, api_key=HIGHLIGHTLY_KEY)


@st.cache_data(ttl=1800)
def load_calibration_log() -> pd.DataFrame:
    path = config.ROOT_DIR / "data" / "tracking" / "predictions_log.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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
    status = auto_status(comp_code)
    st.session_state["season_input"] = status.season
    st.session_state["matchday_input"] = status.matchday
    st.session_state["_status_reason"] = status.reason
    st.session_state["_last_comp"] = comp_code

season = st.sidebar.number_input("Temporada (año de inicio)", step=1, key="season_input")
matchday = st.sidebar.number_input("Jornada", min_value=1, step=1, key="matchday_input")
st.sidebar.caption(f"🎯 {st.session_state.get('_status_reason', '')}")

if st.sidebar.button("🔄 Forzar actualización de datos"):
    load_competition.clear()
    get_predictions.clear()
    auto_status.clear()
    load_live_matches.clear()
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
            "predicciones que mostrar aquí — pero sí puedes ver corners/faltas/tarjetas de sus "
            "partidos más abajo."
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
        show = df.copy()
        show["Partido"] = show["home_team"] + " vs " + show["away_team"]
        show["1X2"] = show.apply(lambda r: f"{r.home_win:.0f}% / {r.draw:.0f}% / {r.away_win:.0f}%", axis=1)
        show["xG"] = show.apply(lambda r: f"{r.home_xg} - {r.away_xg}", axis=1)
        show["⚠"] = show["low_confidence"].map({True: "Baja confianza", False: ""})
        st.dataframe(
            show[["Partido", "xG", "1X2", "over_2_5", "btts", "top_score", "⚠"]].rename(columns={
                "over_2_5": "Over 2.5 %", "btts": "BTTS %", "top_score": "Marcador top",
            }),
            width="stretch", hide_index=True,
        )

    # --- Corners, faltas y tarjetas (partidos ya jugados de la jornada) ---
    st.divider()
    st.subheader("📐 Corners, faltas y tarjetas")
    if not HIGHLIGHTLY_KEY:
        st.caption(
            "Falta HIGHLIGHTLY_API_KEY en el .env — regístrate gratis (sin tarjeta) en "
            "highlightly.net para activar este bloque."
        )
    else:
        played = finished_fixtures(comp_code, int(season), int(matchday))
        if played.empty:
            st.caption("Ningún partido de esta jornada ha terminado todavía.")
        else:
            for _, row in played.iterrows():
                label = f"{row['home_team']} {int(row['home_goals'])}-{int(row['away_goals'])} {row['away_team']}"
                with st.expander(label):
                    stats = load_match_stats(row["home_team"], row["away_team"], row["utc_date"], comp_code)
                    if not stats:
                        st.caption("Sin estadísticas disponibles para este partido en Highlightly.")
                    else:
                        stat_df = pd.DataFrame(stats).T[
                            ["corners", "faltas", "tarjetas_amarillas", "tarjetas_rojas"]
                        ]
                        stat_df.columns = ["Corners", "Faltas", "T. amarillas", "T. rojas"]
                        st.dataframe(stat_df, width="stretch")

    # --- Combinadas sugeridas ---
    st.divider()
    st.subheader("🎯 Combinadas sugeridas")
    st.caption(
        "Cuota propia del modelo (1/probabilidad, sin margen de casa de apuestas), combinando el "
        "mejor pick 1X2 de cada partido. Solo se muestran combinadas entre 6x y 30x, máximo 6 "
        "partidos. El riesgo se calcula por la probabilidad combinada real, no por cantidad de "
        "selecciones — esto es estadística sobre datos históricos, no una garantía de resultado."
    )
    parlays = build_parlays(df)
    if not parlays:
        st.caption("No se armó ninguna combinada en el rango de cuota 6x-30x con esta jornada.")
    else:
        risk_color = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}
        for risk in ("Bajo", "Medio", "Alto"):
            tier = [p for p in parlays if p.risk == risk]
            if not tier:
                continue
            st.markdown(f"**{risk_color[risk]} Riesgo {risk}**")
            for p in tier:
                st.write(f"`{p.combined_odds}x` · prob. combinada {p.combined_probability:.1%} — {p.describe()}")

    # --- Calibración histórica ---
    st.divider()
    st.subheader("📊 Calibración del proyecto")
    log = load_calibration_log()
    resolved = log[log["status"] == "resolved"] if not log.empty else pd.DataFrame()

    if resolved.empty:
        st.caption("Todavía no hay predicciones resueltas en el log.")
    else:
        resolved = resolved.copy()
        resolved["hit"] = resolved["hit"].astype(str).str.lower().isin(["true", "1"])
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

    live_df = load_live_matches(tuple(config.COMPETITIONS.keys()))

    if live_df.empty:
        st.info("No hay partidos en juego ahora mismo en las 6 competiciones.")
    else:
        live_show = live_df.copy()
        live_show["Partido"] = (
            live_show["home_team"] + " " + live_show["home_goals"].astype(str) + " - "
            + live_show["away_goals"].astype(str) + " " + live_show["away_team"]
        )
        live_show["1X2 en vivo"] = live_show.apply(
            lambda r: f"{r.live_home_win:.0f}% / {r.live_draw:.0f}% / {r.live_away_win:.0f}%"
            if pd.notna(r.live_home_win) else "sin datos suficientes",
            axis=1,
        )
        st.dataframe(
            live_show[["competition_name", "Partido", "minuto_estimado", "1X2 en vivo"]].rename(columns={
                "competition_name": "Liga", "minuto_estimado": "Minuto (est.)",
            }),
            width="stretch", hide_index=True,
        )
