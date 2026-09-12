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
from src.fetch_football_data import FootballDataClient, fetch_competition
from src.matchday_predictions import matchday_predictions_df

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
season = st.sidebar.number_input("Temporada (año de inicio)", value=2026, step=1)
matchday = st.sidebar.number_input("Jornada", value=1, min_value=1, step=1)

if st.sidebar.button("🔄 Forzar actualización de datos"):
    load_competition.clear()
    get_predictions.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.caption(
    "Los equipos marcados de baja confianza tienen muy pocos partidos propios en "
    "los datos cargados (casi siempre recién ascendidos): el modelo se apoya "
    "sobre todo en el promedio de liga para ellos."
)

# --- Header ---
st.title(f"{config.COMPETITIONS[comp_code]} · Jornada {matchday}")

try:
    df = get_predictions(comp_code, int(season), int(matchday))
except Exception as e:
    st.error(f"No se pudo calcular la jornada: {e}")
    st.stop()

if df.empty:
    st.warning("No hay partidos programados para esa combinación de competición/temporada/jornada.")
    st.stop()

# --- Métricas resumen ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Partidos", len(df))
c2.metric("Over 2.5 promedio", f"{df['over_2_5'].mean():.0f}%")
c3.metric("BTTS promedio", f"{df['btts'].mean():.0f}%")
c4.metric("Con baja confianza", int(df["low_confidence"].sum()))

st.divider()

# --- Gráfico 1X2 ---
melt = df.melt(
    id_vars=["home_team", "away_team"],
    value_vars=["home_win", "draw", "away_win"],
    var_name="resultado", value_name="probabilidad",
)
melt["partido"] = melt["home_team"] + " vs " + melt["away_team"]
orden_map = {"home_win": (0, "Local"), "draw": (1, "Empate"), "away_win": (2, "Visitante")}
melt["orden"] = melt["resultado"].map(lambda k: orden_map[k][0])
melt["resultado"] = melt["resultado"].map(lambda k: orden_map[k][1])

chart = alt.Chart(melt).mark_bar().encode(
    x=alt.X("probabilidad:Q", stack="zero", axis=alt.Axis(format=".0f", title="Probabilidad (%)")),
    y=alt.Y("partido:N", sort=None, title=None),
    color=alt.Color(
        "resultado:N",
        scale=alt.Scale(domain=["Local", "Empate", "Visitante"], range=["#2a78d6", "#eb6834", "#1baf7a"]),
        legend=alt.Legend(title=None, orient="top"),
    ),
    order=alt.Order("orden:Q"),
    tooltip=["partido", "resultado", alt.Tooltip("probabilidad:Q", format=".1f")],
).properties(height=28 * len(df) + 40)

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
    use_container_width=True, hide_index=True,
)

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
    st.metric("Acierto histórico (todas las competiciones)", f"{overall:.0%}", f"{int(resolved['hit'].sum())}/{len(resolved)}")

    by_comp = resolved.groupby("competition")["hit"].agg(aciertos="sum", total="count").reset_index()
    by_comp["tasa"] = by_comp["aciertos"] / by_comp["total"] * 100
    by_comp["competencia"] = by_comp["competition"].map(config.COMPETITIONS).fillna(by_comp["competition"])

    bar = alt.Chart(by_comp).mark_bar(color="#2a78d6").encode(
        x=alt.X("competencia:N", title=None, sort="-y"),
        y=alt.Y("tasa:Q", title="% de acierto", scale=alt.Scale(domain=[0, 100])),
        tooltip=["competencia", "aciertos", "total", alt.Tooltip("tasa:Q", format=".0f")],
    ).properties(height=280)
    st.altair_chart(bar, use_container_width=True)
