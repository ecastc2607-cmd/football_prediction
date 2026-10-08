"""
Qué datos de partidos ya no pueden cambiar y por eso se leen SIEMPRE del
archivo local, sin volver a consultar la API (pedido del usuario: ahorrar
consultas).

  - Temporada cerrada (todos sus partidos terminados y el último hace más de
    CLOSED_GRACE_DAYS): se guarda versionada en git, en
    data/tracking/closed_seasons/, y no se vuelve a pedir nunca — ni siquiera
    tras un reinicio de Streamlit Cloud (data/processed/ no se versiona).
  - Jornada terminada por completo: mientras se mira esa jornada no hace falta
    refrescar la temporada en curso; nada de esa jornada puede cambiar.
"""
from __future__ import annotations

import pandas as pd

from . import config

CLOSED_SEASONS_DIR = config.CLOSED_SEASONS_DIR
FINAL_STATUSES = {"FINISHED", "CANCELLED", "AWARDED"}
# Un aplazado de una temporada vieja nunca se va a jugar en ESA temporada.
FINAL_IF_OLD = {"POSTPONED", "SUSPENDED"}
CLOSED_GRACE_DAYS = 3


def closed_path(code: str, season: int):
    return CLOSED_SEASONS_DIR / f"matches_{code}_{season}.csv"


def is_closed(df: pd.DataFrame) -> bool:
    if df.empty or "status" not in df:
        return False
    if not df["status"].isin(FINAL_STATUSES | FINAL_IF_OLD).all():
        return False
    ultimo = pd.to_datetime(df["utc_date"], utc=True, errors="coerce").max()
    return pd.notna(ultimo) and ultimo < pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=CLOSED_GRACE_DAYS)


def save_if_closed(df: pd.DataFrame, code: str, season: int) -> bool:
    """Guarda la temporada en el archivo versionado si ya está cerrada (y trae
    el marcador al descanso, para no congelar un esquema viejo)."""
    if not is_closed(df) or "home_ht_goals" not in df.columns:
        return False
    CLOSED_SEASONS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(closed_path(code, season), index=False, encoding="utf-8")
    return True


def jornada_finished_locally(code: str, season: int, matchday: int) -> bool:
    """True si el CSV local ya tiene TODOS los partidos de esa jornada
    terminados (y el esquema actual): no hace falta volver a pedirla."""
    path = config.PROCESSED_DIR / f"matches_{code}_{season}.csv"
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if "home_ht_goals" not in df.columns:
        return False
    jornada = df[df["matchday"] == matchday]
    return not jornada.empty and jornada["status"].isin(FINAL_STATUSES).all()


def local_upcoming_kickoffs(code: str) -> list[str] | None:
    """Inicios (UTC) de los partidos por jugar o en juego según el CSV local más
    reciente de esa competición; None si no hay CSV (hay que preguntar a la API)."""
    archivos = list(config.PROCESSED_DIR.glob(f"matches_{code}_*.csv"))
    if not archivos:
        return None
    reciente = max(archivos, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
    try:
        df = pd.read_csv(reciente, usecols=["status", "utc_date"])
    except Exception:
        return None
    vigentes = {"SCHEDULED", "TIMED", "IN_PLAY", "PAUSED", "LIVE"}
    return df.loc[df["status"].isin(vigentes), "utc_date"].tolist()
