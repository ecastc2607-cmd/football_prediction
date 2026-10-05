"""
Ancla de mercado para la base pre-partido del modelo en vivo.

El ajuste en vivo (live_matches.adjusted_live_probabilities) parte de los
goles esperados de ANTES del partido. Si esa base está mal — como en
selecciones, donde el modelo acierta poco: Francia-Bélgica (oct-2026) le daba
37% / 29% / 34%, casi parejo — todo lo que se calcula encima arrastra el error.

Las casas no publican goles esperados, pero sí cuotas: acá se buscan los goles
esperados (local, visitante) cuyo Poisson reproduce mejor las probabilidades
del mercado (1X2 sin margen + Más/Menos 2.5) y se promedian con los del modelo
(MARKET_WEIGHT) — mismo criterio que value_tips.py y parlay_builder.py.

Solo usa cuotas YA guardadas en el historial (odds_api.SNAPSHOT_PATH) y
consultadas ANTES del inicio: no gasta créditos ni usa cuotas en vivo, que ya
incluyen lo que pasó en el partido.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from . import odds_api

MARKET_WEIGHT = 0.5

_GRID = np.round(np.arange(0.20, 4.001, 0.02), 2)
_GOLES = np.arange(0, 11)
_CACHE: dict = {}


def _grids():
    """Probabilidades Local/Empate/Visitante/Más de 2.5 para cada par de
    goles esperados de la grilla (se calcula una sola vez)."""
    if not _CACHE:
        pmf = poisson.pmf(_GOLES[None, :], _GRID[:, None])  # (n, goles)
        gh, ga = np.meshgrid(_GOLES, _GOLES, indexing="ij")
        def suma(mascara):
            return np.einsum("ig,jh,gh->ij", pmf, pmf, mascara.astype(float))
        _CACHE.update(home=suma(gh > ga), draw=suma(gh == ga), away=suma(gh < ga),
                      over=1 - suma(gh + ga <= 2))
    return _CACHE


def _price(row: dict, key: str) -> float | None:
    try:
        p = float(row.get(f"all_{key}"))
    except (TypeError, ValueError):
        return None
    return p if p == p and p > 1 else None


def market_probabilities(row: dict) -> dict | None:
    """1X2 y Más de 2.5 implícitos en las cuotas, sin el margen de la casa."""
    h, d, a = _price(row, "home"), _price(row, "draw"), _price(row, "away")
    if not (h and d and a):
        return None
    inv = np.array([1 / h, 1 / d, 1 / a])
    ph, pd_, pa = inv / inv.sum()
    probs = {"home": ph, "draw": pd_, "away": pa, "over": None}
    o, u = _price(row, "over"), _price(row, "under")
    if o and u:
        probs["over"] = (1 / o) / (1 / o + 1 / u)
    return probs


def market_lambdas(row: dict) -> tuple[float, float] | None:
    """Goles esperados (local, visitante) que mejor reproducen el mercado."""
    probs = market_probabilities(row)
    if probs is None:
        return None
    g = _grids()
    error = (g["home"] - probs["home"]) ** 2 + (g["draw"] - probs["draw"]) ** 2 + (g["away"] - probs["away"]) ** 2
    if probs["over"] is not None:
        error = error + (g["over"] - probs["over"]) ** 2
    i, j = np.unravel_index(np.argmin(error), error.shape)
    return float(_GRID[i]), float(_GRID[j])


def prematch_market_row(competition: str, home: str, away: str, utc_date: str,
                        home_short: str = "", away_short: str = "") -> dict | None:
    """La última cuota guardada de ese partido consultada ANTES de su inicio."""
    if not odds_api.SNAPSHOT_PATH.exists():
        return None
    hist = pd.read_csv(odds_api.SNAPSHOT_PATH)
    inicio = pd.Timestamp(utc_date)
    previas = hist[(hist["competition"] == competition) & hist["event_id"].notna()
                   & (pd.to_datetime(hist["fetched_at"], utc=True) < inicio)]
    if previas.empty:
        return None
    filas = previas.sort_values("fetched_at", ascending=False).to_dict("records")
    return odds_api.find_event(filas, home, away, utc_date, home_short, away_short)


def anchored_prematch_xg(model_home_xg: float, model_away_xg: float, competition: str,
                         home: str, away: str, utc_date: str,
                         home_short: str = "", away_short: str = "") -> tuple[float, float, dict | None]:
    """(xG local, xG visitante, info). Sin cuota previa guardada devuelve los
    del modelo e info=None."""
    fila = prematch_market_row(competition, home, away, utc_date, home_short, away_short)
    lambdas = market_lambdas(fila) if fila else None
    if lambdas is None:
        return model_home_xg, model_away_xg, None
    mh, ma = lambdas
    return (
        (1 - MARKET_WEIGHT) * model_home_xg + MARKET_WEIGHT * mh,
        (1 - MARKET_WEIGHT) * model_away_xg + MARKET_WEIGHT * ma,
        {"market_home_xg": mh, "market_away_xg": ma, "fetched_at": fila.get("fetched_at")},
    )
