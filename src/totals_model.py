"""
Probabilidad de corners y tarjetas totales de un partido por encima/debajo
de una línea cualquiera (la que ofrezca la casa: 8.5, 9.5... / 3.5, 4.5...).

Antes el proyecto solo comparaba los promedios contra una línea fija (8.5 /
3.5) y decía "más" o "menos", sin probabilidad: así no se podía evaluar el
valor contra una cuota (pedido del usuario, oct-2026: incluir corners y
tarjetas en Tips si de verdad aguantan).

Método (el mismo esquema que los goles en team_strength/poisson_model):
  esperados_local = promedio_liga_local × produce_local(L) × concede_visita(V)
  esperados_visita = promedio_liga_visita × produce_visita(V) × concede_local(L)
con cada factor encogido hacia 1 cuando hay pocos partidos (PRIOR_GAMES). El
total no es Poisson: corners y tarjetas varían más que su media (sobre-
dispersión), así que se usa una binomial negativa cuyo parámetro se estima de
los partidos previos de la misma liga.

Datos: data/tracking/match_stats_log.csv (estadísticas finales por partido).
Tarjetas = amarillas + rojas (como cuentan las casas el "total de tarjetas",
una por tarjeta).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from . import match_stats_log

# Partidos "fantasma" al promedio de liga en cada factor. Backtest walk-forward
# (oct-2026, 283 partidos, totals_backtest.py): con 5 el modelo era peor que la
# simple frecuencia de la liga en 3 de 6 líneas y sobreconfiado; con 15 le gana
# en las 6 (log-loss 0-1.7% mejor) y casi calibrado (dice 74%, acierta 72%).
PRIOR_GAMES = 15
# Lo que queda de sobreconfianza: la probabilidad se acerca a 50% en esta
# proporción (0.74 -> 0.72, lo medido).
CALIBRATION_SLOPE = 0.9
MIN_LEAGUE_MATCHES = 30  # por debajo, no hay base fiable para esa liga
MARKETS = {"corners": "corners", "tarjetas": "tarjetas"}


def match_totals(log: pd.DataFrame | None = None) -> pd.DataFrame:
    """Un renglón por partido: corners/tarjetas de local y visitante."""
    log = match_stats_log.load_log() if log is None else log
    if log.empty:
        return pd.DataFrame()
    log = log.assign(tarjetas=pd.to_numeric(log["tarjetas_amarillas"], errors="coerce")
                     + pd.to_numeric(log["tarjetas_rojas"], errors="coerce").fillna(0),
                     corners=pd.to_numeric(log["corners"], errors="coerce"))
    es_local = log["is_home"].astype(str).str.lower().isin(["true", "1"])
    clave = ["competition", "utc_date", "home_team", "away_team"]
    local = log[es_local].drop_duplicates(clave).set_index(clave)[["corners", "tarjetas"]]
    visita = log[~es_local].drop_duplicates(clave).set_index(clave)[["corners", "tarjetas"]]
    df = local.join(visita, lsuffix="_home", rsuffix="_away", how="inner").reset_index()
    df["inicio"] = pd.to_datetime(df["utc_date"], utc=True, errors="coerce")
    return df.dropna(subset=["inicio"]).sort_values("inicio").reset_index(drop=True)


@dataclass
class TotalsStrength:
    home_avg: float
    away_avg: float
    dispersion: float  # k de la binomial negativa (más chico = más disperso); inf = Poisson
    teams: pd.DataFrame  # índice equipo: prod_home, conc_home, prod_away, conc_away


def _factor(suma: pd.Series, partidos: pd.Series, promedio: float) -> pd.Series:
    return ((suma + PRIOR_GAMES * promedio) / (partidos + PRIOR_GAMES)) / promedio


def fit(prior: pd.DataFrame, market: str) -> TotalsStrength | None:
    """Fuerzas a partir de partidos previos (de UNA liga) para corners o tarjetas."""
    h, a = f"{market}_home", f"{market}_away"
    d = prior.dropna(subset=[h, a])
    if len(d) < MIN_LEAGUE_MATCHES:
        return None
    home_avg, away_avg = d[h].mean(), d[a].mean()
    loc = d.groupby("home_team").agg(n_h=(h, "size"), prod_h=(h, "sum"), conc_h=(a, "sum"))
    vis = d.groupby("away_team").agg(n_a=(a, "size"), prod_a=(a, "sum"), conc_a=(h, "sum"))
    t = loc.join(vis, how="outer").fillna(0)
    teams = pd.DataFrame({
        "prod_home": _factor(t["prod_h"], t["n_h"], home_avg),
        "conc_home": _factor(t["conc_h"], t["n_h"], away_avg),
        "prod_away": _factor(t["prod_a"], t["n_a"], away_avg),
        "conc_away": _factor(t["conc_a"], t["n_a"], home_avg),
    })
    total = d[h] + d[a]
    media, var = total.mean(), total.var()
    # Binomial negativa: var = media + media²/k  ->  k = media² / (var - media).
    dispersion = media ** 2 / (var - media) if var > media else float("inf")
    return TotalsStrength(home_avg, away_avg, dispersion, teams)


def expected_total(s: TotalsStrength, home: str, away: str) -> float:
    t = s.teams
    ph = t.at[home, "prod_home"] if home in t.index else 1.0
    ch = t.at[home, "conc_home"] if home in t.index else 1.0
    pa = t.at[away, "prod_away"] if away in t.index else 1.0
    ca = t.at[away, "conc_away"] if away in t.index else 1.0
    return float(s.home_avg * ph * ca + s.away_avg * pa * ch)


def prob_over(s: TotalsStrength, mu: float, line: float) -> float:
    """P(total > línea) — líneas de medio punto (8.5, 9.5...)."""
    k = int(np.floor(line))
    if not np.isfinite(s.dispersion):
        return float(1 - poisson.cdf(k, mu))
    n = s.dispersion
    return float(1 - nbinom.cdf(k, n, n / (n + mu)))


# --- Uso en vivo (Tips): fuerzas con TODO el historial de la liga ---

_CACHE: dict = {}


def strength_for(competition: str, market: str) -> TotalsStrength | None:
    # La fecha del archivo entra en la clave: cuando llegan estadísticas nuevas
    # (la app de Streamlit vive días), las fuerzas se recalculan.
    path = match_stats_log.LOG_PATH
    clave = (competition, market, path.stat().st_mtime if path.exists() else 0)
    if clave not in _CACHE:
        totales = match_totals()
        _CACHE[clave] = fit(totales[totales["competition"] == competition], market) if not totales.empty else None
    return _CACHE[clave]


def predict(competition: str, home: str, away: str, market: str, line: float) -> tuple[float, float] | None:
    """(esperados, P(más de la línea)) o None si no hay base para esa liga."""
    s = strength_for(competition, market)
    if s is None:
        return None
    mu = expected_total(s, home, away)
    p = prob_over(s, mu, line)
    return mu, 0.5 + CALIBRATION_SLOPE * (p - 0.5)
