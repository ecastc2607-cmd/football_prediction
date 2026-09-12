"""
Vista de partidos en vivo: encuentra los partidos IN_PLAY/PAUSED de una o varias
competiciones, y recalcula la probabilidad 1X2 EN VIVO combinando:
  - el marcador actual real
  - los goles esperados (xG) que ya calculó el modelo pre-partido
  - el tiempo restante estimado (la API no da el minuto exacto, así que se
    aproxima desde la hora de inicio — se marca como estimado en la UI)

Método: se reparte el xG total del modelo proporcional al tiempo que falta,
y se suma como goles adicionales de Poisson al marcador ya en curso.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import poisson

from . import config
from .fetch_football_data import FootballDataClient, fetch_competition
from .team_strength import team_strength_for_competition
from .poisson_model import expected_goals

MAX_EXTRA_GOALS = 6
REGULATION_MINUTES = 90
HALFTIME_BREAK_MINUTES = 15


def estimate_elapsed_minutes(utc_date: str) -> int:
    """Aproxima el minuto de juego a partir de la hora de inicio (la API no
    entrega el minuto real). No contempla tiempo añadido."""
    kickoff = datetime.strptime(utc_date, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    elapsed = (datetime.now(timezone.utc) - kickoff).total_seconds() / 60
    if elapsed > 45:
        elapsed -= HALFTIME_BREAK_MINUTES  # descuenta el entretiempo
    return int(max(0, min(elapsed, REGULATION_MINUTES)))


def live_win_probabilities(home_goals: int, away_goals: int, home_xg_remaining: float,
                            away_xg_remaining: float) -> tuple[float, float, float]:
    """1X2 recalculado: marcador actual + Poisson de los goles restantes esperados."""
    extra = np.arange(0, MAX_EXTRA_GOALS + 1)
    home_extra_probs = poisson.pmf(extra, max(home_xg_remaining, 0.01))
    away_extra_probs = poisson.pmf(extra, max(away_xg_remaining, 0.01))
    matrix = np.outer(home_extra_probs, away_extra_probs)
    matrix /= matrix.sum()

    home_win = draw = away_win = 0.0
    for h_extra in extra:
        for a_extra in extra:
            p = matrix[h_extra, a_extra]
            final_h, final_a = home_goals + h_extra, away_goals + a_extra
            if final_h > final_a:
                home_win += p
            elif final_h < final_a:
                away_win += p
            else:
                draw += p
    return home_win, draw, away_win


def get_live_matches(client: FootballDataClient, codes: list[str], ensure_data=None) -> pd.DataFrame:
    """Devuelve un DataFrame con los partidos IN_PLAY/PAUSED de las competiciones
    dadas, con su 1X2 en vivo recalculado.

    `ensure_data(code, season)` deja los CSV de esa liga/temporada listos en disco
    (por defecto llama a fetch_football_data directo; la app de Streamlit le pasa
    su versión cacheada para no repetir una descarga que la vista principal ya
    hizo hace un momento)."""
    ensure_data = ensure_data or (lambda code, season: fetch_competition(client, code, season))

    rows = []
    for code in codes:
        try:
            # Un solo llamado por liga: la API filtra del lado del servidor,
            # no hace falta pedir metadata de temporada aparte para esto.
            data = client._get(f"/competitions/{code}/matches", {"status": "LIVE"})
        except Exception:
            continue

        live = data.get("matches", [])
        if not live:
            continue

        season = int(live[0].get("season", {}).get("startDate", "")[:4])
        try:
            # Solo la temporada en curso (no la anterior): para el ajuste en vivo
            # alcanza, y evita duplicar la descarga pesada de la vista principal.
            ensure_data(code, season)
            strength = team_strength_for_competition(code, [season])
        except Exception:
            strength = None

        for m in live:
            home, away = m["homeTeam"]["name"], m["awayTeam"]["name"]
            score = m.get("score", {}).get("fullTime", {})
            hg, ag = score.get("home") or 0, score.get("away") or 0
            minute = estimate_elapsed_minutes(m["utcDate"])
            remaining_frac = max(REGULATION_MINUTES - minute, 0) / REGULATION_MINUTES

            home_xg_rem = away_xg_rem = None
            live_h = live_d = live_a = None
            if strength is not None:
                try:
                    home_xg, away_xg = expected_goals(strength, home, away)
                    home_xg_rem = home_xg * remaining_frac
                    away_xg_rem = away_xg * remaining_frac
                    live_h, live_d, live_a = live_win_probabilities(hg, ag, home_xg_rem, away_xg_rem)
                except KeyError:
                    pass

            rows.append({
                "competition": code,
                "competition_name": config.COMPETITIONS.get(code, code),
                "home_team": home, "away_team": away,
                "utc_date": m["utcDate"],
                "home_goals": hg, "away_goals": ag,
                "status": m.get("status"),
                "minuto_estimado": minute,
                "live_home_win": round(live_h * 100, 1) if live_h is not None else None,
                "live_draw": round(live_d * 100, 1) if live_d is not None else None,
                "live_away_win": round(live_a * 100, 1) if live_a is not None else None,
            })
    return pd.DataFrame(rows)
