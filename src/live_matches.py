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


# --- Ajuste en vivo con lo que pasa en la cancha ---
# El cálculo de arriba solo mira marcador + xG de ANTES del partido: un equipo
# que remata 10-4 y va perdiendo 0-1 sigue con el mismo ritmo de gol que se le
# suponía antes de empezar. Esto actualiza ese ritmo con sus remates.
#
# xG aproximado por remate (no tenemos el xG real en vivo): 0.20 por remate a
# puerta y 0.05 por remate desviado/bloqueado — con ~1 de cada 3 remates a
# puerta da ~0.10 por remate, el promedio que reporta la literatura de xG.
SHOT_ON_TARGET_XG = 0.20
SHOT_OFF_TARGET_XG = 0.05
# Cuánto pesa lo esperado antes del partido, en "minutos equivalentes": con 90,
# al descanso lo observado pesa ~1/3 y al minuto 80, ~47%. Actualización
# bayesiana estándar del ritmo de un proceso de Poisson (prior gamma).
PRIOR_MINUTES = 90
# Efecto marcador (Dixon & Robinson 1998): el que va perdiendo ataca más y el
# que gana se repliega. Valores conservadores; NO están validados con datos del
# proyecto (no hay histórico de partidos en vivo para eso).
TRAILING_FACTOR = 1.10
LEADING_FACTOR = 0.95
# Tiempo añadido típico (minutos) — sin él, el minuto 90 "no deja nada por jugar".
FIRST_HALF_ADDED = 2
SECOND_HALF_ADDED = 4


def remaining_minutes(minute: int) -> float:
    if minute <= 45:
        return (45 + FIRST_HALF_ADDED - minute) + 45 + SECOND_HALF_ADDED
    return max(90 + SECOND_HALF_ADDED - minute, 0.5)


def shots_xg(shots_total, shots_on_target) -> float | None:
    """xG aproximado a partir de remates; None si no hay datos."""
    try:
        total, a_puerta = float(shots_total), float(shots_on_target)
    except (TypeError, ValueError):
        return None
    if total != total or a_puerta != a_puerta:  # NaN
        return None
    return SHOT_ON_TARGET_XG * a_puerta + SHOT_OFF_TARGET_XG * max(total - a_puerta, 0)


def adjusted_live_probabilities(home_goals: int, away_goals: int, minute: int,
                                pre_home_xg: float, pre_away_xg: float,
                                home_stats: dict | None = None,
                                away_stats: dict | None = None) -> dict:
    """1X2 en vivo con: ritmo de gol actualizado con los remates de cada
    equipo, efecto marcador y tiempo añadido. Devuelve también el desglose
    (xG observado y goles esperados que faltan), para mostrarlo en la UI."""
    jugado = max(minute, 0)
    resto = remaining_minutes(minute)

    def ritmo(pre_xg, stats):
        observado = shots_xg((stats or {}).get("remates_totales"), (stats or {}).get("remates_a_puerta"))
        if observado is None:
            return pre_xg / 90, None
        return (pre_xg / 90 * PRIOR_MINUTES + observado) / (PRIOR_MINUTES + jugado), observado

    ritmo_local, obs_local = ritmo(pre_home_xg, home_stats)
    ritmo_visita, obs_visita = ritmo(pre_away_xg, away_stats)
    factor_local = factor_visita = 1.0
    if home_goals < away_goals:
        factor_local, factor_visita = TRAILING_FACTOR, LEADING_FACTOR
    elif home_goals > away_goals:
        factor_local, factor_visita = LEADING_FACTOR, TRAILING_FACTOR

    resto_local = ritmo_local * resto * factor_local
    resto_visita = ritmo_visita * resto * factor_visita
    h, d, a = live_win_probabilities(home_goals, away_goals, resto_local, resto_visita)
    return {
        "home_win": h, "draw": d, "away_win": a,
        "home_xg_observed": obs_local, "away_xg_observed": obs_visita,
        "home_xg_remaining": resto_local, "away_xg_remaining": resto_visita,
        "remaining_minutes": resto,
    }


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
            home_xg = away_xg = None
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
                # xG del modelo ANTES del partido (90'): base del ajuste en vivo
                # con estadísticas (adjusted_live_probabilities), que hace app.py.
                "pre_home_xg": home_xg,
                "pre_away_xg": away_xg,
            })
    return pd.DataFrame(rows)
