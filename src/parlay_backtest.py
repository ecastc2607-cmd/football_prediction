"""
Backtest de la estrategia de combinadas sobre las jornadas ya resueltas: para
cada jornada (competición + jornada) arma las combinadas como lo haría el
dashboard y mira cuántas habrían acertado.

Sin trampa de calibración: la calibración que se usa para cada jornada se
ajusta con TODAS LAS DEMÁS jornadas, nunca con la que se evalúa
("leave-one-out"). Así el número mide lo que pasaría con jornadas futuras.

Uso:
    python -m src.parlay_backtest
"""
from __future__ import annotations

import pandas as pd

from .calibration import Calibrator, leg_won, resolved_rows
from .parlay_builder import TIERS, build_parlays
from .prediction_log import load_log


def _as_predictions(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.rename(columns={
        "pred_home_pct": "home_win", "pred_draw_pct": "draw", "pred_away_pct": "away_win",
        "over_2_5_pct": "over_2_5", "btts_pct": "btts",
    })


def backtest(log: pd.DataFrame | None = None) -> pd.DataFrame:
    """Una fila por combinada evaluada: nivel, picks, prob. prometida, cuotas,
    y si acertó."""
    log = load_log() if log is None else log
    resueltas = resolved_rows(log)
    if resueltas.empty:
        return pd.DataFrame()
    grupos = resueltas.groupby(["competition", "matchday"]).groups

    filas = []
    for (comp, jornada), idx in grupos.items():
        jornada_rows = resueltas.loc[idx]
        if len(jornada_rows) < 2:
            continue
        calibracion = Calibrator.from_log(log.drop(index=idx))
        por_partido = {
            f"{r.home_team} vs {r.away_team}": (int(r.actual_home_goals), int(r.actual_away_goals),
                                               r.actual_ht_home_goals, r.actual_ht_away_goals)
            for r in jornada_rows.itertuples()
        }
        for p in build_parlays(_as_predictions(jornada_rows), calibracion):
            acerto = all(leg_won(l.market, l.pick, *por_partido[l.match]) for l in p.legs)
            filas.append({
                "competition": comp, "matchday": jornada, "nivel": p.risk, "picks": len(p.legs),
                "prob_prometida": p.combined_probability, "cuota_justa": p.combined_odds,
                "cuota_estimada": p.estimated_odds, "acerto": acerto,
            })
    return pd.DataFrame(filas)


def summary(resultados: pd.DataFrame) -> pd.DataFrame:
    """Por nivel: combinadas, acertadas, % real vs prometido, y retorno por
    cada peso apostado a la cuota ESTIMADA de casa (>1 = habría ganado plata)."""
    if resultados.empty:
        return pd.DataFrame()
    filas = []
    for nombre, _, _ in TIERS:
        g = resultados[resultados["nivel"] == nombre]
        if g.empty:
            continue
        filas.append({
            "Nivel": nombre,
            "Combinadas": len(g),
            "Acertadas": int(g["acerto"].sum()),
            "Acierto real": g["acerto"].mean(),
            "Prob. prometida": g["prob_prometida"].mean(),
            "Retorno por $1": (g["cuota_estimada"] * g["acerto"]).mean(),
        })
    return pd.DataFrame(filas)


def main():
    resultados = backtest()
    if resultados.empty:
        print("No hay jornadas resueltas para evaluar.")
        return
    print(f"{resultados.groupby(['competition', 'matchday']).ngroups} jornadas evaluadas.\n")
    print(summary(resultados).to_string(index=False, formatters={
        "Acierto real": "{:.0%}".format, "Prob. prometida": "{:.0%}".format,
        "Retorno por $1": "{:.2f}".format,
    }))


if __name__ == "__main__":
    main()
