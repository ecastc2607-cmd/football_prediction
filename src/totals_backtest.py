"""
Backtest walk-forward del modelo de corners/tarjetas (totals_model.py): cada
partido se predice solo con los partidos ANTERIORES de su liga, y se compara
contra la referencia más simple — la frecuencia con que esa línea se superó en
la liga hasta ese momento. Si el modelo no le gana a eso, no aporta nada.

Uso:
    python -m src.totals_backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import totals_model as tm

LINES = {"corners": [8.5, 9.5, 10.5], "tarjetas": [3.5, 4.5, 5.5]}
EPS = 1e-6


def run(totales: pd.DataFrame | None = None) -> pd.DataFrame:
    totales = tm.match_totals() if totales is None else totales
    filas = []
    for comp, liga in totales.groupby("competition"):
        liga = liga.reset_index(drop=True)
        for i in range(len(liga)):
            m = liga.iloc[i]
            previos = liga[liga["inicio"] < m["inicio"]]
            for market, lineas in LINES.items():
                if pd.isna(m[f"{market}_home"]) or pd.isna(m[f"{market}_away"]):
                    continue
                s = tm.fit(previos, market)
                if s is None:
                    continue
                total_prev = previos[f"{market}_home"] + previos[f"{market}_away"]
                mu = tm.expected_total(s, m["home_team"], m["away_team"])
                real = m[f"{market}_home"] + m[f"{market}_away"]
                for linea in lineas:
                    filas.append({
                        "competition": comp, "inicio": m["inicio"], "mercado": market, "linea": linea,
                        "esperado": mu, "real": real, "p_modelo": tm.prob_over(s, mu, linea),
                        "p_base": float((total_prev > linea).mean()), "over": real > linea,
                    })
    return pd.DataFrame(filas)


def _logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(np.where(y, np.log(p), np.log(1 - p))))


def summary(bt: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for (market, linea), g in bt.groupby(["mercado", "linea"]):
        y = g["over"].to_numpy()
        filas.append({
            "mercado": market, "linea": linea, "partidos": len(g), "supera_%": round(y.mean() * 100),
            "logloss_modelo": round(_logloss(g["p_modelo"], y), 4),
            "logloss_base": round(_logloss(g["p_base"], y), 4),
            "brier_modelo": round(float(np.mean((g["p_modelo"] - y) ** 2)), 4),
            "brier_base": round(float(np.mean((g["p_base"] - y) ** 2)), 4),
        })
    return pd.DataFrame(filas)


def calibration(bt: pd.DataFrame, min_prob: float = 0.5) -> pd.DataFrame:
    """Pick = el lado más probable (más/menos). Acierto real por tramo de probabilidad."""
    d = bt.assign(p_pick=np.maximum(bt["p_modelo"], 1 - bt["p_modelo"]),
                  acierto=np.where(bt["p_modelo"] >= 0.5, bt["over"], ~bt["over"]))
    d = d[d["p_pick"] >= min_prob]
    d["tramo"] = pd.cut(d["p_pick"], [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0], right=False)
    return d.groupby(["mercado", "tramo"], observed=True).agg(
        picks=("acierto", "size"), prob_media=("p_pick", "mean"), acierto=("acierto", "mean")).round(3)


def main():
    bt = run()
    pd.set_option("display.width", 200)
    print(f"{bt.groupby(['competition', 'inicio']).ngroups} partidos evaluados (walk-forward por liga)\n")
    print(summary(bt).to_string(index=False))
    print("\nCalibración del pick (lado más probable):")
    print(calibration(bt).to_string())
    print("\nEsperados vs reales (sesgo):")
    print(bt.drop_duplicates(["competition", "inicio", "mercado"]).groupby("mercado")[["esperado", "real"]].mean().round(2))


if __name__ == "__main__":
    main()
