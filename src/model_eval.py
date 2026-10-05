"""
Evaluación walk-forward del modelo de goles para elegir sus parámetros con
datos, no a ojo: cada partido ya jugado de la temporada se predice SOLO con
partidos anteriores a su inicio, y se mide el log-loss (la métrica estándar
para probabilidades: castiga más cuanto más seguro estaba el modelo y falló;
más bajo = mejor) de 1X2, Más/Menos 2.5 y Ambos anotan.

Compara combinaciones de:
  - half_life_days: decaimiento temporal del peso de cada partido (0 = ninguno);
  - rho: corrección de Dixon-Coles para marcadores bajos (0 = Poisson puro).

Uso:
    python -m src.model_eval
"""
from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd

from .poisson_model import predict_match
from .team_strength import build_team_strength, load_finished_matches

LEAGUES = ["PL", "PD", "SA", "BL1", "FL1"]
EPS = 1e-12


def _partidos_a_evaluar(code: str, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    todos = load_finished_matches(code, [season, season - 1])
    todos["inicio_utc"] = pd.to_datetime(todos["utc_date"], utc=True, errors="coerce")
    return todos, todos[todos["season_start_year"] == season].sort_values("inicio_utc")


def evaluate(half_life_days: float, rho: float, season: int = 2026,
             leagues: list[str] = LEAGUES) -> dict:
    perdidas = {"1X2": [], "Goles 2.5": [], "Ambos anotan": []}
    for code in leagues:
        todos, jugados = _partidos_a_evaluar(code, season)
        fuerzas: dict = {}
        for m in jugados.itertuples():
            if m.inicio_utc not in fuerzas:
                previos = todos[todos["inicio_utc"] < m.inicio_utc]
                fuerzas[m.inicio_utc] = build_team_strength(
                    previos, half_life_days=half_life_days, as_of=m.inicio_utc
                ) if len(previos) > 20 else None
            fuerza = fuerzas[m.inicio_utc]
            if fuerza is None:
                continue
            try:
                p = predict_match(fuerza, m.home_team, m.away_team, rho=rho)
            except KeyError:
                continue
            hg, ag = int(m.home_goals), int(m.away_goals)
            real = p.home_win if hg > ag else p.away_win if hg < ag else p.draw
            perdidas["1X2"].append(-np.log(max(real, EPS)))
            over = p.over_2_5 if hg + ag > 2.5 else 1 - p.over_2_5
            perdidas["Goles 2.5"].append(-np.log(max(over, EPS)))
            ambos = p.btts if (hg > 0 and ag > 0) else 1 - p.btts
            perdidas["Ambos anotan"].append(-np.log(max(ambos, EPS)))
    resultado = {k: float(np.mean(v)) for k, v in perdidas.items()}
    resultado["partidos"] = len(perdidas["1X2"])
    return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()
    filas = []
    for vida, rho in product([0, 365, 180, 90], [0.0, -0.05, -0.10, -0.15]):
        r = evaluate(vida, rho, args.season)
        filas.append({"vida_media_días": vida or "sin decaimiento", "rho": rho, **r})
        print(f"vida media {vida or '-':>4} | rho {rho:+.2f} | " + " | ".join(
            f"{k} {v:.4f}" for k, v in r.items() if k != "partidos") + f" | n={r['partidos']}")
    df = pd.DataFrame(filas)
    df["total"] = df["1X2"] + df["Goles 2.5"] + df["Ambos anotan"]
    mejor = df.sort_values("total").iloc[0]
    base = df[(df["vida_media_días"] == "sin decaimiento") & (df["rho"] == 0.0)].iloc[0]
    print(f"\nMejor: vida media {mejor['vida_media_días']}, rho {mejor['rho']:+.2f} "
          f"(log-loss total {mejor['total']:.4f} vs {base['total']:.4f} del modelo actual)")


if __name__ == "__main__":
    main()
