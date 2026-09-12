"""
Genera combinadas (parlays) sugeridas a partir de las predicciones de una o
varias jornadas: junta el mejor pick 1X2 de cada partido (fair odds = 1/prob,
sin margen de casa de apuestas), arma combinaciones de 2 a MAX_LEGS partidos,
se queda con las que caen en el rango de cuota pedido, y las clasifica por
riesgo según la probabilidad combinada real (no por cantidad de selecciones).

Supuesto: los partidos son eventos independientes entre sí (equipos y ligas
distintas), así que la probabilidad combinada = producto de las probabilidades
individuales. Esto es razonable entre partidos distintos; nunca se combinan dos
mercados del mismo partido (estarían correlacionados).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd

MIN_LEGS = 2
MAX_LEGS = 6
MIN_ODDS = 6.0
MAX_ODDS = 30.0
MAX_SUGGESTIONS_PER_TIER = 4


@dataclass
class Leg:
    match: str
    market: str      # "Local", "Empate" o "Visitante"
    probability: float  # 0-1
    fair_odds: float


@dataclass
class Parlay:
    legs: list[Leg]
    combined_probability: float
    combined_odds: float
    risk: str  # "Bajo", "Medio", "Alto"

    def describe(self) -> str:
        return " + ".join(f"{l.match} ({l.market})" for l in self.legs)


def best_leg_per_match(predictions: pd.DataFrame) -> list[Leg]:
    """Un solo pick por partido: el resultado 1X2 con mayor probabilidad."""
    legs = []
    for _, row in predictions.iterrows():
        options = [("Local", row["home_win"]), ("Empate", row["draw"]), ("Visitante", row["away_win"])]
        market, pct = max(options, key=lambda x: x[1])
        prob = pct / 100
        if prob <= 0 or prob >= 1:
            continue
        legs.append(Leg(
            match=f"{row['home_team']} vs {row['away_team']}",
            market=market,
            probability=prob,
            fair_odds=round(1 / prob, 2),
        ))
    return legs


def _classify_risk(probabilities: list[float]) -> dict[int, str]:
    """Asigna Bajo/Medio/Alto por terciles de probabilidad combinada entre las
    combinadas que sí califican (mayor probabilidad = menor riesgo)."""
    if not probabilities:
        return {}
    ordered = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)
    n = len(ordered)
    third = max(1, n // 3)
    labels = {}
    for rank, idx in enumerate(ordered):
        if rank < third:
            labels[idx] = "Bajo"
        elif rank < 2 * third:
            labels[idx] = "Medio"
        else:
            labels[idx] = "Alto"
    return labels


def build_parlays(predictions: pd.DataFrame) -> list[Parlay]:
    legs = best_leg_per_match(predictions)
    if len(legs) < MIN_LEGS:
        return []

    candidates: list[tuple[list[Leg], float, float]] = []
    for size in range(MIN_LEGS, min(MAX_LEGS, len(legs)) + 1):
        for combo in combinations(legs, size):
            combined_odds = 1.0
            combined_prob = 1.0
            for leg in combo:
                combined_odds *= leg.fair_odds
                combined_prob *= leg.probability
            if MIN_ODDS <= combined_odds <= MAX_ODDS:
                candidates.append((list(combo), combined_prob, combined_odds))

    if not candidates:
        return []

    risk_labels = _classify_risk([c[1] for c in candidates])

    parlays = [
        Parlay(legs=legs_, combined_probability=prob, combined_odds=round(odds, 2), risk=risk_labels[i])
        for i, (legs_, prob, odds) in enumerate(candidates)
    ]

    # Dentro de cada nivel de riesgo, prioriza la mayor probabilidad combinada
    # (la apuesta "menos mala" dentro de ese rango de cuota), y limita cuántas se muestran.
    out = []
    for risk in ("Bajo", "Medio", "Alto"):
        tier = sorted((p for p in parlays if p.risk == risk), key=lambda p: p.combined_probability, reverse=True)
        out.extend(tier[:MAX_SUGGESTIONS_PER_TIER])
    return out
