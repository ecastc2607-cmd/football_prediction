"""
Genera combinadas (parlays) sugeridas a partir de las predicciones de una
jornada: para cada partido arma hasta 3 picks posibles (1X2, Más/Menos de 2.5
goles, Ambos anotan) con su cuota justa (1/probabilidad, sin margen de casa de
apuestas), arma combinaciones de 2 a MAX_LEGS partidos usando COMO MUCHO un
pick por partido, se queda con las que caen en el rango de cuota pedido, y las
clasifica por riesgo según la probabilidad combinada real (no por cantidad de
selecciones).

Supuesto: los partidos son eventos independientes entre sí (equipos y ligas
distintas), así que la probabilidad combinada = producto de las probabilidades
individuales. Por eso nunca se combinan dos mercados del MISMO partido en una
misma combinada — estarían correlacionados (ej. "Local" y "Ambos anotan: No"
del mismo juego no son independientes).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import pandas as pd

MIN_LEGS = 2
MAX_LEGS = 6
MIN_ODDS = 6.0
MAX_ODDS = 30.0
MAX_SUGGESTIONS_PER_TIER = 4
MAX_MATCHES_CONSIDERED = 8  # acota la explosión combinatoria con muchos partidos/mercados


@dataclass
class Leg:
    match: str
    market: str  # "Local" / "Empate" / "Visitante" / "Más de 2.5 goles" / "Ambos anotan: Sí" / ...
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


def _leg(match: str, market: str, pct: float) -> Leg | None:
    prob = pct / 100
    if prob <= 0 or prob >= 1:
        return None
    return Leg(match=match, market=market, probability=prob, fair_odds=round(1 / prob, 2))


def candidate_legs_per_match(predictions: pd.DataFrame) -> dict[str, list[Leg]]:
    """Hasta 3 picks por partido: el lado más probable de 1X2, de Over/Under 2.5
    goles, y de BTTS. Cada partido aporta como mucho UNO de estos a una combinada
    (se filtra al armar las combinaciones), pero se calculan los 3 para poder
    elegir el que mejor calce en cada combinada.
    """
    by_match: dict[str, list[Leg]] = {}
    for _, row in predictions.iterrows():
        match = f"{row['home_team']} vs {row['away_team']}"
        legs = []

        opciones_1x2 = [("Local", row["home_win"]), ("Empate", row["draw"]), ("Visitante", row["away_win"])]
        market, pct = max(opciones_1x2, key=lambda x: x[1])
        leg = _leg(match, market, pct)
        if leg:
            legs.append(leg)

        over, under = row["over_2_5"], 100 - row["over_2_5"]
        market, pct = ("Más de 2.5 goles", over) if over >= under else ("Menos de 2.5 goles", under)
        leg = _leg(match, market, pct)
        if leg:
            legs.append(leg)

        si, no = row["btts"], 100 - row["btts"]
        market, pct = ("Ambos anotan: Sí", si) if si >= no else ("Ambos anotan: No", no)
        leg = _leg(match, market, pct)
        if leg:
            legs.append(leg)

        if legs:
            by_match[match] = legs
    return by_match


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
    legs_by_match = candidate_legs_per_match(predictions)
    if len(legs_by_match) < MIN_LEGS:
        return []

    # Acota a los partidos con el pick individual más fuerte, para que combinar
    # múltiples mercados por partido no explote con jornadas grandes (18 partidos
    # x 3 mercados x combinaciones de hasta 6 ya es mucho).
    matches = sorted(
        legs_by_match, key=lambda m: max(l.probability for l in legs_by_match[m]), reverse=True
    )[:MAX_MATCHES_CONSIDERED]

    candidates: list[tuple[list[Leg], float, float]] = []
    for size in range(MIN_LEGS, min(MAX_LEGS, len(matches)) + 1):
        for match_combo in combinations(matches, size):
            leg_options = [legs_by_match[m] for m in match_combo]
            for combo in product(*leg_options):
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
