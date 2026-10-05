"""
Combinadas (parlays) sugeridas a partir de las predicciones de una jornada o de
un día — v2, rediseñada con la calibración real del proyecto (ver
calibration.py y parlay_backtest.py).

Qué cambió respecto de la v1 y por qué (medido sobre el histórico resuelto):
  - v1 exigía cuota combinada de 6x a 30x: eso fuerza probabilidades de 5-17%,
    y aun así etiquetaba "Riesgo bajo" a combinadas que fallaban 3 de cada 4.
    Ahora los niveles se definen por probabilidad ABSOLUTA (TIERS), no por
    terciles relativos.
  - v1 usaba la probabilidad del modelo tal cual. Ahora cada pick usa la
    probabilidad CALIBRADA con resultados reales: "Goles 2.5" (mal calibrado)
    queda fuera solo, y entra "Doble oportunidad" (el mercado más fiable).
  - v1 sugería 4 combinadas casi idénticas por nivel. Ahora no se repiten
    partidos más allá de MAX_SHARED_FRACTION entre combinadas del mismo nivel.
  - Dentro de cada nivel se elige la de MAYOR cuota que respeta su piso de
    probabilidad: la mejor paga posible para ese riesgo, no la "menos mala".

Supuestos que siguen: partidos independientes entre sí (por eso nunca dos
picks del mismo partido) y cuota justa = 1/probabilidad; la de una casa real
paga menos — se muestra aparte una estimación con BOOKMAKER_MARGIN por pick.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import pandas as pd

from .calibration import Calibrator, candidate_legs

# Elegidos con parlay_backtest.py (15 jornadas, oct-2026) entre 0.60/0.65/0.70
# y 1/2 picks por partido: 0.60 + 2 es donde lo PROMETIDO coincide mejor con lo
# REAL en los tres niveles (40% vs 41%, 33% vs 26%, 20% vs 17%). Muestra chica:
# re-evaluar con `python -m src.parlay_backtest` cuando haya más jornadas.
MIN_LEG_PROB = 0.60  # probabilidad CALIBRADA mínima de cada pick
MIN_LEGS = 2
MAX_LEGS = 5
LEGS_PER_MATCH = 2  # mejores picks por partido que se consideran (se usa 1 por combinada)
MAX_MATCHES_CONSIDERED = 12
BOOKMAKER_MARGIN = 0.05  # margen típico por selección de una casa

# (nombre, probabilidad combinada mínima, máxima) — niveles absolutos.
TIERS = [
    ("Alta probabilidad", 0.40, 1.01),
    ("Equilibrada", 0.25, 0.40),
    ("Cuota alta", 0.12, 0.25),
]
MAX_SUGGESTIONS_PER_TIER = 3
MAX_SHARED_FRACTION = 0.5  # dos combinadas del mismo nivel comparten como mucho la mitad de sus partidos


@dataclass
class Leg:
    match: str  # "Sunderland vs Leeds Utd" (local y visitante juntos)
    market: str  # "Resultado" / "Doble oportunidad" / "Goles" / "Ambos anotan"
    pick: str  # "Local" / "1X (local o empate)" / "+2.5 goles" / "Sí"
    probability: float  # calibrada, 0-1
    model_probability: float  # la del modelo antes de calibrar
    fair_odds: float
    home_team: str = ""
    away_team: str = ""


@dataclass
class Parlay:
    legs: list[Leg]
    combined_probability: float
    combined_odds: float  # justa
    estimated_odds: float  # aproximada en una casa (con margen)
    risk: str  # nombre del nivel (TIERS)

    def describe(self) -> str:
        return " + ".join(f"{l.match} ({l.market}: {l.pick})" for l in self.legs)


def legs_by_match(predictions: pd.DataFrame, calibrator: Calibrator | None) -> dict[str, list[Leg]]:
    calibrator = calibrator or Calibrator()
    por_partido: dict[str, list[Leg]] = {}
    for _, row in predictions.iterrows():
        home = row.get("home_team_short") or row["home_team"]
        away = row.get("away_team_short") or row["away_team"]
        partido = f"{home} vs {away}"
        legs = []
        for mercado, pick, p_modelo in candidate_legs(row["home_win"], row["draw"], row["away_win"],
                                                     row.get("over_2_5"), row.get("btts")):
            p = calibrator.calibrate(mercado, p_modelo)
            if p < MIN_LEG_PROB or p >= 1:
                continue
            legs.append(Leg(partido, mercado, pick, p, p_modelo, round(1 / p, 2),
                            row["home_team"], row["away_team"]))
        if legs:
            legs.sort(key=lambda l: l.probability, reverse=True)
            por_partido[partido] = legs[:LEGS_PER_MATCH]
    return por_partido


def _tier(prob: float) -> str | None:
    for nombre, piso, techo in TIERS:
        if piso <= prob < techo:
            return nombre
    return None


def build_parlays(predictions: pd.DataFrame, calibrator: Calibrator | None = None) -> list[Parlay]:
    por_partido = legs_by_match(predictions, calibrator)
    if len(por_partido) < MIN_LEGS:
        return []
    partidos = sorted(por_partido, key=lambda m: por_partido[m][0].probability,
                      reverse=True)[:MAX_MATCHES_CONSIDERED]

    candidatas: dict[str, list[Parlay]] = {nombre: [] for nombre, _, _ in TIERS}
    for tam in range(MIN_LEGS, min(MAX_LEGS, len(partidos)) + 1):
        for combo_partidos in combinations(partidos, tam):
            for legs in product(*(por_partido[m] for m in combo_partidos)):
                prob = 1.0
                for l in legs:
                    prob *= l.probability
                nivel = _tier(prob)
                if nivel is None:
                    continue
                justa = 1 / prob
                candidatas[nivel].append(Parlay(
                    legs=list(legs), combined_probability=prob, combined_odds=round(justa, 2),
                    estimated_odds=round(justa * (1 - BOOKMAKER_MARGIN) ** len(legs), 2), risk=nivel,
                ))

    elegidas: list[Parlay] = []
    for nombre, _, _ in TIERS:
        # La mejor paga REAL posible para ese nivel de riesgo: se ordena por la
        # cuota estimada en casa (no la justa), que ya castiga cada pick extra
        # con su margen — a igual cuota justa, gana la de menos picks.
        orden = sorted(candidatas[nombre], key=lambda p: (-p.estimated_odds, len(p.legs)))
        del_nivel: list[Parlay] = []
        for p in orden:
            partidos_p = {l.match for l in p.legs}
            if any(len(partidos_p & {l.match for l in q.legs}) > MAX_SHARED_FRACTION * min(len(p.legs), len(q.legs))
                   for q in del_nivel):
                continue
            del_nivel.append(p)
            if len(del_nivel) == MAX_SUGGESTIONS_PER_TIER:
                break
        elegidas.extend(del_nivel)
    return elegidas
