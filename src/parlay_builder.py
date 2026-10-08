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

from .calibration import HALF_MARKETS, Calibrator, candidate_legs

# Elegidos con parlay_backtest.py (15 jornadas, oct-2026) entre 0.60/0.65/0.70
# y 1/2 picks por partido: 0.60 + 2 fue donde lo PROMETIDO coincidió mejor con
# lo REAL en los tres niveles. Muestra chica: re-evaluar con
# `python -m src.parlay_backtest` cuando haya más jornadas.
MIN_LEG_PROB = 0.60  # probabilidad CALIBRADA mínima de cada pick
MIN_LEGS = 2
MAX_LEGS = 5
LEGS_PER_MATCH = 2  # mejores picks por partido que se consideran (se usa 1 por combinada)
MAX_MATCHES_CONSIDERED = 12
BOOKMAKER_MARGIN = 0.05  # margen típico por selección de una casa
# Mercados de un solo tiempo: las casas cobran más margen ahí, y no tenemos su
# cuota real (The Odds API los cobra por partido) — se estiman con este margen.
HALF_MARKET_MARGIN = 0.07
# Si entran picks de un solo tiempo, y con cuántos picks resueltos de ese
# mercado como mínimo (backtest + registro). Decidido con half_backtest.py
# (oct-2026, 352 partidos, 34 jornadas, calibración sin la jornada evaluada):
#   - Gol 1T (67% real vs 69% prometido) y Gol 2T (80% vs 77%): bien calibrados.
#   - +/-1.5 en 1T (61% vs 68%) y en 2T (50% vs 61%): sobreestimados.
#   - Combinadas, retorno por $1 sin -> con picks por tiempo: Alta 0.99 -> 0.99,
#     Equilibrada 1.01 -> 0.92, Cuota alta 0.90 -> 0.58. Aun dejando solo los
#     bien calibrados: 0.99 / 1.00 / 0.66. El margen extra de la casa en esos
#     mercados se come la cuota más alta. Apagado hasta que la evidencia cambie:
#     re-evaluar con `python -m src.half_backtest` (la tarea semanal lo corre).
HALF_MARKETS_ENABLED = False
HALF_MIN_SAMPLE = 150

# (nombre, probabilidad combinada mínima, máxima) — niveles absolutos.
TIERS = [
    ("Alta probabilidad", 0.40, 1.01),
    ("Equilibrada", 0.25, 0.40),
    ("Cuota alta", 0.12, 0.25),
]
MAX_SUGGESTIONS_PER_TIER = 3
MAX_SHARED_FRACTION = 0.5  # dos combinadas del mismo nivel comparten como mucho la mitad de sus partidos

# Monto plano por combinada: una fracción FIJA y chica de la banca, solo para las
# que tienen valor confirmado a cuota real. Kelly no sirve acá: en combinadas da
# montos minúsculos y es muy sensible a errores de probabilidad, que se
# multiplican pick a pick.
COMBO_STAKE_FRACTION = 0.01
# Con cuota real, la probabilidad del pick es un promedio entre la calibrada del
# modelo y la que implica la cuota: un Poisson simple se equivoca más que el
# mercado. Medido: sin esto, Nations League J4 (oct-2026) daba combinadas con
# +100% de valor esperado — imposible; era el modelo (34% de acierto en esa
# competición) contradiciendo al mercado. Mismo criterio que value_tips.py.
MODEL_WEIGHT_WITH_ODDS = 0.5
# Si la casa paga más de esto veces la cuota justa del modelo, el mercado
# discrepa demasiado (lesión, rotación, error de emparejamiento): el pick se
# DESCARTA en vez de tratarlo como una ganga.
MAX_ODDS_VS_FAIR = 1.3
MIN_LEG_ODDS = 1.05  # un pick que paga menos que esto no aporta y solo suma riesgo


def odds_key(market: str, pick: str) -> str | None:
    """Clave de odds_api.PRICE_KEYS para un pick, o None si ese mercado no
    tiene cuota real disponible (Ambos anotan)."""
    if market == "Resultado":
        return {"Local": "home", "Empate": "draw", "Visitante": "away"}.get(pick)
    if market == "Doble oportunidad":
        return "dc_1x" if pick.startswith("1X") else "dc_x2"
    if market == "Goles":
        return "over" if pick.startswith("+") else "under"
    return None


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
    book_odds: float | None = None  # mejor cuota real encontrada (None = no hay)
    bookmaker: str = ""

    @property
    def payout_odds(self) -> float:
        """La real si existe; si no, la justa menos el margen típico de una casa
        (mayor en los mercados de un solo tiempo)."""
        if self.book_odds:
            return self.book_odds
        margen = HALF_MARKET_MARGIN if self.market in HALF_MARKETS else BOOKMAKER_MARGIN
        return self.fair_odds * (1 - margen)


@dataclass
class Parlay:
    legs: list[Leg]
    combined_probability: float
    combined_odds: float  # justa
    estimated_odds: float  # lo que pagaría una casa: real donde la hay, estimada donde no
    risk: str  # nombre del nivel (TIERS)

    @property
    def all_real_odds(self) -> bool:
        return all(l.book_odds for l in self.legs)

    @property
    def expected_value(self) -> float:
        """Ganancia esperada por peso apostado (0.05 = +5%). Solo es una
        medida de VALOR real si all_real_odds; si no, es una estimación."""
        return self.combined_probability * self.estimated_odds - 1

    def stake(self, bankroll: float) -> int:
        """Monto sugerido (múltiplo de $100) — 0 si no hay valor confirmado."""
        if not self.all_real_odds or self.expected_value <= 0:
            return 0
        return int(bankroll * COMBO_STAKE_FRACTION // 100 * 100)

    def describe(self) -> str:
        return " + ".join(f"{l.match} ({l.market}: {l.pick})" for l in self.legs)


def legs_by_match(predictions: pd.DataFrame, calibrator: Calibrator | None,
                  odds_lookup=None, include_half: bool | None = None) -> dict[str, list[Leg]]:
    """`odds_lookup(fila_de_predicción, clave) -> (cuota, casa) | None` — de
    dónde sacar la cuota real de cada pick (ver app.py); None = sin cuotas.
    `include_half`: forzar con/sin mercados de un solo tiempo (backtest); por
    defecto, HALF_MARKETS_ENABLED."""
    calibrator = calibrator or Calibrator()
    con_mitades = HALF_MARKETS_ENABLED if include_half is None else include_half
    por_partido: dict[str, list[Leg]] = {}
    for _, row in predictions.iterrows():
        home = row.get("home_team_short") or row["home_team"]
        away = row.get("away_team_short") or row["away_team"]
        partido = f"{home} vs {away}"
        legs = []
        mitades = (tuple(row.get(c) for c in ("h1_home_xg", "h1_away_xg", "h2_home_xg", "h2_away_xg"))
                   if con_mitades else None)
        for mercado, pick, p_modelo in candidate_legs(row["home_win"], row["draw"], row["away_win"],
                                                     row.get("over_2_5"), row.get("btts"), mitades):
            if mercado in HALF_MARKETS and calibrator.market_sample(mercado) < HALF_MIN_SAMPLE:
                continue  # sin evidencia suficiente de ese mercado todavía
            p = calibrator.calibrate(mercado, p_modelo, row.get("competition"))
            if p < MIN_LEG_PROB or p >= 1:
                continue
            clave = odds_key(mercado, pick)
            real = odds_lookup(row, clave) if (odds_lookup and clave) else None
            if real:
                if real[0] > MAX_ODDS_VS_FAIR / p:
                    continue  # el mercado contradice fuerte al modelo: fuera
                if real[0] < MIN_LEG_ODDS:
                    continue  # paga casi nada y suma riesgo a la combinada
                p = MODEL_WEIGHT_WITH_ODDS * p + (1 - MODEL_WEIGHT_WITH_ODDS) / real[0]
                if p < MIN_LEG_PROB:
                    continue
            leg = Leg(partido, mercado, pick, p, p_modelo, round(1 / p, 2), row["home_team"], row["away_team"])
            if real:
                leg.book_odds, leg.bookmaker = round(real[0], 2), real[1]
            legs.append(leg)
        if legs:
            legs.sort(key=lambda l: l.probability, reverse=True)
            por_partido[partido] = legs[:LEGS_PER_MATCH]
    return por_partido


def _tier(prob: float) -> str | None:
    for nombre, piso, techo in TIERS:
        if piso <= prob < techo:
            return nombre
    return None


def build_parlays(predictions: pd.DataFrame, calibrator: Calibrator | None = None,
                  odds_lookup=None, include_half: bool | None = None) -> list[Parlay]:
    por_partido = legs_by_match(predictions, calibrator, odds_lookup, include_half)
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
                paga = 1.0
                for l in legs:
                    paga *= l.payout_odds
                candidatas[nivel].append(Parlay(
                    legs=list(legs), combined_probability=prob, combined_odds=round(1 / prob, 2),
                    estimated_odds=round(paga, 2), risk=nivel,
                ))

    elegidas: list[Parlay] = []
    for nombre, _, _ in TIERS:
        # Primero el mayor VALOR ESPERADO (prob. calibrada x lo que paga la
        # casa): con cuotas reales premia los picks que la casa paga de más; sin
        # ellas, castiga cada pick extra con su margen. A igual valor, la que
        # más paga; a igual paga, la de menos picks.
        orden = sorted(candidatas[nombre],
                       key=lambda p: (-round(p.expected_value, 4), -p.estimated_odds, len(p.legs)))
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
