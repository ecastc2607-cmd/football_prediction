"""
Capa de calibración: corrige la probabilidad que da el modelo de Poisson con
lo que REALMENTE pasó en las predicciones ya resueltas (predictions_log.csv),
por mercado y por tramo de probabilidad.

Por qué existe (medido sobre 185 predicciones resueltas, oct-2026):
  - "Resultado" con el favorito >=65%: acertó 71-88%; por debajo de 60%, solo
    34-42% aunque el modelo decía 45-62% (sobreestimaba).
  - "Doble oportunidad" (1X / X2) >=80%: acertó 78% en 51 casos.
  - "Goles 2.5": sin relación estable entre lo que dice el modelo y lo que pasa
    (60-65% -> 41% real; 75-80% -> 44% real).
Usar la probabilidad del modelo tal cual para armar combinadas trata como
igual de fiable un 65% de "Goles" (que en la práctica es ~45%) que un 65% de
"Resultado" (~70%).

Método: por cada (mercado, tramo) se cuentan aciertos reales y se mezclan con
la probabilidad del modelo como prior de PRIOR_STRENGTH partidos "fantasma":
    p_calibrada = (aciertos + PRIOR_STRENGTH * p_modelo) / (n + PRIOR_STRENGTH)
Con poca muestra manda el modelo; con mucha, la realidad. Se recalcula sola
cada vez que se resuelven predicciones nuevas — no hay números fijos a mano.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

BINS = [0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
PRIOR_STRENGTH = 15
# Segundo nivel: por competición y mercado. La tabla de arriba es global; pero
# el modelo no rinde igual en todas (Nations League: 46% en Goles 2.5 contra
# ~57% global). factor = (aciertos + K) / (esperados + K) sobre los picks de esa
# competición y mercado, con K "aciertos fantasma" que lo acercan a 1 si hay
# poca muestra. Se aplica multiplicando la probabilidad ya calibrada global, y
# solo hacia abajo (ver competition_factor).
COMPETITION_PRIOR = 10
EXCLUDED_SOURCES = {"dashboard_v1_qualitative"}  # predicciones "a ojo", no del modelo

FULL_MARKETS = ["Resultado", "Doble oportunidad", "Goles", "Ambos anotan"]
# Mercados de un solo tiempo (oct-2026): "gol en la mitad" y "+/-1.5 en la mitad".
HALF_MARKETS = ["Gol 1er tiempo", "Goles 1er tiempo", "Gol 2do tiempo", "Goles 2do tiempo"]
MARKETS = FULL_MARKETS + HALF_MARKETS


def candidate_legs(home_win: float, draw: float, away_win: float,
                   over_2_5: float, btts: float,
                   half_xg: tuple | None = None) -> list[tuple[str, str, float]]:
    """(mercado, pick, probabilidad_del_modelo 0-1) — el lado más probable de
    cada mercado. Lo comparten la calibración y el armado de combinadas, así
    se calibra exactamente el mismo tipo de pick que después se recomienda.
    Entradas en porcentaje (0-100), como vienen de matchday_predictions_df.
    `half_xg`: (local_1T, visita_1T, local_2T, visita_2T) — si viene completo,
    se agregan los mercados de un solo tiempo."""
    h, d, a = home_win / 100, draw / 100, away_win / 100
    legs = []
    resultado = max((("Local", h), ("Empate", d), ("Visitante", a)), key=lambda x: x[1])
    legs.append(("Resultado", *resultado))
    doble = max((("1X (local o empate)", h + d), ("X2 (empate o visitante)", d + a)), key=lambda x: x[1])
    legs.append(("Doble oportunidad", *doble))
    if pd.notna(over_2_5):
        o = over_2_5 / 100
        legs.append(("Goles", "+2.5 goles", o) if o >= 0.5 else ("Goles", "-2.5 goles", 1 - o))
    if pd.notna(btts):
        b = btts / 100
        legs.append(("Ambos anotan", "Sí", b) if b >= 0.5 else ("Ambos anotan", "No", 1 - b))
    if half_xg is not None and all(pd.notna(x) for x in half_xg):
        from .half_strength import half_probabilities
        for mitad, (local, visita) in (("1er tiempo", half_xg[:2]), ("2do tiempo", half_xg[2:])):
            probs = half_probabilities(float(local), float(visita))
            g, o = probs["p_goal"], probs["p_over_1_5"]
            legs.append((f"Gol {mitad}", "Sí", g) if g >= 0.5 else (f"Gol {mitad}", "No", 1 - g))
            legs.append((f"Goles {mitad}", "+1.5", o) if o >= 0.5 else (f"Goles {mitad}", "-1.5", 1 - o))
    return legs


def leg_won(market: str, pick: str, home_goals: int, away_goals: int,
            ht_home: int | None = None, ht_away: int | None = None) -> bool | None:
    """Si el pick ganó. Los mercados de un solo tiempo necesitan el marcador al
    descanso: sin él devuelven None (no se puede resolver)."""
    if market in HALF_MARKETS:
        if ht_home is None or ht_away is None or ht_home != ht_home or ht_away != ht_away:
            return None
        goles = ht_home + ht_away if "1er" in market else (home_goals - ht_home) + (away_goals - ht_away)
        if market.startswith("Gol "):
            return (goles > 0) == (pick == "Sí")
        return (goles > 1.5) == pick.startswith("+")
    if market == "Resultado":
        real = "Local" if home_goals > away_goals else "Visitante" if home_goals < away_goals else "Empate"
        return pick == real
    if market == "Doble oportunidad":
        if pick.startswith("1X"):
            return home_goals >= away_goals
        return away_goals >= home_goals
    if market == "Goles":
        return (home_goals + away_goals > 2.5) == pick.startswith("+")
    if market == "Ambos anotan":
        return (home_goals > 0 and away_goals > 0) == (pick == "Sí")
    raise ValueError(f"Mercado desconocido: {market}")


def resolved_rows(log: pd.DataFrame) -> pd.DataFrame:
    """Predicciones del modelo ya resueltas, con goles reales numéricos."""
    if log.empty:
        return log
    r = log[(log["status"] == "resolved") & (~log["source"].isin(EXCLUDED_SOURCES))].copy()
    for col in ("actual_home_goals", "actual_away_goals", "pred_home_pct", "pred_draw_pct",
                "pred_away_pct", "over_2_5_pct", "btts_pct") + HALF_COLUMNS:
        r[col] = pd.to_numeric(r[col], errors="coerce") if col in r else float("nan")
    return r.dropna(subset=["actual_home_goals", "actual_away_goals", "pred_home_pct"])


HALF_COLUMNS = ("h1_home_xg", "h1_away_xg", "h2_home_xg", "h2_away_xg",
                "actual_ht_home_goals", "actual_ht_away_goals")


def row_legs(r) -> list[tuple[str, str, float, bool]]:
    """(mercado, pick, prob_modelo, ganó) de una fila resuelta del log (o del
    backtest por tiempo, que usa el mismo esquema). Omite lo que no se puede
    resolver (mercados por tiempo sin marcador al descanso)."""
    hg, ag = int(r.actual_home_goals), int(r.actual_away_goals)
    half = (r.h1_home_xg, r.h1_away_xg, r.h2_home_xg, r.h2_away_xg)
    salida = []
    for mercado, pick, p in candidate_legs(r.pred_home_pct, r.pred_draw_pct, r.pred_away_pct,
                                           r.over_2_5_pct, r.btts_pct, half):
        gano = leg_won(mercado, pick, hg, ag, r.actual_ht_home_goals, r.actual_ht_away_goals)
        if gano is not None:
            salida.append((mercado, pick, p, gano))
    return salida


def _bin(p: float) -> int:
    for i in range(len(BINS) - 1):
        if BINS[i] <= p < BINS[i + 1]:
            return i
    return len(BINS) - 2


@dataclass
class Calibrator:
    # {(mercado, tramo): [n, aciertos]}
    table: dict[tuple[str, int], list[int]] = field(default_factory=dict)
    # {(competición, mercado): [aciertos, esperados_según_calibración_global]}
    by_competition: dict[tuple[str, str], list[float]] = field(default_factory=dict)

    @classmethod
    def from_log(cls, log: pd.DataFrame, half_backtest: pd.DataFrame | None = None) -> "Calibrator":
        """`half_backtest`: filas del backtest por tiempo (half_backtest.py) —
        solo aportan a los mercados de UN SOLO TIEMPO, para no mezclar sus
        predicciones retroactivas en la calibración de los de partido completo."""
        filas = resolved_rows(log)
        picks = []
        tabla: dict[tuple[str, int], list[int]] = {}
        fuentes = [(filas, None)]
        if half_backtest is not None and not half_backtest.empty:
            fuentes.append((resolved_rows(half_backtest), set(HALF_MARKETS)))
        for datos, solo in fuentes:
            for r in datos.itertuples():
                for mercado, pick, p, gano in row_legs(r):
                    if solo is not None and mercado not in solo:
                        continue
                    picks.append((r.competition, mercado, p, gano))
                    celda = tabla.setdefault((mercado, _bin(p)), [0, 0])
                    celda[0] += 1
                    celda[1] += int(gano)
        calibrador = cls(tabla)
        por_comp: dict[tuple[str, str], list[float]] = {}
        for comp, mercado, p, gano in picks:
            celda = por_comp.setdefault((comp, mercado), [0.0, 0.0])
            celda[0] += float(gano)
            celda[1] += calibrador._global(mercado, p)
        calibrador.by_competition = por_comp
        return calibrador

    def _global(self, market: str, p_model: float) -> float:
        n, aciertos = self.table.get((market, _bin(p_model)), (0, 0))
        return (aciertos + PRIOR_STRENGTH * p_model) / (n + PRIOR_STRENGTH)

    def competition_factor(self, competition: str | None, market: str) -> float:
        """<=1 siempre: se acepta la evidencia de que el modelo rinde PEOR en
        una competición, pero nunca se suben probabilidades por rachas buenas
        de muestras chicas. Comparado con parlay_backtest (oct-2026, 16
        jornadas): retorno promedio 1.13 (solo baja) vs 1.16 (sin factor) vs
        1.02 (factor completo) — empate dentro del ruido, pero es la única
        variante que evita el valor ficticio con cuotas reales en Nations
        League (+100% esperado sin factor, +5% con él)."""
        aciertos, esperados = self.by_competition.get((competition, market), (0.0, 0.0))
        return min(1.0, (aciertos + COMPETITION_PRIOR) / (esperados + COMPETITION_PRIOR))

    def calibrate(self, market: str, p_model: float, competition: str | None = None) -> float:
        p = self._global(market, p_model) * self.competition_factor(competition, market)
        return min(max(p, 0.01), 0.99)

    def sample_size(self, market: str, p_model: float) -> int:
        return self.table.get((market, _bin(p_model)), (0, 0))[0]

    def market_sample(self, market: str) -> int:
        """Picks resueltos de ese mercado, en todos los tramos."""
        return sum(n for (m, _), (n, _a) in self.table.items() if m == market)

    def summary(self) -> pd.DataFrame:
        filas = []
        for (mercado, i), (n, aciertos) in sorted(self.table.items()):
            filas.append({"Mercado": mercado, "Tramo del modelo": f"{BINS[i]:.0%}–{min(BINS[i + 1], 1):.0%}",
                          "Casos": n, "Acierto real": aciertos / n if n else None})
        return pd.DataFrame(filas)
