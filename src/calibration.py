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
EXCLUDED_SOURCES = {"dashboard_v1_qualitative"}  # predicciones "a ojo", no del modelo

MARKETS = ["Resultado", "Doble oportunidad", "Goles", "Ambos anotan"]


def candidate_legs(home_win: float, draw: float, away_win: float,
                   over_2_5: float, btts: float) -> list[tuple[str, str, float]]:
    """(mercado, pick, probabilidad_del_modelo 0-1) — el lado más probable de
    cada mercado. Lo comparten la calibración y el armado de combinadas, así
    se calibra exactamente el mismo tipo de pick que después se recomienda.
    Entradas en porcentaje (0-100), como vienen de matchday_predictions_df."""
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
    return legs


def leg_won(market: str, pick: str, home_goals: int, away_goals: int) -> bool:
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
                "pred_away_pct", "over_2_5_pct", "btts_pct"):
        r[col] = pd.to_numeric(r[col], errors="coerce")
    return r.dropna(subset=["actual_home_goals", "actual_away_goals", "pred_home_pct"])


def _bin(p: float) -> int:
    for i in range(len(BINS) - 1):
        if BINS[i] <= p < BINS[i + 1]:
            return i
    return len(BINS) - 2


@dataclass
class Calibrator:
    # {(mercado, tramo): [n, aciertos]}
    table: dict[tuple[str, int], list[int]] = field(default_factory=dict)

    @classmethod
    def from_log(cls, log: pd.DataFrame) -> "Calibrator":
        tabla: dict[tuple[str, int], list[int]] = {}
        for r in resolved_rows(log).itertuples():
            hg, ag = int(r.actual_home_goals), int(r.actual_away_goals)
            for mercado, pick, p in candidate_legs(r.pred_home_pct, r.pred_draw_pct, r.pred_away_pct,
                                                   r.over_2_5_pct, r.btts_pct):
                celda = tabla.setdefault((mercado, _bin(p)), [0, 0])
                celda[0] += 1
                celda[1] += int(leg_won(mercado, pick, hg, ag))
        return cls(tabla)

    def calibrate(self, market: str, p_model: float) -> float:
        n, aciertos = self.table.get((market, _bin(p_model)), (0, 0))
        return (aciertos + PRIOR_STRENGTH * p_model) / (n + PRIOR_STRENGTH)

    def sample_size(self, market: str, p_model: float) -> int:
        return self.table.get((market, _bin(p_model)), (0, 0))[0]

    def summary(self) -> pd.DataFrame:
        filas = []
        for (mercado, i), (n, aciertos) in sorted(self.table.items()):
            filas.append({"Mercado": mercado, "Tramo del modelo": f"{BINS[i]:.0%}–{min(BINS[i + 1], 1):.0%}",
                          "Casos": n, "Acierto real": aciertos / n if n else None})
        return pd.DataFrame(filas)
