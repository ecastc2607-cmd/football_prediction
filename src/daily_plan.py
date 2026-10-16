"""
Plan del día: concentrar un PRESUPUESTO diario en hasta MAX_TICKETS apuestas
(sencillas o combinadas), en vez de repartir ¼ de Kelly de la banca en muchas
apuestas chicas (pedido del usuario, oct-2026: "apostar 1% en cada partido para
ganar $600 es muy engorroso").

Solo se usan picks que YA pasaron la regla de valor de value_tips.evaluate
(estado "Apostar"): esto cambia cuánto y en cuántos boletos, no qué se apuesta.

Tres planes para comparar (mismo presupuesto):
  - Conservador: hasta 3 sencillas.
  - Equilibrado: las 2 mejores sencillas + 1 combinada de las 3 mejores.
  - Agresivo: la mejor sencilla + 1 combinada de las 3 mejores.
El presupuesto se reparte entre los boletos en proporción a su Kelly (más
monto donde más valor hay), redondeado a STAKE_ROUNDING. Las métricas de riesgo
se calculan enumerando todos los resultados posibles de los picks (partidos
independientes entre sí; un mismo pick puede estar en una sencilla y en la
combinada, y eso se respeta).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import pandas as pd

MAX_TICKETS = 3
STAKE_ROUNDING = 500  # COP


@dataclass
class Ticket:
    kind: str  # "Sencilla" / "Combinada (3)"
    picks: list[int]  # índices en el DataFrame de picks
    odds: float
    prob: float
    stake: int = 0

    @property
    def ev(self) -> float:
        return self.prob * self.odds - 1

    @property
    def kelly(self) -> float:
        return max(self.ev / (self.odds - 1), 0.0) if self.odds > 1 else 0.0


@dataclass
class Plan:
    name: str
    description: str
    tickets: list[Ticket] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _ticket(picks: pd.DataFrame, idx: list[int]) -> Ticket:
    filas = picks.loc[idx]
    return Ticket(
        kind="Sencilla" if len(idx) == 1 else f"Combinada ({len(idx)})",
        picks=list(idx),
        odds=float(filas["cuota"].astype(float).prod()),
        prob=float(filas["prob_ajustada"].prod()),
    )


def _allocate(tickets: list[Ticket], budget: float) -> list[Ticket]:
    """Reparte el presupuesto en proporción al Kelly de cada boleto."""
    tickets = [t for t in tickets if t.kelly > 0]
    total = sum(t.kelly for t in tickets)
    if not tickets or total <= 0:
        return []
    for t in tickets:
        t.stake = int(budget * t.kelly / total // STAKE_ROUNDING * STAKE_ROUNDING)
    sobrante = int(budget) - sum(t.stake for t in tickets)
    if sobrante >= STAKE_ROUNDING:  # lo que quedó por redondeo, al de más valor
        max(tickets, key=lambda t: t.kelly).stake += sobrante // STAKE_ROUNDING * STAKE_ROUNDING
    return [t for t in tickets if t.stake > 0]


def _metrics(tickets: list[Ticket], picks: pd.DataFrame) -> dict:
    usados = sorted({i for t in tickets for i in t.picks})
    prob = {i: float(picks.loc[i, "prob_ajustada"]) for i in usados}
    apostado = sum(t.stake for t in tickets)
    esperado = gana_algo = pierde_todo = 0.0
    maximo = 0.0
    for resultado in product([True, False], repeat=len(usados)):
        acierta = dict(zip(usados, resultado))
        p = 1.0
        for i in usados:
            p *= prob[i] if acierta[i] else 1 - prob[i]
        neto = sum(t.stake * (t.odds - 1) if all(acierta[i] for i in t.picks) else -t.stake for t in tickets)
        esperado += p * neto
        maximo = max(maximo, neto)
        if neto > 0:
            gana_algo += p
        if neto <= -apostado + 1e-9:
            pierde_todo += p
    return {"apostado": apostado, "esperado": esperado, "prob_gana": gana_algo,
            "prob_pierde_todo": pierde_todo, "max_ganancia": maximo}


def build_plans(evaluated: pd.DataFrame, budget: float) -> list[Plan]:
    """`evaluated`: salida de value_tips.evaluate. Devuelve los planes posibles
    (vacío si no hay ningún pick con valor)."""
    if evaluated.empty or "estado" not in evaluated:
        return []
    picks = evaluated[evaluated["estado"] == "Apostar"].sort_values("valor", ascending=False)
    if picks.empty:
        return []
    top = list(picks.index[:MAX_TICKETS])
    planes = [Plan("Conservador", "Hasta 3 sencillas: si una falla, las otras se cobran igual.",
                   [_ticket(picks, [i]) for i in top])]
    if len(top) >= 2:
        planes.append(Plan(
            "Equilibrado", f"Las 2 mejores sencillas + 1 combinada de las {len(top)} mejores.",
            [_ticket(picks, [i]) for i in top[:2]] + [_ticket(picks, top)]))
        planes.append(Plan(
            "Agresivo", f"La mejor sencilla + 1 combinada de las {len(top)} mejores: paga más, acierta menos.",
            [_ticket(picks, [top[0]]), _ticket(picks, top)]))
    for plan in planes:
        plan.tickets = _allocate(plan.tickets, budget)
        plan.metrics = _metrics(plan.tickets, picks) if plan.tickets else {}
    return [p for p in planes if p.tickets]


def ticket_rows(plan: Plan, picks: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for n, t in enumerate(plan.tickets, 1):
        detalle = picks.loc[t.picks]
        filas.append({
            "Boleto": f"{n}. {t.kind}",
            # Un pick por línea: en una combinada larga, todo en una sola línea no se leía.
            "Picks": "\n".join(f"{r.partido}: {r.pick} ({float(r.cuota):.2f})" for r in detalle.itertuples()),
            "Cuota": round(t.odds, 2),
            "Prob.": round(t.prob * 100),
            "Apostar": t.stake,
            "Recibes si acierta": round(t.stake * t.odds),
            "Ganancia neta": round(t.stake * (t.odds - 1)),
        })
    return pd.DataFrame(filas)
