"""
Historial de las combinaciones que genera la pestaña Tips (pedido del
usuario, oct-2026) y su resultado al terminar cada día — base de la pestaña
Reportes. Versionado en git: data/tracking/tips_log.csv.

Una fila por PICK (pierna) de cada boleto, con los datos del boleto repetidos
(id, plan, cuota, monto). Se guardan:
  - los planes de "Plan del día" (Conservador / Equilibrado / Agresivo),
  - las "Combinadas del día — todas las ligas",
  - los tips en vivo con valor ("En vivo").
Un boleto se identifica por día + plan + picks (sin la cuota): si la cuota se
mueve, no se duplica; si cambian los picks, es un boleto nuevo.

resolve(): marca acierto/fallo de cada pick con el marcador final de
365scores (gratis; 1 consulta por competición y día) y, con eso, si el boleto
se ganó o se perdió y su ganancia neta.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config, scores365
from .odds_api import find_event

PATH = config.ROOT_DIR / "data" / "tracking" / "tips_log.csv"
COLUMNS = [
    "ticket_id", "fecha", "generated_at", "origin", "plan", "boleto", "tipo",
    "ticket_odds", "ticket_prob", "stake", "budget",
    "competition", "competition_name", "utc_date", "home_team", "away_team",
    "market", "pick", "prob_modelo", "prob_ajustada", "cuota", "en_vivo",
    "leg_result", "home_goals", "away_goals", "ticket_result", "ticket_profit", "resolved_at",
]
PENDING, HIT, MISS, NO_DATA = "pendiente", "acierto", "fallo", "sin resultado"
WON, LOST = "ganada", "perdida"
# Un partido se da por terminado (y se busca su marcador) pasado este tiempo
# desde el inicio; si a los NO_DATA_AFTER_DAYS no apareció, queda "sin resultado".
FINISHED_AFTER = timedelta(minutes=150)
NO_DATA_AFTER_DAYS = 3


def load() -> pd.DataFrame:
    if not PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(PATH)


def _save(df: pd.DataFrame) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    df.reindex(columns=COLUMNS).to_csv(PATH, index=False, encoding="utf-8")


def _ticket_id(fecha: str, plan: str, legs: list[dict]) -> str:
    firma = "|".join(sorted(f"{l['home_team']}~{l['away_team']}~{l['market']}~{l['pick']}" for l in legs))
    return hashlib.sha1(f"{fecha}|{plan}|{firma}".encode("utf-8")).hexdigest()[:12]


def record(fecha: str, plan: str, tickets: list[dict], origin: str) -> int:
    """`tickets`: [{"tipo", "odds", "prob", "stake", "budget", "legs": [ {competition,
    competition_name, utc_date, home_team, away_team, market, pick, prob_modelo,
    prob_ajustada, cuota, en_vivo} ]}]. Devuelve cuántos boletos nuevos guardó.
    Nunca lanza: guardar el historial no debe tumbar la pestaña Tips."""
    try:
        hist = load()
        existentes = set(hist["ticket_id"]) if not hist.empty else set()
        ahora = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        filas, nuevos = [], 0
        for n, t in enumerate(tickets, 1):
            legs = t.get("legs") or []
            if not legs:
                continue
            tid = _ticket_id(fecha, plan, legs)
            if tid in existentes:
                continue
            existentes.add(tid)
            nuevos += 1
            for leg in legs:
                filas.append({
                    "ticket_id": tid, "fecha": fecha, "generated_at": ahora, "origin": origin,
                    "plan": plan, "boleto": n, "tipo": t.get("tipo"),
                    "ticket_odds": round(float(t["odds"]), 3) if t.get("odds") else None,
                    "ticket_prob": round(float(t["prob"]), 4) if t.get("prob") is not None else None,
                    "stake": t.get("stake"), "budget": t.get("budget"),
                    **{k: leg.get(k) for k in ("competition", "competition_name", "utc_date", "home_team",
                                               "away_team", "market", "pick", "en_vivo")},
                    "prob_modelo": _round(leg.get("prob_modelo")), "prob_ajustada": _round(leg.get("prob_ajustada")),
                    "cuota": _round(leg.get("cuota"), 3),
                    "leg_result": PENDING, "ticket_result": PENDING,
                })
        if filas:
            _save(pd.concat([hist, pd.DataFrame(filas)], ignore_index=True) if not hist.empty
                  else pd.DataFrame(filas))
        return nuevos
    except Exception:
        return 0


def _round(v, nd: int = 4):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v != v else round(v, nd)


# --- Conversión desde lo que arma la pestaña Tips ---

def _leg_from_pick(r) -> dict:
    return {
        "competition": r.get("competition"), "competition_name": r.get("competition_name"),
        "utc_date": r.get("utc_date"), "home_team": r.get("home_team"), "away_team": r.get("away_team"),
        "market": r.get("mercado"), "pick": r.get("pick"), "prob_modelo": r.get("prob_modelo"),
        "prob_ajustada": r.get("prob_ajustada"), "cuota": r.get("cuota"), "en_vivo": r.get("en_vivo"),
    }


def tickets_from_plan(plan, picks: pd.DataFrame, budget: float) -> list[dict]:
    """Boletos de un daily_plan.Plan (picks = el DataFrame evaluado)."""
    return [{"tipo": t.kind, "odds": t.odds, "prob": t.prob, "stake": t.stake, "budget": budget,
             "legs": [_leg_from_pick(picks.loc[i]) for i in t.picks]} for t in plan.tickets]


def tickets_from_parlays(parlays: list, day_predictions: pd.DataFrame, bankroll: float) -> list[dict]:
    """Combinadas de parlay_builder; competición e inicio salen de las predicciones del día."""
    por_partido = {(r.home_team, r.away_team): r for r in day_predictions.itertuples()}
    boletos = []
    for p in parlays:
        legs = []
        for l in p.legs:
            fila = por_partido.get((l.home_team, l.away_team))
            legs.append({
                "competition": getattr(fila, "competition", None), "competition_name": getattr(fila, "competition_name", None),
                "utc_date": getattr(fila, "utc_date", None), "home_team": l.home_team, "away_team": l.away_team,
                "market": l.market, "pick": l.pick, "prob_modelo": l.model_probability,
                "prob_ajustada": l.probability, "cuota": l.book_odds,
            })
        boletos.append({"tipo": f"Combinada ({len(legs)})", "odds": p.estimated_odds, "prob": p.combined_probability,
                        "stake": p.stake(bankroll), "budget": None, "legs": legs})
    return boletos


def tickets_from_singles(evaluated: pd.DataFrame) -> list[dict]:
    """Sencillas con estado "Apostar" (tips en vivo)."""
    apostar = evaluated[evaluated["estado"] == "Apostar"] if not evaluated.empty else evaluated
    return [{"tipo": "Sencilla", "odds": r.get("cuota"), "prob": r.get("prob_ajustada"), "stake": r.get("monto"),
             "budget": None, "legs": [_leg_from_pick(r)]} for _, r in apostar.iterrows()]


# --- Resolución ---

def leg_outcome(market: str, pick: str, hg: int, ag: int) -> bool | None:
    """¿Acertó el pick con el marcador final? None si el mercado no se reconoce."""
    pick = str(pick or "").strip()
    if market == "Resultado":
        return {"Local": hg > ag, "Empate": hg == ag, "Visitante": hg < ag}.get(pick)
    if market == "Doble oportunidad":
        if pick.startswith("1X"):
            return hg >= ag
        if pick.startswith("X2"):
            return hg <= ag
        if pick.startswith("12"):
            return hg != ag
        return None
    if market == "Goles":
        if pick.startswith(("Más", "+")):
            return hg + ag > 2.5
        if pick.startswith(("Menos", "-")):
            return hg + ag < 2.5
        return None
    if market == "Ambos anotan":
        ambos = hg > 0 and ag > 0
        return ambos if pick == "Sí" else (not ambos) if pick == "No" else None
    return None


def _final_scores(code: str, dia: str) -> list[dict]:
    """Partidos TERMINADOS de esa competición y día (UTC), como "eventos" para find_event."""
    desde = f"{dia}T00:00:00Z"
    hasta = (datetime.strptime(dia, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
    eventos = []
    for g in scores365.games(code, desde, hasta):
        if g.get("statusGroup") != scores365.STATUS_FINISHED:
            continue
        hg, ag = scores365._score(g.get("homeCompetitor")), scores365._score(g.get("awayCompetitor"))
        if hg is None or ag is None:
            continue
        eventos.append({"home_team": (g.get("homeCompetitor") or {}).get("name"),
                        "away_team": (g.get("awayCompetitor") or {}).get("name"),
                        "commence_time": scores365._utc(g.get("startTime")), "hg": hg, "ag": ag})
    return eventos


def resolve(now: datetime | None = None) -> int:
    """Resuelve los picks pendientes cuyos partidos ya terminaron. Devuelve
    cuántos picks se resolvieron. Nunca lanza."""
    try:
        df = load()
        if df.empty:
            return 0
        now = now or datetime.now(timezone.utc)
        inicio = pd.to_datetime(df["utc_date"], utc=True, errors="coerce")
        pendientes = df.index[(df["leg_result"] == PENDING) & inicio.notna() & (inicio + FINISHED_AFTER < now)]
        if len(pendientes) == 0:
            return 0
        resueltos = 0
        cache: dict = {}
        for i in pendientes:
            r = df.loc[i]
            dia = inicio[i].strftime("%Y-%m-%d")
            clave = (r["competition"], dia)
            if clave not in cache:
                cache[clave] = _final_scores(r["competition"], dia) if r["competition"] in scores365.COMPETITION_IDS else []
            ev = find_event(cache[clave], r["home_team"], r["away_team"], r["utc_date"]) if cache[clave] else None
            if ev:
                gano = leg_outcome(r["market"], r["pick"], ev["hg"], ev["ag"])
                df.at[i, "home_goals"], df.at[i, "away_goals"] = ev["hg"], ev["ag"]
                df.at[i, "leg_result"] = NO_DATA if gano is None else (HIT if gano else MISS)
                resueltos += 1
            elif now - inicio[i] > timedelta(days=NO_DATA_AFTER_DAYS):
                df.at[i, "leg_result"] = NO_DATA
                resueltos += 1
        _resolve_tickets(df, now)
        _save(df)
        return resueltos
    except Exception:
        return 0


def _resolve_tickets(df: pd.DataFrame, now: datetime) -> None:
    for col in ("ticket_result", "resolved_at", "leg_result"):
        df[col] = df[col].astype(object)  # columnas vacías se leen como float
    for tid, legs in df.groupby("ticket_id"):
        if (legs["ticket_result"] != PENDING).all():
            continue
        resultados = set(legs["leg_result"])
        if MISS in resultados:
            estado = LOST
        elif resultados == {HIT}:
            estado = WON
        elif PENDING in resultados:
            continue
        else:
            estado = NO_DATA
        stake = pd.to_numeric(legs["stake"], errors="coerce").iloc[0]
        cuota = pd.to_numeric(legs["ticket_odds"], errors="coerce").iloc[0]
        ganancia = None
        if stake == stake and stake:  # hay monto
            ganancia = round(stake * (cuota - 1)) if estado == WON else -stake if estado == LOST else 0
        df.loc[legs.index, "ticket_result"] = estado
        df.loc[legs.index, "ticket_profit"] = ganancia
        df.loc[legs.index, "resolved_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
