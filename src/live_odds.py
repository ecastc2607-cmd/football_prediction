"""
Fotos de cuotas EN VIVO (365scores / BWIN) junto a la probabilidad del modelo
en ese mismo minuto — base para evaluar después una estrategia de apuestas en
vivo (pedido del usuario, oct-2026: guardar el historial en el repo).

Se guarda una fila solo cuando la cuota de ese partido CAMBIÓ desde su última
foto (la pestaña "En vivo" la consulta cada pocos minutos). Archivo versionado:
data/tracking/live_odds_snapshots.csv.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .odds_api import PRICE_KEYS

PATH = config.ROOT_DIR / "data" / "tracking" / "live_odds_snapshots.csv"
PRICE_COLS = [f"co_{k}" for k in PRICE_KEYS]
KEEP = ["fetched_at", "competition", "event_id", "commence_time", "home_team", "away_team",
        "game_minute", "home_score", "away_score", *PRICE_COLS,
        "model_home", "model_draw", "model_away", "model_home_xg_rem", "model_away_xg_rem"]


def _same(a, b) -> bool:
    a_nan = a is None or (isinstance(a, float) and a != a)
    b_nan = b is None or (isinstance(b, float) and b != b)
    return (a_nan and b_nan) or (not a_nan and not b_nan and float(a) == float(b))


def record(rows: list[dict]) -> int:
    """Agrega las filas cuya cuota cambió (o el marcador). Devuelve cuántas.
    Nunca lanza: si no se puede escribir, no se guarda y listo."""
    if not rows:
        return 0
    try:
        hist = pd.read_csv(PATH) if PATH.exists() else pd.DataFrame(columns=KEEP)
        ultimas = {r["event_id"]: r for r in
                   hist.sort_values("fetched_at").drop_duplicates("event_id", keep="last").to_dict("records")}
        nuevas = []
        for r in rows:
            previa = ultimas.get(r.get("event_id"))
            cambia = previa is None or not all(
                _same(r.get(c), previa.get(c)) for c in [*PRICE_COLS, "home_score", "away_score"])
            if cambia:
                nuevas.append({k: r.get(k) for k in KEEP})
        if not nuevas:
            return 0
        PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([hist, pd.DataFrame(nuevas)], ignore_index=True).to_csv(PATH, index=False, encoding="utf-8")
        return len(nuevas)
    except Exception:
        return 0
