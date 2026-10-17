"""
Datos de la pestaña Reportes (pura pandas, sin Streamlit): aciertos de las
predicciones del modelo por día, liga, equipo y mercado, y resultado de los
boletos de Tips.

"Esperado" = la probabilidad que el modelo le daba a su pick (ej. 68% al
local). Si a la larga el acierto real queda por encima, el modelo es más
fuerte de lo que cree en ese mercado/liga; por debajo, se sobreestima.
Corners/faltas/tarjetas no guardan probabilidad: su pick es "más o menos que
una línea" puesta en el promedio, así que se comparan contra 50%.
"""
from __future__ import annotations

import pandas as pd

from .timezones import to_bogota

# (mercado, columna de acierto en predictions_log)
MARKETS = [
    ("Resultado (1X2)", "hit"),
    ("Goles +/-2.5", "hit_over_2_5"),
    ("Ambos anotan", "hit_btts"),
    ("Corners", "hit_corners"),
    ("Faltas", "hit_faltas"),
    ("Tarjetas", "hit_tarjetas"),
]
STAT_MARKET_BASELINE = 0.5


def _fecha(utc_date, logged_at) -> object:
    for valor in (utc_date, logged_at):
        if isinstance(valor, str) and valor:
            local = to_bogota(valor[:19] + "Z" if not valor.endswith("Z") else valor)
            if local is not None:
                return local.date()
    return None


def _as_bool(serie: pd.Series) -> pd.Series:
    return serie.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0") if pd.notna(v) else None)


def predictions_long(log: pd.DataFrame, include_backtest: bool = True) -> pd.DataFrame:
    """Una fila por (predicción resuelta, mercado) con acierto y probabilidad esperada."""
    if log.empty:
        return pd.DataFrame()
    r = log[log["status"] == "resolved"].copy()
    if not include_backtest:
        r = r[~r["source"].astype(str).str.startswith("backtest")]
    if r.empty:
        return pd.DataFrame()
    r["fecha"] = [_fecha(u, l) for u, l in zip(r.get("utc_date", [None] * len(r)), r["logged_at"])]
    p1x2 = r[["pred_home_pct", "pred_draw_pct", "pred_away_pct"]].max(axis=1) / 100
    over = r["over_2_5_pct"] / 100
    btts = r["btts_pct"] / 100
    esperado = {
        "hit": p1x2,
        "hit_over_2_5": over.where(over >= 0.5, 1 - over),
        "hit_btts": btts.where(btts >= 0.5, 1 - btts),
    }
    filas = []
    for mercado, col in MARKETS:
        if col not in r:
            continue
        acierto = _as_bool(r[col])
        sub = r[acierto.notna()]
        if sub.empty:
            continue
        filas.append(pd.DataFrame({
            "fecha": sub["fecha"], "competition": sub["competition"],
            "home_team": sub["home_team"], "away_team": sub["away_team"],
            "mercado": mercado, "acierto": acierto[sub.index].astype(bool),
            "esperado": esperado[col][sub.index] if col in esperado else STAT_MARKET_BASELINE,
            "source": sub["source"],
        }))
    return pd.concat(filas, ignore_index=True) if filas else pd.DataFrame()


def by_team(long: pd.DataFrame) -> pd.DataFrame:
    """Cada predicción cuenta para los DOS equipos del partido."""
    if long.empty:
        return long
    return pd.concat([long.assign(equipo=long["home_team"]), long.assign(equipo=long["away_team"])],
                     ignore_index=True)


def summarize(long: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()
    t = long.groupby(by, dropna=False).agg(
        predicciones=("acierto", "size"), aciertos=("acierto", "sum"), esperado=("esperado", "mean"))
    t["acierto_pct"] = t["aciertos"] / t["predicciones"] * 100
    t["esperado_pct"] = t["esperado"] * 100
    t["diferencia"] = t["acierto_pct"] - t["esperado_pct"]
    return t.drop(columns="esperado").reset_index()


# --- Tips ---

def tips_tickets(tlog: pd.DataFrame) -> pd.DataFrame:
    """Un renglón por boleto: picks (uno por línea), cuota, monto y resultado."""
    if tlog.empty:
        return pd.DataFrame()
    filas = []
    for tid, legs in tlog.groupby("ticket_id", sort=False):
        primero = legs.iloc[0]
        picks = "\n".join(
            f"{'✔' if l.leg_result == 'acierto' else '✘' if l.leg_result == 'fallo' else '·'} "
            f"{l.home_team} vs {l.away_team}: {l.pick}" + (f" ({l.cuota:.2f})" if pd.notna(l.cuota) else "")
            for l in legs.itertuples())
        filas.append({
            "fecha": primero["fecha"], "plan": primero["plan"], "tipo": primero["tipo"], "picks": picks,
            "ligas": ", ".join(sorted(set(legs["competition"].dropna().astype(str)))),
            "cuota": primero["ticket_odds"], "prob": primero["ticket_prob"], "monto": primero["stake"],
            "resultado": primero["ticket_result"], "ganancia": primero["ticket_profit"],
        })
    return pd.DataFrame(filas)


def tips_by_plan(tickets: pd.DataFrame) -> pd.DataFrame:
    res = tickets[tickets["resultado"].isin(["ganada", "perdida"])] if not tickets.empty else tickets
    if res.empty:
        return pd.DataFrame()
    t = res.groupby("plan").agg(
        boletos=("resultado", "size"), ganados=("resultado", lambda s: (s == "ganada").sum()),
        apostado=("monto", lambda s: pd.to_numeric(s, errors="coerce").sum()),
        ganancia=("ganancia", lambda s: pd.to_numeric(s, errors="coerce").sum()),
    )
    t["ganados_pct"] = t["ganados"] / t["boletos"] * 100
    t["roi_pct"] = (t["ganancia"] / t["apostado"] * 100).where(t["apostado"] > 0)
    return t.reset_index()


def tips_legs_long(tlog: pd.DataFrame) -> pd.DataFrame:
    """Picks de Tips ya resueltos, en el mismo formato que predictions_long."""
    if tlog.empty:
        return pd.DataFrame()
    r = tlog[tlog["leg_result"].isin(["acierto", "fallo"])]
    if r.empty:
        return pd.DataFrame()
    r = r.drop_duplicates(subset=["fecha", "home_team", "away_team", "market", "pick"])
    return pd.DataFrame({
        "fecha": pd.to_datetime(r["fecha"]).dt.date, "competition": r["competition"],
        "home_team": r["home_team"], "away_team": r["away_team"],
        "mercado": "Tips: " + r["market"].astype(str), "acierto": r["leg_result"] == "acierto",
        "esperado": pd.to_numeric(r["prob_ajustada"], errors="coerce"), "source": "tips",
    })
