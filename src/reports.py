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
    ("Doble oportunidad", "hit_dc"),  # derivada: ver _double_chance
    ("Goles +/-2.5", "hit_over_2_5"),
    ("Ambos anotan", "hit_btts"),
    ("Corners", "hit_corners"),
    ("Faltas", "hit_faltas"),
    ("Tarjetas", "hit_tarjetas"),
]
STAT_MARKET_BASELINE = 0.5
# Tipo de predicción (segmento de Reportes): las del modelo, o los picks que
# entraron en boletos de Tips según su plan.
TIPO_MODELO = "Modelo"
TIPOS = [TIPO_MODELO, "Conservador", "Equilibrado", "Agresivo", "Combinadas del día", "En vivo"]
# Nombre del mercado en tips_log -> el mismo nombre que usan las del modelo.
TIPS_MARKET_NAMES = {"Resultado": "Resultado (1X2)", "Goles": "Goles +/-2.5"}
DAY_MARKET_NAMES = {"Resultado": "1X2", "Goles": "Goles 2.5"}


def plan_tipo(plan: str) -> str:
    plan = str(plan or "")
    if plan.startswith("Combinada del día"):
        return "Combinadas del día"
    return plan if plan in TIPOS else plan
_SIDE = {"H": "Local", "D": "Empate", "A": "Visitante"}


def _fecha(utc_date, logged_at) -> object:
    for valor in (utc_date, logged_at):
        if isinstance(valor, str) and valor:
            local = to_bogota(valor[:19] + "Z" if not valor.endswith("Z") else valor)
            if local is not None:
                return local.date()
    return None


def _as_bool(serie: pd.Series) -> pd.Series:
    return serie.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0") if pd.notna(v) else None)


def _double_chance(r: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(pick, prob, acierto) de doble oportunidad: la más probable entre 1X y
    X2 según el 1X2 del modelo. No se logueaba aparte, pero sale exacta de lo
    que sí se guardó (probabilidades 1X2 + resultado real), también hacia atrás."""
    p1x = (r["pred_home_pct"] + r["pred_draw_pct"]) / 100
    px2 = (r["pred_away_pct"] + r["pred_draw_pct"]) / 100
    pick = pd.Series("1X", index=r.index).where(p1x >= px2, "X2")
    prob = p1x.where(p1x >= px2, px2)
    real = r.get("actual_result")
    acierto = pd.Series(None, index=r.index, dtype=object)
    if real is not None:
        ok = real.isin(["H", "D", "A"]) & p1x.notna()
        acierto[ok] = ((pick == "1X") & real.isin(["H", "D"]) | (pick == "X2") & real.isin(["A", "D"]))[ok]
    return pick, prob, acierto


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
    pick_dc, prob_dc, r["hit_dc"] = _double_chance(r)
    over = r["over_2_5_pct"] / 100
    btts = r["btts_pct"] / 100
    # El pick que hizo el modelo en cada mercado (para entender un ✔/✘: ej.
    # "Ambos anotan: No" y quedó 1-1 = fallo).
    picks = {
        "hit": r["favored_side"].map(_SIDE),
        "hit_dc": pick_dc,
        "hit_over_2_5": over.map(lambda o: None if pd.isna(o) else ("+2.5" if o >= 0.5 else "-2.5")),
        "hit_btts": btts.map(lambda b: None if pd.isna(b) else ("Sí" if b >= 0.5 else "No")),
    }
    for m in ("corners", "faltas", "tarjetas"):
        if f"pred_{m}_line" in r:
            picks[f"hit_{m}"] = [
                f"{'Más' if lado == 'Sí' else 'Menos'} de {linea:g}" if pd.notna(lado) and pd.notna(linea) else None
                for lado, linea in zip(r[f"pred_{m}_probable"], r[f"pred_{m}_line"])]
    picks = {k: pd.Series(v, index=r.index) for k, v in picks.items()}
    marcador = [f"{int(h)}-{int(a)}" if pd.notna(h) and pd.notna(a) else None
                for h, a in zip(r["actual_home_goals"], r["actual_away_goals"])]
    r["marcador"] = marcador
    esperado = {
        "hit": p1x2,
        "hit_dc": prob_dc,
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
            "source": sub["source"], "tipo": TIPO_MODELO,
            "pick": picks[col][sub.index] if col in picks else None,
            "marcador": sub["marcador"],
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
            f"{'✔' if l.leg_result == 'acierto' else '✘' if l.leg_result == 'fallo' else '↺' if l.leg_result == 'devuelto' else '·'} "
            f"{l.home_team} vs {l.away_team}: {l.pick}" + (f" ({l.cuota:.2f})" if pd.notna(l.cuota) else "")
            for l in legs.itertuples())
        filas.append({
            "fecha": primero["fecha"], "plan": primero["plan"], "boleto": primero.get("boleto"),
            "tipo": primero["tipo"], "picks": picks,
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
    r = r.assign(tipo=r["plan"].map(plan_tipo))
    # Un mismo pick puede estar en varios boletos del mismo plan (sencilla y
    # combinada): cuenta una vez por tipo.
    r = r.drop_duplicates(subset=["tipo", "fecha", "home_team", "away_team", "market", "pick"])
    return pd.DataFrame({
        "fecha": pd.to_datetime(r["fecha"]).dt.date, "competition": r["competition"],
        "home_team": r["home_team"], "away_team": r["away_team"],
        "mercado": r["market"].map(lambda m: TIPS_MARKET_NAMES.get(m, m)), "acierto": r["leg_result"] == "acierto",
        "esperado": pd.to_numeric(r["prob_ajustada"], errors="coerce"), "source": "tips", "tipo": r["tipo"],
        "pick": r["pick"].astype(str).str.split(" \(").str[0],
        "marcador": [f"{int(h)}-{int(a)}" if pd.notna(h) and pd.notna(a) else None
                     for h, a in zip(r["home_goals"], r["away_goals"])],
    })


# --- Partidos de un día: picks de cada partido y si se ganaron ---


def _estado(valor, resuelto: bool) -> str:
    if not resuelto or valor is None or (isinstance(valor, float) and valor != valor):
        return "Pendiente"
    return "Acertado" if str(valor).strip().lower() in ("true", "1", "1.0") else "Fallado"


def day_picks(log: pd.DataFrame, tlog: pd.DataFrame, fecha, tipo: str = TIPO_MODELO) -> pd.DataFrame:
    """Un renglón por (partido, pick) del día `fecha` (Colombia): con tipo
    "Modelo", los picks del modelo (resueltos o no); con un plan de Tips, los
    picks que entraron en boletos de ese plan."""
    filas = []
    if tipo == TIPO_MODELO and not log.empty:
        dias = pd.Series([_fecha(u, l) for u, l in zip(log.get("utc_date", [None] * len(log)), log["logged_at"])],
                         index=log.index)
        d = log[dias == fecha].drop_duplicates(subset=["competition", "home_team", "away_team"])
        if not d.empty:
            dc_pick, dc_prob, dc_hit = _double_chance(d)
            for i, r in d.iterrows():
                resuelto = r["status"] == "resolved"
                marcador = (f"{int(r['actual_home_goals'])}-{int(r['actual_away_goals'])}"
                            if resuelto and pd.notna(r.get("actual_home_goals")) else "vs")
                partido = f"{r['home_team']} {marcador} {r['away_team']}"
                base = {"partido": partido, "competition": r["competition"], "utc_date": r.get("utc_date")}
                probs = {"H": r["pred_home_pct"], "D": r["pred_draw_pct"], "A": r["pred_away_pct"]}
                if pd.notna(r.get("favored_side")):
                    filas.append({**base, "mercado": "1X2", "pick": _SIDE.get(r["favored_side"], r["favored_side"]),
                                  "prob": probs.get(r["favored_side"]), "estado": _estado(r.get("hit"), resuelto)})
                if pd.notna(dc_prob[i]):
                    filas.append({**base, "mercado": "Doble oportunidad", "pick": dc_pick[i], "prob": dc_prob[i] * 100,
                                  "estado": _estado(dc_hit[i], resuelto)})
                if pd.notna(r.get("over_2_5_pct")):
                    over = r["over_2_5_pct"]
                    filas.append({**base, "mercado": "Goles 2.5", "pick": "+2.5" if over >= 50 else "-2.5",
                                  "prob": max(over, 100 - over), "estado": _estado(r.get("hit_over_2_5"), resuelto)})
                if pd.notna(r.get("btts_pct")):
                    b = r["btts_pct"]
                    filas.append({**base, "mercado": "Ambos anotan", "pick": "Sí" if b >= 50 else "No",
                                  "prob": max(b, 100 - b), "estado": _estado(r.get("hit_btts"), resuelto)})
                for mercado in ("corners", "faltas", "tarjetas"):
                    lado, linea = r.get(f"pred_{mercado}_probable"), r.get(f"pred_{mercado}_line")
                    if pd.notna(lado) and pd.notna(linea):
                        filas.append({**base, "mercado": mercado.capitalize(), "pick": f"{'Más' if lado == 'Sí' else 'Menos'} de {linea:g}",
                                      "prob": None, "estado": _estado(r.get(f"hit_{mercado}"), resuelto)})
    if tipo != TIPO_MODELO and not tlog.empty:
        t = tlog[(pd.to_datetime(tlog["fecha"]).dt.date == fecha) & (tlog["plan"].map(plan_tipo) == tipo)]
        t = t.drop_duplicates(subset=["home_team", "away_team", "market", "pick"])
        for _, r in t.iterrows():
            marcador = (f"{int(r['home_goals'])}-{int(r['away_goals'])}" if pd.notna(r.get("home_goals")) else "vs")
            estado = {"acierto": "Acertado", "fallo": "Fallado"}.get(r["leg_result"], "Pendiente")
            filas.append({"partido": f"{r['home_team']} {marcador} {r['away_team']}", "competition": r["competition"],
                          "utc_date": r["utc_date"], "mercado": DAY_MARKET_NAMES.get(r["market"], r["market"]), "pick": str(r["pick"]).split(" (")[0],
                          "prob": (r["prob_ajustada"] * 100) if pd.notna(r.get("prob_ajustada")) else None,
                          "estado": estado})
    return pd.DataFrame(filas)


def prediction_detail(long: pd.DataFrame) -> pd.DataFrame:
    """Cada predicción con el pick del modelo, su probabilidad, el marcador y
    si acertó — para entender cada ✔/✘ de los agregados."""
    if long.empty:
        return pd.DataFrame()
    d = long
    return pd.DataFrame({
        "Día": d["fecha"], "Liga": d["competition"], "Partido": d["home_team"] + " vs " + d["away_team"],
        "Marcador": d.get("marcador"), "Mercado": d["mercado"], "Pick del modelo": d.get("pick"),
        "Prob.": (d["esperado"] * 100).round(0), "Resultado": d["acierto"].map({True: "✔ Acertado", False: "✘ Fallado"}),
    }).sort_values(["Día", "Partido", "Mercado"], ascending=[False, True, True])


def team_detail(long: pd.DataFrame, equipo: str) -> pd.DataFrame:
    return prediction_detail(long[(long["home_team"] == equipo) | (long["away_team"] == equipo)])


# --- Fuerza por mercado para resaltar la vista Jornada ---
STRENGTH_PRIOR = 10  # "predicciones fantasma" en 0: con poca muestra la fuerza se acerca a neutra


def strength_tables(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(liga × mercado, equipo × mercado) con diferencia (acierto − esperado,
    puntos) y predicciones — lo mismo que muestran 'Fortalezas y debilidades' y
    'Por equipo' de Reportes."""
    if long.empty:
        vacio = pd.DataFrame(columns=["diferencia", "predicciones"])
        return vacio, vacio
    liga = summarize(long, ["competition", "mercado"]).set_index(["competition", "mercado"])
    equipo = summarize(by_team(long), ["equipo", "mercado"]).set_index(["equipo", "mercado"])
    return liga[["diferencia", "predicciones"]], equipo[["diferencia", "predicciones"]]


def match_strength(liga: pd.DataFrame, equipo: pd.DataFrame, competition: str, home: str, away: str,
                   mercado: str) -> tuple[float | None, int]:
    """Fuerza del modelo en ese mercado para ese partido: promedio ponderado por
    cantidad de predicciones de la liga y de los dos equipos, encogido hacia 0
    con STRENGTH_PRIOR. (fuerza en puntos, predicciones detrás) o (None, 0)."""
    partes = []
    for tabla, clave in ((liga, (competition, mercado)), (equipo, (home, mercado)), (equipo, (away, mercado))):
        if clave in tabla.index:
            fila = tabla.loc[clave]
            partes.append((float(fila["diferencia"]), int(fila["predicciones"])))
    n = sum(p[1] for p in partes)
    if not n:
        return None, 0
    return sum(d * k for d, k in partes) / (n + STRENGTH_PRIOR), n
