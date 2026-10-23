"""
Tips de valor del día: de todos los partidos del día (cualquier liga), elige
los picks más probables según el modelo, los compara contra la mejor cuota
disponible y sugiere cuánto apostar de la banca con Kelly fraccional.

También evalúa partidos EN JUEGO (live_candidate_picks): misma regla, con la
probabilidad ajustada en vivo y la cuota en vivo de BWIN (365scores).

Reglas (las estándar de apuesta de valor, en su versión conservadora):
  1. Solo picks con probabilidad del modelo >= MIN_MODEL_PROB — el "bajo
     riesgo" del pedido — y nunca de partidos marcados de baja confianza (ahí
     el modelo se apoya casi todo en el promedio de liga, no en datos propios).
  2. Máximo un pick por partido: dos mercados del mismo partido están
     correlacionados y duplicarían el riesgo sin que se note.
  3. Solo apuestas SIMPLES, no combinadas: el margen de la casa se multiplica
     en cada selección que se agrega.
  4. Hay valor solo si la probabilidad supera a la que paga la cuota. Como un
     modelo de Poisson simple se equivoca más que el mercado, a la ventaja que
     cree ver el modelo se le descuenta la mitad (se promedia su probabilidad
     con la que implica la cuota) y aun así debe quedar >= MIN_EDGE.
  5. Monto: un cuarto de Kelly, con tope por apuesta y tope total del día.

Nada de esto convierte la apuesta en segura: un pick de 75% falla 1 de cada 4.
"""
from __future__ import annotations

import pandas as pd

# 65% (antes 70%). Calibración con ~450 picks resueltos (oct-2026, partidos
# reales + backtest): el tramo 65-70% acertó 72% — igual que el de 70-75%
# (67%) —, mientras que 60-65% acertó solo 51% (el modelo se sobreestima ahí).
# Bajar a 65% suma opciones sin bajar el acierto; bajar a 60% no. El filtro que
# de verdad protege sigue siendo el de VALOR contra la cuota (MIN_EDGE).
MIN_MODEL_PROB = 0.65
MODEL_WEIGHT = 0.5  # peso del modelo al promediar con la probabilidad implícita de la cuota
# En vivo el modelo pesa menos: el ajuste en vivo (remates, efecto marcador)
# usa pesos de la literatura que todavía NO se validaron contra el mercado con
# datos propios (oct-2026; para eso se juntan live_odds_snapshots.csv). Ej. real:
# Rennes 0-1 OFI al descanso, el modelo daba 46% a OFI y BWIN pagaba 3.50 (29%).
# Revisar este peso cuando haya historia suficiente.
LIVE_MODEL_WEIGHT = 0.25
# Corners y tarjetas (totals_model.py): el modelo le gana poco a la simple
# frecuencia de la liga (backtest oct-2026: 0-1.7% de log-loss) y la casa cobra
# ~9% de margen en esos mercados, así que el modelo pesa menos al calcular el valor.
TOTALS_MODEL_WEIGHT = 0.35
TOTALS_MARKETS = {"corners": "Corners", "cards": "Tarjetas"}
# Mercados que el modelo NO sabe predecir en una liga: no se recomiendan ahí.
# Liga BetPlay (backtest walk-forward 2026, 320 partidos): subestima los goles
# (más de 2.5: predice 43%, pasa 48%; ambos anotan: 45% vs 56%) y no distingue
# partidos (dice 30% o 58% y pasa ~48% igual). 1X2 sí le gana a la base (log-loss
# 1.038 vs 1.067) y doble oportunidad acierta 77%.
# Corners en Colombia (totals_backtest, 732 partidos): peor que la frecuencia
# de la liga en las 3 líneas; tarjetas sí le gana (1.2-1.8%) y está calibrado.
EXCLUDED_MARKETS = {"COL": {"Goles", "Ambos anotan", "Corners"}}
# Picks ya prácticamente decididos: la casa paga ~1.01, no aportan nada.
LIVE_MAX_PROB = 0.97
MIN_EDGE = 0.03  # valor esperado mínimo (+3% por peso apostado), ya descontado
KELLY_FRACTION = 0.25
MAX_STAKE_PER_BET = 0.05  # de la banca
MAX_DAILY_EXPOSURE = 0.20  # suma de todas las apuestas del día, de la banca
# Cuotas implausibles: o es un error de tipeo (verificado: en un navegador en
# español "1.36" se guarda como 136), o el mercado sabe algo que el modelo no
# (lesiones, rotaciones) — un Poisson simple no le gana por tanto al mercado.
# En ambos casos se marca para revisar en vez de recomendar esa apuesta.
MAX_VALID_ODDS = 20.0
MAX_ODDS_VS_FAIR = 1.5

# (columna de probabilidad en el DataFrame de predicciones, mercado, pick, clave de cuota)
_OPCIONES = [
    ("home_win", "Resultado", "Local", "home"),
    ("draw", "Resultado", "Empate", "draw"),
    ("away_win", "Resultado", "Visitante", "away"),
    ("over_2_5", "Goles", "Más de 2.5", "over"),
    ("under_2_5", "Goles", "Menos de 2.5", "under"),
    ("dc_1x", "Doble oportunidad", "1X (local o empate)", "dc_1x"),
    ("dc_x2", "Doble oportunidad", "X2 (empate o visitante)", "dc_x2"),
]


def candidate_picks(predictions: pd.DataFrame, min_prob: float = MIN_MODEL_PROB) -> pd.DataFrame:
    """Una fila por pick que supera `min_prob`, de partidos sin baja confianza.
    Puede haber más de un pick por partido acá (ej. Local y Menos de 2.5):
    la regla de uno-por-partido se aplica después, al ver cuál tiene más valor."""
    if predictions.empty:
        return pd.DataFrame()
    preds = predictions.copy()
    preds["under_2_5"] = 100 - preds["over_2_5"]
    preds["dc_1x"] = preds["home_win"] + preds["draw"]
    preds["dc_x2"] = preds["away_win"] + preds["draw"]
    if "low_confidence" in preds:
        preds = preds[~preds["low_confidence"].astype(bool)]

    filas = []
    for _, r in preds.iterrows():
        excluidos = EXCLUDED_MARKETS.get(r.get("competition"), set())
        for columna, mercado, pick, clave in _OPCIONES:
            prob = r[columna] / 100
            if prob < min_prob or mercado in excluidos:
                continue
            home_short = r.get("home_team_short") or r["home_team"]
            away_short = r.get("away_team_short") or r["away_team"]
            filas.append({
                "competition": r.get("competition", ""),
                "competition_name": r.get("competition_name", ""),
                "utc_date": r["utc_date"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "home_team_short": home_short,
                "away_team_short": away_short,
                "partido": f"{home_short} vs {away_short}",
                "mercado": mercado,
                "pick": pick,
                "odds_key": clave,
                "prob_modelo": prob,
                "cuota_justa": round(1 / prob, 2),
            })
    return pd.DataFrame(filas)


def live_candidate_picks(live: list[dict], min_prob: float = MIN_MODEL_PROB) -> pd.DataFrame:
    """Picks de partidos EN JUEGO. Cada elemento de `live`: datos del partido
    + "probs" {columna de _OPCIONES: prob 0-1, ajustada en vivo} + "odds"
    {clave de cuota: cuota BWIN en vivo} + "en_vivo" (texto minuto/marcador).
    Solo partidos con base anclada al mercado (cuotas previas guardadas): sin
    eso el ajuste en vivo parte del modelo solo, que en copas es poco fiable."""
    filas = []
    for m in live:
        for columna, mercado, pick, clave in _OPCIONES:
            prob = (m.get("probs") or {}).get(columna)
            if prob is None or prob < min_prob or prob >= LIVE_MAX_PROB:
                continue
            filas.append({
                "competition": m["competition"], "competition_name": m.get("competition_name", ""),
                "utc_date": m["utc_date"], "home_team": m["home_team"], "away_team": m["away_team"],
                "home_team_short": m["home_team"], "away_team_short": m["away_team"],
                "partido": f"{m['home_team']} vs {m['away_team']}", "en_vivo": m.get("en_vivo", ""),
                "mercado": mercado, "pick": pick, "odds_key": clave,
                "prob_modelo": prob, "cuota_justa": round(1 / prob, 2),
                "cuota_auto": (m.get("odds") or {}).get(clave), "casa": "BWIN en vivo (365scores)",
            })
    return pd.DataFrame(filas)


def totals_candidates(day_predictions: pd.DataFrame, find_odds_row, min_prob: float = MIN_MODEL_PROB) -> pd.DataFrame:
    """Picks de corners/tarjetas en la línea PRINCIPAL de la casa. `find_odds_row(fila)`
    devuelve la fila de cuotas de 365scores de ese partido (o None). Solo con
    cuota real: sin la línea de la casa no hay qué evaluar."""
    from . import totals_model
    from .odds_api import row_price

    filas = []
    for _, r in day_predictions.iterrows():
        ev = find_odds_row(r)
        if not ev:
            continue
        for clave, mercado in TOTALS_MARKETS.items():
            if mercado in EXCLUDED_MARKETS.get(r.get("competition"), set()):
                continue
            linea = ev.get(f"{clave}_line")
            if linea is None or linea != linea:
                continue
            modelo = totals_model.predict(r.get("competition"), r["home_team"], r["away_team"],
                                          "corners" if clave == "corners" else "tarjetas", float(linea))
            if modelo is None:
                continue
            esperados, p_over = modelo
            for lado, prob, etiqueta in (("over", p_over, "Más"), ("under", 1 - p_over, "Menos")):
                precio = row_price(ev, f"{clave}_{lado}")
                if prob < min_prob or not precio:
                    continue
                home_short = r.get("home_team_short") or r["home_team"]
                away_short = r.get("away_team_short") or r["away_team"]
                filas.append({
                    "competition": r.get("competition", ""), "competition_name": r.get("competition_name", ""),
                    "utc_date": r["utc_date"], "home_team": r["home_team"], "away_team": r["away_team"],
                    "home_team_short": home_short, "away_team_short": away_short,
                    "partido": f"{home_short} vs {away_short}", "mercado": mercado,
                    "pick": f"{etiqueta} de {float(linea):g} (esp. {esperados:.1f})", "odds_key": f"{clave}_{lado}",
                    "prob_modelo": prob, "cuota_justa": round(1 / prob, 2),
                    "cuota_auto": precio[0], "casa": precio[1], "peso_modelo": TOTALS_MODEL_WEIGHT,
                })
    return pd.DataFrame(filas)


def evaluate(picks: pd.DataFrame, bankroll: float, max_exposure: float = MAX_DAILY_EXPOSURE,
             model_weight: float = MODEL_WEIGHT) -> pd.DataFrame:
    """Recibe los candidatos con columnas `cuota` (la que se usará: manual si
    se escribió, si no la automática) y devuelve, además, valor, estado y
    monto sugerido. Estado: "Apostar" / "Sin valor" / "Falta cuota"."""
    if picks.empty:
        return picks.assign(prob_ajustada=[], valor=[], estado=[], fraccion=[], monto=[], ganancia=[])
    df = picks.copy()
    cuota = pd.to_numeric(df["cuota"], errors="coerce")
    tiene_cuota = (
        (cuota > 1.0) & (cuota <= MAX_VALID_ODDS)
        & (cuota <= MAX_ODDS_VS_FAIR / df["prob_modelo"])
    )
    cuota = cuota.where(tiene_cuota)

    prob_mercado = 1 / cuota.where(tiene_cuota)
    # Peso del modelo por pick si la fila lo trae (corners/tarjetas pesan menos).
    peso = df["peso_modelo"].fillna(model_weight) if "peso_modelo" in df else model_weight
    df["prob_ajustada"] = peso * df["prob_modelo"] + (1 - peso) * prob_mercado
    df["valor"] = df["prob_ajustada"] * cuota - 1
    kelly = (df["prob_ajustada"] * cuota - 1) / (cuota - 1)

    df["estado"] = "Falta cuota"
    df.loc[df["cuota"].notna() & ~tiene_cuota, "estado"] = (
        "Cuota demasiado alta: ¿error de tipeo, o noticia que el modelo no conoce?"
    )
    df.loc[tiene_cuota & (df["valor"] < MIN_EDGE), "estado"] = "Sin valor"
    apostable = tiene_cuota & (df["valor"] >= MIN_EDGE)
    df.loc[apostable, "estado"] = "Apostar"

    # Uno por partido: si un partido tiene varios picks con valor, se queda el
    # de mayor valor y los demás pasan a "Otro pick del mismo partido".
    df["_orden"] = df["valor"].where(apostable, -1)
    mejor_por_partido = df.sort_values("_orden", ascending=False).drop_duplicates(
        subset=["home_team", "away_team", "utc_date"]
    ).index
    duplicado = apostable & ~df.index.isin(mejor_por_partido)
    df.loc[duplicado, "estado"] = "Otro pick del mismo partido"
    apostable = apostable & ~duplicado
    df = df.drop(columns="_orden")

    fraccion = (kelly * KELLY_FRACTION).clip(lower=0, upper=MAX_STAKE_PER_BET).where(apostable, 0.0)
    total = fraccion.sum()
    if total > max_exposure:
        fraccion = fraccion * (max(max_exposure, 0.0) / total)
    df["fraccion"] = fraccion.fillna(0.0)
    # Montos redondeados a $100 hacia abajo — nunca apostar más de lo calculado.
    df["monto"] = (df["fraccion"] * bankroll // 100 * 100).astype(int)
    df["ganancia"] = (df["monto"] * (cuota - 1)).where(df["monto"] > 0, 0).fillna(0).round(0).astype(int)
    return df


def day_summary(evaluated: pd.DataFrame, bankroll: float) -> dict:
    apuestas = evaluated[evaluated["monto"] > 0] if not evaluated.empty else evaluated
    if apuestas.empty:
        return {"n": 0, "apostado": 0, "esperado": 0.0, "si_todo_acierta": bankroll,
                "si_todo_falla": bankroll, "prob_todo_acierta": 0.0}
    apostado = int(apuestas["monto"].sum())
    esperado = float((apuestas["monto"] * apuestas["valor"]).sum())
    prob_todo = float(apuestas["prob_ajustada"].prod())
    return {
        "n": len(apuestas),
        "apostado": apostado,
        "esperado": esperado,
        "si_todo_acierta": bankroll + int(apuestas["ganancia"].sum()),
        "si_todo_falla": bankroll - apostado,
        "prob_todo_acierta": prob_todo,
    }


# --- Mercados especiales (vista Avanzada, special_markets.py) -------------------
# Solo entran los mercados que el backtest walk-forward marca "Candidato" o
# "Prometedor", en las ligas donde sus picks aciertan >= 75% con muestra, y que
# BWIN ofrece (si no hay cuota no hay qué evaluar). Experimental: las cuotas
# especiales de BWIN se guardan recién desde oct-2026, así que el modelo pesa
# poco frente a la cuota, igual que en corners/tarjetas.
SPECIAL_MODEL_WEIGHT = 0.35
SPECIAL_MIN_LEAGUE_PICKS = 15
SPECIAL_VERDICTS = {"Candidato", "Prometedor"}
# Roja/penal: la "probabilidad" es solo la frecuencia de la liga y BWIN ya la
# cobra justo (oct-2026). Estadísticas: sin modelo en vivo para el día todavía.
SPECIAL_EXCLUDED_FAMILIES = {"Eventos", "Estadísticas"}
# Nombre del mercado en Tips / tips_log y cómo se arma el pick.
SPECIAL_TIPS_MARKET = {
    "Favorito empate no válido (HA 0)": ("Empate no válido", "fav", None),
    "No favorito +1.5": ("Hándicap asiático", "und", 1.5),
    "No favorito +1": ("Hándicap asiático", "und", 1.0),
    "Favorito -1": ("Hándicap asiático", "fav", -1.0),
    "Favorito marca primero": ("Primer gol", "fav", None),
    "Más de 1.5 goles": ("Goles (otra línea)", None, "Más de 1.5"),
    "Menos de 3.5 goles": ("Goles (otra línea)", None, "Menos de 3.5"),
    "1T más de 0.5": ("1er tiempo", None, "Más de 0.5"),
    "1T menos de 1.5": ("1er tiempo", None, "Menos de 1.5"),
}
_SPECIAL_CACHE: dict = {}


def special_rules() -> tuple[dict, set]:
    """({mercado: corrección de probabilidad}, {(liga, mercado) permitidos})
    según el backtest guardado. La corrección resta lo que el modelo prometió
    de más en ese mercado (acierto real − esperado)."""
    from . import special_markets as sm

    clave = sm.BACKTEST_PATH.stat().st_mtime if sm.BACKTEST_PATH.exists() else 0
    if clave not in _SPECIAL_CACHE:
        bt = sm.load_backtest()
        if bt.empty:
            _SPECIAL_CACHE[clave] = ({}, set())
        else:
            bt = bt[~bt["familia"].isin(SPECIAL_EXCLUDED_FAMILIES) & bt["mercado"].isin(SPECIAL_TIPS_MARKET)]
            total = sm.summarize(bt)
            total["veredicto"] = total.apply(sm.verdict, axis=1)
            buenos = total[total["veredicto"].isin(SPECIAL_VERDICTS)]
            correccion = ((buenos["acierto_pct"] - buenos["esperado_pct"]) / 100).set_axis(buenos["mercado"]).to_dict()
            liga = sm.summarize(bt[bt["mercado"].isin(correccion)], by=["competition"])
            ok = liga[(liga["picks"] >= SPECIAL_MIN_LEAGUE_PICKS) & (liga["acierto_pct"] >= sm.MIN_PROB * 100)]
            _SPECIAL_CACHE[clave] = (correccion, set(zip(ok["competition"], ok["mercado"])))
    return _SPECIAL_CACHE[clave]


def special_candidates(day_predictions: pd.DataFrame, lines_for) -> pd.DataFrame:
    """Picks de mercados especiales. `lines_for(fila)` devuelve las líneas de
    BWIN del partido (columnas de events365.special_odds) o None; se llama solo
    para los partidos que tienen algún pick."""
    from . import special_markets as sm

    correccion, permitidos = special_rules()
    if not permitidos or day_predictions.empty:
        return pd.DataFrame()
    share = pd.read_csv(sm.SHARE_PATH, index_col=0).iloc[:, 0].to_dict() if sm.SHARE_PATH.exists() else {}
    filas = []
    for _, r in day_predictions.iterrows():
        comp = r.get("competition")
        try:
            hx, ax = float(r["home_xg"]), float(r["away_xg"])
        except (TypeError, ValueError, KeyError):
            continue
        fav_local = hx >= ax
        picks = [(merc, prob, p_dev) for _, merc, prob, p_dev in
                 sm.goal_market_probs(sm.Match(hx, ax, share.get(comp, sm.DEFAULT_FIRST_HALF_SHARE)))
                 if (comp, merc) in permitidos and prob >= sm.MIN_PROB]
        if not picks:
            continue
        lineas = lines_for(r)
        if lineas is None or lineas.empty:
            continue
        home_short = r.get("home_team_short") or r["home_team"]
        away_short = r.get("away_team_short") or r["away_team"]
        fav, und = (r["home_team"], r["away_team"]) if fav_local else (r["away_team"], r["home_team"])
        for merc, prob, p_dev in picks:
            precio = sm.bwin_price(merc, fav_local, lineas)
            if not precio:
                continue
            mercado, lado, extra = SPECIAL_TIPS_MARKET[merc]
            equipo = fav if lado == "fav" else und if lado == "und" else None
            if mercado == "Hándicap asiático":
                pick = f"{equipo} {extra:+g}"
            elif equipo:
                pick = equipo
            else:
                pick = extra
            prob_cal = min(max(prob + correccion.get(merc, 0.0), 0.01), 0.99)
            filas.append({
                "competition": comp, "competition_name": r.get("competition_name", ""),
                "utc_date": r["utc_date"], "home_team": r["home_team"], "away_team": r["away_team"],
                "home_team_short": home_short, "away_team_short": away_short,
                "partido": f"{home_short} vs {away_short}", "mercado": mercado, "pick": pick,
                "odds_key": f"especial:{merc}", "prob_modelo": prob_cal, "cuota_justa": round(1 / prob_cal, 2),
                "cuota_auto": precio[0], "casa": "BWIN", "peso_modelo": SPECIAL_MODEL_WEIGHT,
                "explicacion": sm.market_help(merc)
                + (f" (Si se devuelve: {p_dev:.0%} de probabilidad; la prob. es sin contar devolución.)"
                   if p_dev else ""),
            })
    return pd.DataFrame(filas)
