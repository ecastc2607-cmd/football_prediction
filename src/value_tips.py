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
        for columna, mercado, pick, clave in _OPCIONES:
            prob = r[columna] / 100
            if prob < min_prob:
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
    df["prob_ajustada"] = model_weight * df["prob_modelo"] + (1 - model_weight) * prob_mercado
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
