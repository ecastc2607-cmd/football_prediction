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

from . import config

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
EXCLUDED_MARKETS = {"COL": {"Goles", "Ambos anotan", "Corners"},
                    "CBP": {"Goles", "Ambos anotan", "Corners"}}  # Copa BetPlay: modelo de la liga
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
# Cuando el modelo le da al pick mucho más que la casa, casi siempre se equivoca
# el modelo (tips resueltos a nov-2026: con >15 puntos de diferencia acertaron
# 39% y perdieron 31% de lo apostado; con <=5 puntos, 80% y +8.5%). Esos picks
# no se recomiendan: es más probable una noticia que el modelo no conoce.
MAX_MODEL_GAP = 0.15
# Regla de CONSENSO (nov-2026) para "Menos de 3.5 goles": buscar valor contra la
# casa ahí pierde (backtest con cuotas BWIN: 19 picks, 47% de acierto, -33%),
# porque el "valor" aparece cuando la cuota está alta, o sea cuando la casa
# espera muchos goles, y suele tener razón. Lo que gana es lo contrario: apostar
# cuando el modelo (>= 70%) y la casa coinciden (a <= 10 pts): 141 partidos, 83%
# de acierto a cuota 1,31, +8,3% (+11% oct, +6% nov; de +3% a +17% variando los
# límites). Monto fijo y chico: el margen es delgado.
CONSENSUS_MARKETS = {"Menos de 3.5 goles"}
CONSENSUS_MIN_PROB = 0.70
CONSENSUS_MAX_GAP = 0.10
CONSENSUS_STAKE = 0.015  # de la banca
GAP_STATE = "Discrepa mucho con la casa (> 15 pts): probable error del modelo"

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
    from .calibration import adjust_predictions, recalibrate_over

    # Favorito corregido en las ligas donde el modelo lo sobrevalora (Nations League).
    preds = adjust_predictions(predictions).copy()
    # Goles: el modelo es demasiado extremo ahí (calibration.py); se usa la
    # probabilidad recalibrada con resultados reales.
    preds["over_2_5"] = preds["over_2_5"].map(lambda v: 100 * recalibrate_over(v / 100) if pd.notna(v) else v)
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
    discrepa = tiene_cuota & (df["prob_modelo"] - prob_mercado > MAX_MODEL_GAP)
    # Mercados de consenso: no se mira el valor sino que modelo y casa coincidan.
    consenso = (df["consenso"].fillna(False).astype(bool) if "consenso" in df
                else pd.Series(False, index=df.index)) & tiene_cuota
    prob_cruda = df["prob_raw"] if "prob_raw" in df else df["prob_modelo"]
    acuerdo = consenso & (prob_cruda >= CONSENSUS_MIN_PROB) & ((prob_cruda - prob_mercado).abs() <= CONSENSUS_MAX_GAP)
    apostable = (tiene_cuota & (df["valor"] >= MIN_EDGE) & ~discrepa & ~consenso) | acuerdo
    df.loc[apostable, "estado"] = "Apostar"
    df.loc[discrepa & (df["valor"] >= MIN_EDGE) & ~consenso, "estado"] = GAP_STATE
    df.loc[consenso & ~acuerdo, "estado"] = "Sin consenso con la casa (regla de Menos de 3.5)"

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

    fraccion = (kelly * KELLY_FRACTION).clip(lower=0, upper=MAX_STAKE_PER_BET)
    fraccion = fraccion.where(~consenso, CONSENSUS_STAKE).where(apostable, 0.0)
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
# "Prometedor" (o los de umbral 65%, sm.LOWER_PROB_MARKETS, por calibración),
# en las ligas donde sus picks aciertan >= 75% con muestra, y que
# BWIN ofrece (si no hay cuota no hay qué evaluar). Experimental: las cuotas
# especiales de BWIN se guardan recién desde oct-2026, así que el modelo pesa
# poco frente a la cuota, igual que en corners/tarjetas.
SPECIAL_MODEL_WEIGHT = 0.35
SPECIAL_MIN_LEAGUE_PICKS = 15
SPECIAL_VERDICTS = {"Candidato", "Prometedor"}
# Roja/penal: la "probabilidad" es solo la frecuencia de la liga y BWIN ya la
# cobra justo (oct-2026).
SPECIAL_EXCLUDED_FAMILIES = {"Eventos"}
# Tiros al arco y fueras de juego (desde oct-2026, con las estadísticas
# europeas de 365): mismo modelo que corners/tarjetas, misma regla de ligas.
STAT_TIPS = {"off": "Fueras de juego", "sot": "Tiros al arco", "cor": "Corners", "tar": "Tarjetas",
             "fal": "Faltas", "sho": "Remates totales"}
TEAM_STAT_TIPS_MARKET = "Tiros al arco por equipo"  # pick: "<equipo>: Más de 3.5 (esp. 4.6)"
# Pedido del usuario (nov-2026): las casas de Colombia no siempre abren fueras de
# juego ni tiros al arco. Esos picks entran a Tips solo si BWIN publica ese
# mercado (tipo de línea de 365) en el partido. Tiros al arco por equipo no
# tiene línea propia en 365: se toma el total de tiros al arco como señal.
BWIN_REQUIRED_MARKET = {"Fueras de juego": 140, "Tiros al arco": 139, TEAM_STAT_TIPS_MARKET: 139,
                        "Más tiros al arco": 139, "Más fueras de juego": 140}
# Prioridades del usuario (nov-2026), porque las casas a veces bloquean tiros al
# arco y fueras de juego:
#  - Goles: si ni Más ni Menos de 2.5 llega al MIN_MODEL_PROB, se ofrece Más de
#    1.5 (probabilidad recalibrada) en cualquier liga donde Goles no esté excluido.
#    Backtest: calibrado (78% prometido, 77% real) pero a la cuota de BWIN (~1,26)
#    sin ganancia en general: la regla de valor decide si se apuesta.
#  - Corners: Más de 7.5 en todas las ligas desde 70% (acierta 73% contra 70% de
#    la frecuencia de la línea; entre 65% y 70% no le gana a la frecuencia).
# Más tarjetas (qué equipo recibe más tarjetas que el rival): el modelo casi
# nunca pasa de 55% en un mercado de 3 opciones, así que el umbral es propio
# (ver special_markets.MORE_CARDS_RECAL); BWIN no lo publica en 365: cuota manual.
MORE_CARDS_MARKET = "Más tarjetas"
MORE_CARDS_MIN_PROB = 0.48
# Más corners: qué equipo saca más corners (mismo esquema; desde 60%, ver
# special_markets.MORE_CORNERS_RECAL). Tampoco está en BWIN/365: cuota manual.
MORE_CORNERS_MARKET = "Más corners"
MORE_CORNERS_MIN_PROB = 0.60
# Más tiros al arco: desde 65% (recalibrado; ver special_markets.MORE_SOT_RECAL).
# Como los demás de tiros al arco, solo si BWIN abre tiros al arco en el partido.
MORE_SOT_MARKET = "Más tiros al arco"
MORE_SOT_MIN_PROB = 0.65
# Más fueras de juego: desde 50% (ver special_markets.MORE_OFF_RECAL); solo si
# BWIN abre fueras de juego en el partido.
MORE_OFF_MARKET = "Más fueras de juego"
MORE_OFF_MIN_PROB = 0.50
# Más faltas: desde 55% (ver special_markets.MORE_FOULS_RECAL). BWIN no publica
# faltas en 365: cuota manual.
MORE_FOULS_MARKET = "Más faltas"
MORE_FOULS_MIN_PROB = 0.55
# (mercado, función de special_markets, umbral, qué se cuenta, acierto y base del backtest)
MORE_MARKETS = [
    (MORE_CARDS_MARKET, "more_cards_prediction", MORE_CARDS_MIN_PROB, "tarjetas (amarillas + rojas)",
     "recibe", "acierta ~49% cuando dice ≥48%, contra ~45% sin modelo"),
    (MORE_CORNERS_MARKET, "more_corners_prediction", MORE_CORNERS_MIN_PROB, "corners", "saca",
     "acierta ~64% cuando dice ≥60%, contra ~56% sin modelo"),
    (MORE_SOT_MARKET, "more_sot_prediction", MORE_SOT_MIN_PROB, "tiros al arco", "hace",
     "acierta ~72% cuando dice ≥65%, contra ~56% sin modelo"),
    (MORE_OFF_MARKET, "more_off_prediction", MORE_OFF_MIN_PROB, "fueras de juego", "cae en",
     "acierta ~58% cuando dice ≥50%, contra ~41% sin modelo"),
    (MORE_FOULS_MARKET, "more_fouls_prediction", MORE_FOULS_MIN_PROB, "faltas", "comete",
     "acierta ~61% cuando dice ≥55%, contra ~48% sin modelo"),
]
GOALS_FALLBACK_MARKET = "Más de 1.5 goles"
CORNERS_PRIORITY_MARKET = "Corners: más de 7.5"
CORNERS_PRIORITY_PROB = 0.70
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


# --- Marca de fortaleza del mercado en la liga (columna de Tips) ---------------
# Fuerza = cuánto acierta el pick por ENCIMA de lo que saldría sin modelo, en
# puntos, encogida hacia 0 con poca muestra (n/(n+10), igual que Jornada):
#   - 1X2 / doble oportunidad / goles 2.5: tabla de Reportes (acierto − esperado).
#   - Corners / tarjetas en la línea principal (totals_candidates): mismo
#     modelo que las estadísticas de Avanzada, así que se mide con su backtest
#     (todas las líneas de esa estadística en la liga), no con la tendencia de
#     Reportes, que tiene apenas ~40 partidos por liga.
#   - Mercados de Avanzada: acierto de los picks del backtest en la liga menos
#     la frecuencia de ese mercado en TODOS los partidos de la liga.
# Backtest de nov-2026: tarjetas y faltas sí le ganan a la frecuencia de la
# liga (+15 a +27 pts en las líneas medias), corners casi nada (0 a +9).
STRENGTH_MARK_MIN = 5.0
STRENGTH_PRIOR = 10
_REPORT_MARKET = {"Resultado": "Resultado (1X2)", "Doble oportunidad": "Doble oportunidad",
                  "Goles": "Goles +/-2.5", "Corners": "Corners", "Tarjetas": "Tarjetas"}
_STRENGTH_CACHE: dict = {}


def _strength_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    from . import reports, special_markets as sm
    from .prediction_log import LOG_PATH, load_log

    clave = tuple(p.stat().st_mtime if p.exists() else 0 for p in (LOG_PATH, sm.BACKTEST_PATH))
    if clave not in _STRENGTH_CACHE:
        log = load_log()
        liga = (reports.strength_tables(reports.predictions_long(log, include_backtest=True))[0]
                if not log.empty else pd.DataFrame())
        bt = sm.load_backtest()
        filas = []
        if not bt.empty:
            bt = bt.dropna(subset=["gano"])
            bt = bt.assign(base=bt.groupby(["competition", "mercado"])["gano"].transform("mean"))
            for (comp, merc), g in bt.groupby(["competition", "mercado"]):
                picks = g[g["prob"] >= sm.min_prob_for(merc)]
                if len(picks):
                    filas.append({"competition": comp, "mercado": merc, "n": len(picks),
                                  "diferencia": 100 * (picks["gano"] - picks["base"]).mean()})
            # Por estadística completa (todas sus líneas): para corners/tarjetas de Tips.
            est = bt[(bt["familia"] == "Estadísticas") & (bt["prob"] >= 0.65)]
            for (comp, stat), g in est.groupby([est["competition"], est["mercado"].str.split(":").str[0]]):
                filas.append({"competition": comp, "mercado": f"stat:{stat}", "n": len(g),
                              "diferencia": 100 * (g["gano"] - g["base"]).mean()})
        _STRENGTH_CACHE.clear()
        _STRENGTH_CACHE[clave] = (liga, pd.DataFrame(filas).set_index(["competition", "mercado"]) if filas
                                  else pd.DataFrame())
    return _STRENGTH_CACHE[clave]


def market_strength(competition: str, mercado: str, odds_key: str, pick: str = "") -> float | None:
    """Fuerza (pts, encogida) del mercado del pick en su liga; None sin datos."""
    import re

    liga, especial = _strength_sources()
    comp = config.strength_code(competition)
    if str(odds_key).startswith(("corners_", "cards_")):
        # La línea exacta del pick si el backtest la tiene; si no, la estadística completa.
        stat = "Corners" if str(odds_key).startswith("corners_") else "Tarjetas"
        m = re.match(r"(Más|Menos) de ([0-9.]+)", str(pick))
        exacta = f"{stat}: {m.group(1).lower()} de {float(m.group(2)):g}" if m else None
        clave = next((c for c in ((comp, exacta), (comp, f"stat:{stat}"))
                      if c[1] and not especial.empty and c in especial.index), None)
        if clave is None:
            return None
        n, dif = float(especial.loc[clave, "n"]), float(especial.loc[clave, "diferencia"])
    elif str(odds_key).startswith("especial:"):
        clave, tabla = (comp, str(odds_key).split(":", 1)[1]), especial
        if tabla.empty or clave not in tabla.index:
            return None
        n, dif = float(tabla.loc[clave, "n"]), float(tabla.loc[clave, "diferencia"])
    else:
        nombre = _REPORT_MARKET.get(mercado)
        if not nombre or liga.empty or (comp, nombre) not in liga.index:
            return None
        n, dif = float(liga.loc[(comp, nombre), "predicciones"]), float(liga.loc[(comp, nombre), "diferencia"])
    return dif * n / (n + STRENGTH_PRIOR)


def strength_label(fuerza: float | None) -> str:
    if fuerza is None or pd.isna(fuerza):
        return "· sin datos"
    if fuerza >= STRENGTH_MARK_MIN:
        return f"🔵 Fortaleza ({fuerza:+.0f})"
    if fuerza <= -STRENGTH_MARK_MIN:
        return f"🔴 Débil ({fuerza:+.0f})"
    return f"⚪ Neutral ({fuerza:+.0f})"


# --- Candidatos redundantes ----------------------------------------------------
# Pedido del usuario (nov-2026): "Newcastle: Menos de 5.5" y "Newcastle: Menos de
# 6.5" son la misma apuesta a otra línea. De cada grupo (partido, mercado,
# equipo, Más/Menos) queda una sola: la de más valor contra su cuota; si BWIN no
# trae cuota para ninguna, la de más fuerza en la liga (y, empatadas, la línea
# más exigente, que paga más).
_FAMILIA = {"Goles (otra línea)": "Goles"}


def drop_redundant(picks: pd.DataFrame, model_weight: float = MODEL_WEIGHT) -> pd.DataFrame:
    import re

    if picks.empty or "pick" not in picks:
        return picks
    patron = re.compile(r"^\s*(?:(.+?): )?(Más|Menos) de ([0-9]+(?:\.[0-9]+)?)")
    partes = picks["pick"].astype(str).map(lambda p: patron.match(p))
    con_linea = partes.map(bool)
    if not con_linea.any():
        return picks
    d = picks[con_linea].copy()
    m = partes[con_linea]
    # Un pick por partido, mercado y equipo, sin importar el lado (pedido del
    # usuario, nov-2026: "Menos de 3.5" y "Más de 1.5" del mismo partido eran
    # dos apuestas de goles redundantes).
    d["_grupo"] = (d["partido"].astype(str) + "|" + d["mercado"].map(lambda x: _FAMILIA.get(x, x)) + "|"
                   + m.map(lambda x: x.group(1) or ""))
    d["_linea"] = m.map(lambda x: float(x.group(3)))
    cuota = pd.to_numeric(d["cuota_auto"], errors="coerce") if "cuota_auto" in d else pd.Series(float("nan"), index=d.index)
    peso = d["peso_modelo"].fillna(model_weight) if "peso_modelo" in d else model_weight
    d["_valor"] = (peso * d["prob_modelo"] + (1 - peso) / cuota) * cuota - 1
    d["_con_cuota"] = cuota.notna()
    d["_fuerza"] = [market_strength(r.competition, r.mercado, r.odds_key, r.pick) for r in d.itertuples()]
    # Línea más exigente: la más alta en "Más", la más baja en "Menos".
    d["_exigencia"] = d["_linea"].where(m.map(lambda x: x.group(2)) == "Más", -d["_linea"])
    d["_cuota"] = cuota
    # Queda el de más fuerza en la liga y, empatados, el de mejor cuota.
    d = d.sort_values(["_fuerza", "_cuota", "_valor", "_exigencia"], ascending=False, na_position="last")
    quedan = d.drop_duplicates("_grupo").index
    return picks[~con_linea | picks.index.isin(quedan)]


# --- Cuánto se dio el mercado en cada equipo (columnas "Fortaleza local/visitante") ---
# Pedido del usuario (nov-2026): de los últimos TEAM_FORM_GAMES partidos de cada
# equipo (cualquier competición con datos), en cuántos se dio lo que pide el
# pick. Si el pick es de UN equipo (1X2, hándicap, tiros al arco por equipo),
# al otro se le cuenta lo que lograron SUS rivales: p. ej. "Arsenal +1.5" →
# visitante = en cuántos partidos del visitante su rival no perdió por 2 o más.
TEAM_FORM_GAMES = 10
TEAM_FORM_MIN = 3
_STAT_PICK_COL = {"Corners": "cor", "Tarjetas": "tar", "Fueras de juego": "off", "Tiros al arco": "sot",
                  "Faltas": "fal", "Remates totales": "sho", TEAM_STAT_TIPS_MARKET: "sot", "Más tarjetas": "tar",
                  "Más corners": "cor", "Más tiros al arco": "sot", "Más fueras de juego": "off",
                  "Más faltas": "fal"}
_FORM_CACHE: dict = {}


def _team_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(goles, estadísticas): un renglón por equipo y partido, con 'gf'/'ga'
    (goles) o '<col>_own'/'<col>_opp' (estadísticas), del más reciente al más viejo."""
    from . import match_stats_log
    from . import special_markets as sm
    from .prediction_log import LOG_PATH, load_log

    rutas = (LOG_PATH, match_stats_log.LOG_PATH)
    clave = tuple(p.stat().st_mtime if p.exists() else 0 for p in rutas)
    if clave in _FORM_CACHE:
        return _FORM_CACHE[clave]

    def largo(df, pares):
        partes = []
        for propio, rival, equipo in (("home", "away", "home_team"), ("away", "home", "away_team")):
            d = df[["utc_date", equipo]].rename(columns={equipo: "team"})
            for nombre, (c_home, c_away) in pares.items():
                d[f"{nombre}_own"] = df[c_home if propio == "home" else c_away].values
                d[f"{nombre}_opp"] = df[c_away if propio == "home" else c_home].values
            partes.append(d)
        out = pd.concat(partes, ignore_index=True)
        out["_t"] = pd.to_datetime(out["utc_date"], utc=True, errors="coerce")
        return out.sort_values("_t", ascending=False)

    log = load_log()
    goles = pd.DataFrame()
    if not log.empty:
        log = log[log["status"] == "resolved"].copy()
        for c in ("actual_home_goals", "actual_away_goals"):
            log[c] = pd.to_numeric(log[c], errors="coerce")
        log = log.dropna(subset=["actual_home_goals", "actual_away_goals"]).drop_duplicates(
            ["home_team", "away_team", "utc_date"], keep="last")
        goles = largo(log, {"g": ("actual_home_goals", "actual_away_goals")})
    stats = sm.stat_totals_frame()
    if not stats.empty:
        stats = largo(stats, {c: (f"{c}_home", f"{c}_away") for c in sm.STAT_LINES})
    _FORM_CACHE.clear()
    _FORM_CACHE[clave] = (goles, stats)
    return goles, stats


def _form_event(mercado: str, pick: str, home: str, away: str):
    """(fuente, función(fila, es_el_equipo_del_pick) -> bool, equipo del pick) o None."""
    import re

    m = re.match(r"^\s*(?:(.+?): )?(Más|Menos) de ([0-9]+(?:\.[0-9]+)?)", str(pick))
    if mercado in _STAT_PICK_COL and m:
        col, mas, linea = _STAT_PICK_COL[mercado], m.group(2) == "Más", float(m.group(3))
        if mercado == TEAM_STAT_TIPS_MARKET:
            def ev(f, propio):
                v = f[f"{col}_own"] if propio else f[f"{col}_opp"]
                return v > linea if mas else v < linea
            return "stats", ev, m.group(1)
        return "stats", lambda f, _: (f[f"{col}_own"] + f[f"{col}_opp"] > linea) if mas else \
            (f[f"{col}_own"] + f[f"{col}_opp"] < linea), None
    if mercado in (MORE_CARDS_MARKET, MORE_CORNERS_MARKET, MORE_SOT_MARKET, MORE_OFF_MARKET, MORE_FOULS_MARKET):  # "<equipo> (más X que <rival>)"
        col = _STAT_PICK_COL[mercado]
        equipo = str(pick).split(" (más ")[0]
        return "stats", lambda f, propio: (f[f"{col}_own"] > f[f"{col}_opp"]) if propio else \
            (f[f"{col}_opp"] > f[f"{col}_own"]), equipo
    if mercado in ("Goles", "Goles (otra línea)") and m:
        mas, linea = m.group(2) == "Más", float(m.group(3))
        return "goles", lambda f, _: (f["g_own"] + f["g_opp"] > linea) if mas else (f["g_own"] + f["g_opp"] < linea), None
    if mercado == "Ambos anotan":
        si = str(pick).strip().lower().startswith("s")
        return "goles", lambda f, _: ((f["g_own"] > 0) & (f["g_opp"] > 0)) == si, None

    def margen(cond):  # cond(goles a favor, goles en contra) desde el equipo del pick
        return lambda f, propio: cond(f["g_own"], f["g_opp"]) if propio else cond(f["g_opp"], f["g_own"])

    if mercado == "Resultado":
        if pick == "Empate":
            return "goles", lambda f, _: f["g_own"] == f["g_opp"], None
        return "goles", margen(lambda a, b: a > b), home if pick == "Local" else away
    if mercado == "Doble oportunidad":
        return "goles", margen(lambda a, b: a >= b), home if str(pick).startswith("1X") else away
    if mercado == "Empate no válido":
        return "goles", margen(lambda a, b: a > b), pick
    h = re.match(r"^(.+) ([+-][0-9]+(?:\.[0-9]+)?)$", str(pick))
    if mercado == "Hándicap asiático" and h:
        linea = float(h.group(2))
        return "goles", margen(lambda a, b: a - b + linea > 0), h.group(1)
    return None


_FORM_TEXT_CACHE: dict = {}


def team_form(mercado: str, pick: str, home: str, away: str) -> tuple[str, str]:
    """Textos "70% (7/10)" para el local y el visitante; "–" si el mercado no
    se puede medir así (1er tiempo, primer gol) o el equipo no tiene historial.
    Guardado por pick mientras no cambien los historiales (cada clic de la app
    recalculaba ~190 picks)."""
    _team_history()  # deja en _FORM_CACHE la clave de los historiales vigentes
    clave = (next(iter(_FORM_CACHE), None), mercado, pick, home, away)
    if clave not in _FORM_TEXT_CACHE:
        if len(_FORM_TEXT_CACHE) > 5000:
            _FORM_TEXT_CACHE.clear()
        _FORM_TEXT_CACHE[clave] = _team_form(mercado, pick, home, away)
    return _FORM_TEXT_CACHE[clave]


def _team_form(mercado: str, pick: str, home: str, away: str) -> tuple[str, str]:
    regla = _form_event(mercado, pick, home, away)
    if regla is None:
        return "–", "–"
    fuente, ev, del_pick = regla
    goles, stats = _team_history()
    datos = goles if fuente == "goles" else stats
    textos = []
    for equipo in (home, away):
        if datos.empty:
            textos.append("–")
            continue
        d = datos[datos["team"] == equipo]
        cols = [c for c in d.columns if c.endswith(("_own", "_opp"))]
        d = d.dropna(subset=cols if fuente == "goles" else
                     [c for c in cols if c.split("_")[0] in {"g", _STAT_PICK_COL.get(mercado)}]).head(TEAM_FORM_GAMES)
        if len(d) < TEAM_FORM_MIN:
            textos.append("–")
            continue
        propio = del_pick is None or del_pick == equipo
        si = int(pd.Series(ev(d, propio)).astype(bool).sum())
        textos.append(f"{si / len(d):.0%} ({si}/{len(d)})")
    return textos[0], textos[1]


def team_form_columns(picks: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(columna local, columna visitante) para la tabla de Tips."""
    if picks.empty:
        return pd.Series(dtype=str), pd.Series(dtype=str)
    pares = []
    for r in picks[["mercado", "pick", "home_team", "away_team"]].itertuples():
        try:
            pares.append(team_form(r.mercado, r.pick, r.home_team, r.away_team))
        except Exception:
            pares.append(("–", "–"))
    return (pd.Series([p[0] for p in pares], index=picks.index),
            pd.Series([p[1] for p in pares], index=picks.index))


def strength_marks(picks: pd.DataFrame) -> pd.Series:
    """Etiqueta de fortaleza por fila de candidatos de Tips."""
    if picks.empty:
        return pd.Series(dtype=str)
    return pd.Series([strength_label(market_strength(r.competition, r.mercado, r.odds_key, r.pick))
                      for r in picks[["competition", "mercado", "odds_key", "pick"]].itertuples()], index=picks.index)


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
            bt = bt[~bt["familia"].isin(SPECIAL_EXCLUDED_FAMILIES)
                    & (bt["mercado"].isin(SPECIAL_TIPS_MARKET) | (bt["familia"] == "Estadísticas"))]
            bajos = bt["mercado"].isin(sm.LOWER_PROB_MARKETS)
            total = sm.summarize(bt[~bajos])
            total["veredicto"] = total.apply(sm.verdict, axis=1)
            buenos = total[total["veredicto"].isin(SPECIAL_VERDICTS)]
            correccion = ((buenos["acierto_pct"] - buenos["esperado_pct"]) / 100).set_axis(buenos["mercado"]).to_dict()
            # Mercados con umbral 65%: aprobados por calibración (no por veredicto
            # a 75%); la corrección solo resta, nunca infla la probabilidad.
            bajo = sm.summarize(bt[bajos], min_prob=sm.LOWER_PROB)
            correccion |= ((bajo["acierto_pct"] - bajo["esperado_pct"]).clip(upper=0) / 100
                           ).set_axis(bajo["mercado"]).to_dict()
            liga = pd.concat([sm.summarize(bt[~bajos & bt["mercado"].isin(correccion)], by=["competition"]),
                              sm.summarize(bt[bajos], min_prob=sm.LOWER_PROB, by=["competition"])])
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
        # Copa BetPlay: reglas, reparto por tiempo y estadísticas de la Liga BetPlay.
        comp = config.strength_code(r.get("competition"))
        try:
            hx, ax = float(r["home_xg"]), float(r["away_xg"])
        except (TypeError, ValueError, KeyError):
            continue
        fav_local = hx >= ax
        # "Baja confianza" es del modelo de GOLES (pocos partidos propios en la
        # copa, típico de Europa/Conference League en fase de liga): en esos
        # partidos no hay picks de goles, pero sí de estadísticas, que salen de
        # otro modelo (corners, tarjetas... de match_stats_log).
        baja = str(r.get("low_confidence", False)).lower() in ("true", "1")
        probs_goles = [] if baja else \
            sm.goal_market_probs(sm.Match(hx, ax, share.get(comp, sm.DEFAULT_FIRST_HALF_SHARE)))
        # Menos de 3.5 va por regla de consenso (CONSENSUS_MARKETS) en cualquier liga.
        picks = [(merc, prob, p_dev) for _, merc, prob, p_dev in probs_goles
                 if (merc in CONSENSUS_MARKETS and prob >= CONSENSUS_MIN_PROB)
                 or (merc not in CONSENSUS_MARKETS and (comp, merc) in permitidos and prob >= sm.min_prob_for(merc))]
        # Respaldo de goles: el 2.5 no llega al umbral -> Más de 1.5 (recalibrado).
        respaldo = {}
        if probs_goles and "Goles" not in EXCLUDED_MARKETS.get(comp, set()) \
                and GOALS_FALLBACK_MARKET not in [p[0] for p in picks]:
            from .calibration import recalibrate_over
            try:
                p25 = recalibrate_over(float(r["over_2_5"]) / 100)
            except (TypeError, ValueError, KeyError):
                p25 = None
            if p25 is not None and max(p25, 1 - p25) < MIN_MODEL_PROB:
                for _, merc, prob, p_dev in probs_goles:
                    p15 = recalibrate_over(prob, 1.5) if merc == GOALS_FALLBACK_MARKET else 0
                    if p15 >= MIN_MODEL_PROB:
                        picks.append((merc, p15, p_dev))
                        respaldo[merc] = p25
        # Tiros al arco / fueras de juego en las líneas que el backtest aprueba en la liga.
        stat_picks = []
        for col, nombre in STAT_TIPS.items():
            for linea in sm.STAT_LINES[col][2]:
                pr = sm.stat_prediction(comp, r["home_team"], r["away_team"], col, linea)
                if not pr:
                    break
                for lado, prob in (("más", pr["p_mas"]), ("menos", 1 - pr["p_mas"])):
                    merc = f"{nombre}: {lado} de {linea:g}"
                    if ((comp, merc) in permitidos and prob >= sm.min_prob_for(merc)) \
                            or (merc == CORNERS_PRIORITY_MARKET and prob >= CORNERS_PRIORITY_PROB):
                        stat_picks.append((merc, nombre, lado, linea, prob, pr["esperado"], None))
        # Por equipo (tiros al arco del local / del visitante).
        for col, (nombre, lineas_eq) in sm.TEAM_STAT_LINES.items():
            for side, equipo in (("home", r["home_team"]), ("away", r["away_team"])):
                for linea in lineas_eq:
                    pr = sm.stat_prediction_side(comp, r["home_team"], r["away_team"], col, side, linea)
                    if not pr:
                        break
                    for lado, prob in (("más", pr["p_mas"]), ("menos", 1 - pr["p_mas"])):
                        merc = f"{nombre} {sm.SIDE_NAMES[side]}: {lado} de {linea:g}"
                        if (comp, merc) in permitidos and prob >= sm.min_prob_for(merc):
                            stat_picks.append((merc, TEAM_STAT_TIPS_MARKET, lado, linea, prob, pr["esperado"], equipo))
        # Más tarjetas / corners / tiros al arco: el equipo con más probabilidad de superar al rival.
        mas_filas = []
        for mercado_mt, funcion, umbral, que, verbo, backtest in MORE_MARKETS:
            mt = getattr(sm, funcion)(comp, r["home_team"], r["away_team"])
            if not mt:
                continue
            lado_mt = "home" if mt["home"] >= mt["away"] else "away"
            if mt[lado_mt] < umbral:
                continue
            hs, as_ = r.get("home_team_short") or r["home_team"], r.get("away_team_short") or r["away_team"]
            equipo_mt, rival_mt = (r["home_team"], r["away_team"]) if lado_mt == "home" else (r["away_team"], r["home_team"])
            corto_mt, rival_corto = (hs, as_) if lado_mt == "home" else (as_, hs)
            corto_que = que.split(" (")[0]
            mas_filas.append({
                "competition": r.get("competition"), "competition_name": r.get("competition_name", ""),
                "utc_date": r["utc_date"], "home_team": r["home_team"], "away_team": r["away_team"],
                "home_team_short": hs, "away_team_short": as_, "partido": f"{hs} vs {as_}",
                "mercado": mercado_mt, "pick": f"{equipo_mt} (más {corto_que} que {rival_mt})",
                "odds_key": f"especial:{mercado_mt}", "prob_modelo": mt[lado_mt],
                "cuota_justa": round(1 / mt[lado_mt], 2), "cuota_auto": None, "casa": "",
                "peso_modelo": SPECIAL_MODEL_WEIGHT,
                "explicacion": (
                    f"Gana si {corto_mt} {verbo} MÁS {que} que {rival_corto}; si empatan, pierde (el empate es otra "
                    f"opción). Modelo: {hs} {mt['home']:.0%} · empate {mt['draw']:.0%} · {as_} {mt['away']:.0%}; "
                    f"{corto_que} esperad{'as' if corto_que == 'tarjetas' else 'os'} {mt['esp_home']:.1f} vs "
                    f"{mt['esp_away']:.1f}. Backtest: {backtest}. "
                    "BWIN no lo publica: escribe la cuota de tu casa en 'Cuota manual'."),
            })
        if not picks and not stat_picks and not mas_filas:
            continue
        lineas = lines_for(r)
        if lineas is None or lineas.empty:
            lineas = pd.DataFrame(columns=["line_type_id", "option", "line", "odds"])
        # Fueras de juego / tiros al arco: fuera si BWIN no abre ese mercado.
        tipos = set(pd.to_numeric(lineas["line_type_id"], errors="coerce").dropna().astype(int))
        stat_picks = [s for s in stat_picks
                      if BWIN_REQUIRED_MARKET.get(s[1]) is None or BWIN_REQUIRED_MARKET[s[1]] in tipos]
        filas += [f for f in mas_filas
                  if BWIN_REQUIRED_MARKET.get(f["mercado"]) is None or BWIN_REQUIRED_MARKET[f["mercado"]] in tipos]
        if lineas.empty and not stat_picks:
            continue
        home_short = r.get("home_team_short") or r["home_team"]
        away_short = r.get("away_team_short") or r["away_team"]
        fav, und = (r["home_team"], r["away_team"]) if fav_local else (r["away_team"], r["home_team"])
        for merc, nombre, lado, linea, prob, esperado, equipo in stat_picks:
            # Sin esa línea en BWIN queda sin cuota: el pick se ve en Tips como
            # "Falta cuota" y se puede escribir la de otra casa a mano.
            cuota = sm.event_or_stat_price(merc, lineas) if not lineas.empty else None
            prob_cal = min(max(prob + correccion.get(merc, 0.0), 0.01), 0.99)
            filas.append({
                "competition": r.get("competition"), "competition_name": r.get("competition_name", ""),
                "utc_date": r["utc_date"], "home_team": r["home_team"], "away_team": r["away_team"],
                "home_team_short": home_short, "away_team_short": away_short,
                "partido": f"{home_short} vs {away_short}", "mercado": nombre,
                "pick": (f"{equipo}: " if equipo else "") + f"{lado.capitalize()} de {linea:g} (esp. {esperado:.1f})",
                "odds_key": f"especial:{merc}", "prob_modelo": prob_cal, "cuota_justa": round(1 / prob_cal, 2),
                "cuota_auto": cuota, "casa": "BWIN" if cuota else "", "peso_modelo": SPECIAL_MODEL_WEIGHT,
                "explicacion": sm.market_help(merc) + (
                    "" if cuota else
                    " BWIN abre este mercado en el partido, pero no esta línea: escribe la cuota de tu casa en "
                    "'Cuota manual'." if nombre in BWIN_REQUIRED_MARKET else
                    " BWIN no ofrece esta línea en este partido: escribe la cuota de tu casa en 'Cuota manual'."),
            })
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
            # El respaldo ya viene recalibrado: no se le suma la corrección del backtest.
            prob_cal = min(max(prob + (0.0 if merc in respaldo else correccion.get(merc, 0.0)), 0.01), 0.99)
            filas.append({
                "competition": r.get("competition"), "competition_name": r.get("competition_name", ""),
                "utc_date": r["utc_date"], "home_team": r["home_team"], "away_team": r["away_team"],
                "home_team_short": home_short, "away_team_short": away_short,
                "partido": f"{home_short} vs {away_short}", "mercado": mercado, "pick": pick,
                "odds_key": f"especial:{merc}", "prob_modelo": prob_cal, "cuota_justa": round(1 / prob_cal, 2),
                "cuota_auto": precio[0], "casa": "BWIN", "peso_modelo": SPECIAL_MODEL_WEIGHT,
                "consenso": merc in CONSENSUS_MARKETS, "prob_raw": prob,
                "explicacion": sm.market_help(merc)
                + (f" (Si se devuelve: {p_dev:.0%} de probabilidad; la prob. es sin contar devolución.)"
                   if p_dev else "")
                + (f" Regla de consenso: se apuesta solo si el modelo ({prob:.0%}) y la casa coinciden a ≤10 pts; "
                   "monto fijo chico." if merc in CONSENSUS_MARKETS else "")
                + (f" Respaldo de goles: el 2.5 no llegó al {MIN_MODEL_PROB:.0%} (más {respaldo[merc]:.0%} / menos "
                   f"{1 - respaldo[merc]:.0%})." if merc in respaldo else ""),
            })
    return pd.DataFrame(filas)
