"""
Evidencia para los mercados de UN SOLO TIEMPO, sin esperar semanas de registro.

1. build(): backtest walk-forward de todos los partidos ya jugados de la
   temporada (ligas, Champions, Europa League, Nations League): cada partido se
   predice SOLO con partidos anteriores a su inicio — partido completo y cada
   mitad — y se guarda con el resultado real y el marcador al descanso en
   data/tracking/half_markets_backtest.csv (mismo esquema que predictions_log).
   Lo usa la calibración, solo para los mercados por tiempo.

2. compare(): arma las combinadas jornada por jornada sobre esos partidos, CON
   y SIN picks de un solo tiempo, con calibración que nunca ve la jornada que
   evalúa, y compara acierto y retorno. Es la evidencia para decidir
   parlay_builder.HALF_MARKETS_ENABLED.

Uso:
    python -m src.half_backtest
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import config
from .calibration import HALF_MARKETS, Calibrator, leg_won, resolved_rows, row_legs
from .cross_competition_strength import fill_missing_with_domestic_strength
from .half_strength import half_prediction, half_strengths_from_matches
from .parlay_builder import TIERS, build_parlays
from .poisson_model import predict_match
from .team_strength import build_team_strength, confidence_note, load_finished_matches

BACKTEST_PATH = config.ROOT_DIR / "data" / "tracking" / "half_markets_backtest.csv"
SOURCE = "half_walkforward_v1"
COMPETITIONS = ["PL", "PD", "SA", "BL1", "FL1", "CL", "EL", "NT"]


def _rows_for(code: str, season: int) -> list[dict]:
    seasons = [season, season - 1]
    try:
        todos = load_finished_matches(code, seasons)
    except FileNotFoundError:
        return []
    if "home_ht_goals" not in todos.columns:
        return []
    todos = todos.assign(inicio_utc=pd.to_datetime(todos["utc_date"], utc=True, errors="coerce"))
    jugados = todos[(todos["season_start_year"] == season)
                    & todos["home_ht_goals"].notna()].sort_values("inicio_utc")
    filas, cache = [], {}
    for m in jugados.itertuples():
        if m.inicio_utc not in cache:
            previos = todos[todos["inicio_utc"] < m.inicio_utc]
            if len(previos) < 20:
                cache[m.inicio_utc] = (None, None)
            else:
                cache[m.inicio_utc] = (build_team_strength(previos), half_strengths_from_matches(previos))
        fuerza, mitades = cache[m.inicio_utc]
        if fuerza is None:
            continue
        if code in config.CUP_STYLE_COMPETITIONS:
            fuerza, _ = fill_missing_with_domestic_strength(fuerza, {m.home_team, m.away_team}, seasons,
                                                            before=m.inicio_utc)
        try:
            p = predict_match(fuerza, m.home_team, m.away_team)
        except KeyError:
            continue
        por_tiempo = half_prediction(mitades, m.home_team, m.away_team) or {}
        filas.append({
            "logged_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "competition": code, "season": season, "matchday": m.matchday,
            "home_team": m.home_team, "away_team": m.away_team, "utc_date": m.utc_date,
            "source": SOURCE, "status": "resolved",
            "pred_home_pct": round(p.home_win * 100, 1), "pred_draw_pct": round(p.draw * 100, 1),
            "pred_away_pct": round(p.away_win * 100, 1),
            "over_2_5_pct": round(p.over_2_5 * 100, 1), "btts_pct": round(p.btts * 100, 1),
            "low_confidence": confidence_note(fuerza, m.home_team, m.away_team) is not None,
            "h1_home_xg": por_tiempo.get("1T", {}).get("home_xg"),
            "h1_away_xg": por_tiempo.get("1T", {}).get("away_xg"),
            "h2_home_xg": por_tiempo.get("2T", {}).get("home_xg"),
            "h2_away_xg": por_tiempo.get("2T", {}).get("away_xg"),
            "actual_home_goals": int(m.home_goals), "actual_away_goals": int(m.away_goals),
            "actual_ht_home_goals": int(m.home_ht_goals), "actual_ht_away_goals": int(m.away_ht_goals),
        })
    return filas


def build(season: int = 2026) -> pd.DataFrame:
    filas = []
    for code in COMPETITIONS:
        filas.extend(_rows_for(code, season))
    df = pd.DataFrame(filas)
    BACKTEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BACKTEST_PATH, index=False, encoding="utf-8")
    return df


def load() -> pd.DataFrame:
    return pd.read_csv(BACKTEST_PATH) if BACKTEST_PATH.exists() else pd.DataFrame()


def leg_calibration(bt: pd.DataFrame) -> pd.DataFrame:
    """Por mercado de un solo tiempo: picks, acierto real vs probabilidad media del modelo."""
    filas = [(m, p, g) for r in resolved_rows(bt).itertuples() for m, _pick, p, g in row_legs(r)
             if m in HALF_MARKETS]
    df = pd.DataFrame(filas, columns=["Mercado", "p", "gano"])
    return df.groupby("Mercado").agg(Picks=("gano", "size"), Acierto=("gano", "mean"),
                                     Prometido=("p", "mean")).reset_index()


def compare(bt: pd.DataFrame) -> pd.DataFrame:
    """Combinadas jornada por jornada, sin y con picks de un solo tiempo."""
    filas_bt = resolved_rows(bt)
    pred = filas_bt.rename(columns={"pred_home_pct": "home_win", "pred_draw_pct": "draw",
                                    "pred_away_pct": "away_win", "over_2_5_pct": "over_2_5",
                                    "btts_pct": "btts"})
    resultados = []
    for (comp, jornada), idx in filas_bt.groupby(["competition", "matchday"]).groups.items():
        if len(idx) < 2:
            continue
        # Calibración sin esta jornada: mercados completos y por tiempo salen
        # del mismo backtest (comparación pareja entre las dos estrategias).
        resto = filas_bt.drop(index=idx)
        calibracion = Calibrator.from_log(resto, half_backtest=resto)
        goles = {f"{r.home_team} vs {r.away_team}": (r.actual_home_goals, r.actual_away_goals,
                                                     r.actual_ht_home_goals, r.actual_ht_away_goals)
                 for r in filas_bt.loc[idx].itertuples()}
        for con_mitades in (False, True):
            for p in build_parlays(pred.loc[idx], calibracion, include_half=con_mitades):
                acerto = all(leg_won(l.market, l.pick, *goles[l.match]) for l in p.legs)
                resultados.append({
                    "Estrategia": "Con picks por tiempo" if con_mitades else "Solo partido completo",
                    "Nivel": p.risk, "acerto": acerto, "prob": p.combined_probability,
                    "paga": p.estimated_odds,
                    "picks_por_tiempo": sum(l.market in HALF_MARKETS for l in p.legs),
                })
    df = pd.DataFrame(resultados)
    if df.empty:
        return df
    orden = {nombre: i for i, (nombre, _, _) in enumerate(TIERS)}
    resumen = df.groupby(["Estrategia", "Nivel"]).agg(
        Combinadas=("acerto", "size"), Acertadas=("acerto", "sum"),
        Acierto=("acerto", "mean"), Prometido=("prob", "mean"),
        Retorno=("paga", lambda s: (s * df.loc[s.index, "acerto"]).mean()),
        Picks_por_tiempo=("picks_por_tiempo", "mean"),
    ).reset_index()
    resumen["_o"] = resumen["Nivel"].map(orden)
    return resumen.sort_values(["Estrategia", "_o"]).drop(columns="_o")


def main():
    bt = build()
    print(f"{len(bt)} partidos backtesteados en {BACKTEST_PATH.relative_to(config.ROOT_DIR)} "
          f"({bt['h1_home_xg'].notna().sum()} con predicción por tiempo).\n")
    print("Calibración de los picks por tiempo:")
    print(leg_calibration(bt).to_string(index=False, formatters={
        "Acierto": "{:.0%}".format, "Prometido": "{:.0%}".format}))
    print("\nCombinadas con y sin picks por tiempo (calibración sin la jornada evaluada):")
    print(compare(bt).to_string(index=False, formatters={
        "Acierto": "{:.0%}".format, "Prometido": "{:.0%}".format, "Retorno": "{:.2f}".format,
        "Picks_por_tiempo": "{:.1f}".format}))


if __name__ == "__main__":
    main()
