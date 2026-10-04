"""
Backtest "walk-forward" de partidos YA JUGADOS que nunca se loguearon antes de
jugarse (ej. Nations League y Europa League, integradas con la temporada ya
en marcha): para cada partido se recalcula la predicción usando SOLO los
partidos terminados ANTES de su hora de inicio — lo que el modelo habría dicho
en ese momento — y se resuelve contra el resultado real.

No es lo mismo que una predicción logueada de verdad antes del partido, y por
eso queda marcada con source = SOURCE y el dashboard la muestra aparte:
  - el modelo se diseñó viendo datos de esta misma época;
  - el historial de selecciones se armó con la lista de equipos de la edición
    actual (eso sí se sabía antes, pero el archivo se bajó después).
El corte por fecha evita lo grave (usar el resultado del propio partido o de
partidos posteriores), así que sirve como referencia de calibración — no como
reemplazo del registro real.

Corners/faltas/tarjetas no se backtestean: para elegir una tendencia hace
falta historial de estadísticas ANTERIOR al partido, y para estas
competiciones no existe (match_stats_log.csv no tenía nada de ellas).

Un partido que ya tiene predicción en el log (real o de un backtest previo)
nunca se pisa.

Uso:
    python -m src.backtest_walkforward --comp NT --season 2026
    python -m src.backtest_walkforward --comp EL --season 2026
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from . import config
from .cross_competition_strength import fill_missing_with_domestic_strength
from .poisson_model import predict_match
from .prediction_log import (
    KEY_COLUMNS,
    append_predictions,
    btts_hit,
    favored_side,
    load_log,
    over_2_5_hit,
    result_letter,
)
from .team_strength import build_team_strength, confidence_note, load_finished_matches

SOURCE = "backtest_walkforward_v1"


def backtest_competition(code: str, season: int, seasons_back: int = 1) -> tuple[list[dict], list[str]]:
    """(filas_resueltas, avisos) para todos los partidos terminados de
    `season` que todavía no estén en el log."""
    seasons = [season - i for i in range(seasons_back + 1)]
    todos = load_finished_matches(code, seasons)
    todos["_inicio"] = pd.to_datetime(todos["utc_date"], utc=True, errors="coerce")
    jugados = todos[todos["season_start_year"] == season].sort_values("_inicio")

    log = load_log()
    ya_logueados = set(tuple(r) for r in log[KEY_COLUMNS].itertuples(index=False)) if not log.empty else set()

    ahora = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    filas, avisos = [], []
    fuerza_por_inicio: dict = {}
    for _, m in jugados.iterrows():
        clave = (code, season, int(m["matchday"]), m["home_team"], m["away_team"])
        if clave in ya_logueados:
            continue
        inicio = m["_inicio"]
        if inicio not in fuerza_por_inicio:
            previos = todos[todos["_inicio"] < inicio]
            try:
                fuerza_por_inicio[inicio] = build_team_strength(previos) if not previos.empty else None
            except ValueError:
                fuerza_por_inicio[inicio] = None
        strength = fuerza_por_inicio[inicio]
        if strength is None:
            avisos.append(f"{m['home_team']} vs {m['away_team']}: sin partidos previos para calcular fuerza")
            continue

        prestados: dict[str, str] = {}
        if code in config.CUP_STYLE_COMPETITIONS:
            strength, prestados = fill_missing_with_domestic_strength(
                strength, {m["home_team"], m["away_team"]}, seasons, before=inicio
            )
        try:
            pred = predict_match(strength, m["home_team"], m["away_team"])
        except KeyError:
            avisos.append(f"{m['home_team']} vs {m['away_team']}: algún equipo sin datos previos")
            continue

        nota = confidence_note(strength, m["home_team"], m["away_team"])
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        over_pct, btts_pct = round(pred.over_2_5 * 100, 1), round(pred.btts * 100, 1)
        acierto_over, real_over = over_2_5_hit(over_pct, hg, ag)
        acierto_btts, real_btts = btts_hit(btts_pct, hg, ag)
        favorito = favored_side(pred.home_win, pred.draw, pred.away_win)
        real = result_letter(hg, ag)
        filas.append({
            "logged_at": ahora,
            "resolved_at": ahora,
            "competition": code,
            "season": season,
            "matchday": int(m["matchday"]),
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "source": SOURCE,
            "pred_home_pct": round(pred.home_win * 100, 1),
            "pred_draw_pct": round(pred.draw * 100, 1),
            "pred_away_pct": round(pred.away_win * 100, 1),
            "home_xg": round(pred.home_xg, 2),
            "away_xg": round(pred.away_xg, 2),
            "over_2_5_pct": over_pct,
            "btts_pct": btts_pct,
            "low_confidence": nota is not None or bool(prestados),
            "status": "resolved",
            "actual_home_goals": hg,
            "actual_away_goals": ag,
            "actual_result": real,
            "favored_side": favorito,
            "hit": favorito == real,
            "actual_over_2_5": real_over,
            "hit_over_2_5": acierto_over,
            "actual_btts": real_btts,
            "hit_btts": acierto_btts,
        })
    return filas, avisos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comp", required=True)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--seasons-back", type=int, default=1)
    args = parser.parse_args()

    code = args.comp.upper()
    filas, avisos = backtest_competition(code, args.season, args.seasons_back)
    agregadas, saltadas = append_predictions(filas)
    nombre = config.COMPETITIONS.get(code, code)
    print(f"{nombre}: {agregadas} partidos jugados backtesteados ({saltadas} ya estaban).")
    if filas:
        df = pd.DataFrame(filas)
        for etiqueta, col in [("Resultado (1X2)", "hit"), ("Goles (O/U 2.5)", "hit_over_2_5"),
                              ("Ambos anotan", "hit_btts")]:
            print(f"  {etiqueta:<17} {int(df[col].sum())}/{len(df)} ({df[col].mean():.0%})")
        por_jornada = df.groupby("matchday")["hit"].agg(["sum", "count"])
        print("  1X2 por jornada:", ", ".join(f"J{j}: {int(r['sum'])}/{int(r['count'])}"
                                             for j, r in por_jornada.iterrows()))
    for aviso in avisos:
        print(f"  (omitido) {aviso}")


if __name__ == "__main__":
    main()
