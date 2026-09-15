"""
Corre el modelo sobre los próximos partidos de una competición/jornada y los
agrega al registro de calibración (data/tracking/predictions_log.csv) como
"pending" — ANTES de que se jueguen, para poder validar después sin sesgo
retrospectivo.

Uso:
    python -m src.log_predictions --comp PL --season 2026 --matchday 4
    python -m src.log_predictions --comp all --season 2026 --matchdays PL=4,PD=5,SA=4,BL1=3,FL1=4
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from . import config
from .cross_competition_strength import fill_missing_with_domestic_strength
from .match_tendencies import load_team_averages, pick_tendencies
from .poisson_model import predict_match
from .predict_matchday import upcoming_fixtures
from .prediction_log import append_predictions, favored_side
from .team_strength import confidence_note, team_strength_for_competition


def log_competition(code: str, season: int, matchday: int, seasons_back: int = 1) -> list[dict]:
    seasons = [season - i for i in range(seasons_back + 1)]
    strength = team_strength_for_competition(code, seasons)
    fixtures = upcoming_fixtures(code, season, matchday)

    if code in config.CUP_STYLE_COMPETITIONS and not fixtures.empty:
        equipos = set(fixtures["home_team"]) | set(fixtures["away_team"])
        strength, _ = fill_missing_with_domestic_strength(strength, equipos, seasons)

    # Tendencia de corners/faltas/tarjetas — mismo promedio histórico que ve el
    # usuario en "Detalle por partido" al momento de loguear. Si la competición
    # todavía no tiene historial en match_stats_log.csv, sale vacío y esos
    # campos quedan en None para todos los partidos (no es un error).
    tendency_averages = load_team_averages(code)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for _, row in fixtures.iterrows():
        try:
            pred = predict_match(strength, row["home_team"], row["away_team"])
        except KeyError:
            continue
        note = confidence_note(strength, row["home_team"], row["away_team"])

        fila = {
            "logged_at": now,
            "resolved_at": "",
            "competition": code,
            "season": season,
            "matchday": matchday,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "source": "model_poisson_v1",
            "pred_home_pct": round(pred.home_win * 100, 1),
            "pred_draw_pct": round(pred.draw * 100, 1),
            "pred_away_pct": round(pred.away_win * 100, 1),
            "home_xg": round(pred.home_xg, 2),
            "away_xg": round(pred.away_xg, 2),
            "over_2_5_pct": round(pred.over_2_5 * 100, 1),
            "btts_pct": round(pred.btts * 100, 1),
            "low_confidence": note is not None,
            "status": "pending",
            "actual_home_goals": "",
            "actual_away_goals": "",
            "actual_result": "",
            "favored_side": favored_side(pred.home_win, pred.draw, pred.away_win),
            "hit": "",
        }

        # pick.label es "Corners"/"Faltas"/"Tarjetas" — en minúscula calza
        # directo con las claves de prediction_log.STAT_MARKETS.
        picks = pick_tendencies(tendency_averages, row["home_team"], row["away_team"])
        for pick in picks:
            prefijo = pick.label.lower()
            fila[f"pred_{prefijo}_expected"] = pick.expected
            fila[f"pred_{prefijo}_line"] = pick.line
            fila[f"pred_{prefijo}_probable"] = pick.probable

        rows.append(fila)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comp", required=True, help="Código de competición, ej. PL")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--matchday", type=int, required=True)
    parser.add_argument("--seasons-back", type=int, default=1)
    args = parser.parse_args()

    rows = log_competition(args.comp.upper(), args.season, args.matchday, args.seasons_back)
    added, skipped = append_predictions(rows)
    comp_name = config.COMPETITIONS.get(args.comp.upper(), args.comp)
    print(f"{comp_name} J{args.matchday}: {added} predicciones logueadas, {skipped} ya existían.")


if __name__ == "__main__":
    main()
