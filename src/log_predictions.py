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
from .poisson_model import predict_match
from .predict_matchday import upcoming_fixtures
from .prediction_log import append_predictions, favored_side
from .team_strength import confidence_note, team_strength_for_competition


def log_competition(code: str, season: int, matchday: int, seasons_back: int = 1) -> list[dict]:
    seasons = [season - i for i in range(seasons_back + 1)]
    strength = team_strength_for_competition(code, seasons)
    fixtures = upcoming_fixtures(code, season, matchday)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for _, row in fixtures.iterrows():
        try:
            pred = predict_match(strength, row["home_team"], row["away_team"])
        except KeyError:
            continue
        note = confidence_note(strength, row["home_team"], row["away_team"])
        rows.append({
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
        })
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
