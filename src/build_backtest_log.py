"""
Carga retroactiva, en el esquema unificado de predictions_log.csv, de las 18
predicciones cualitativas ("a ojo") del dashboard de la Jornada 1 de Champions
(2026-09-07), ya resueltas contra el resultado real.

De aquí en adelante las predicciones se loguean con log_predictions.py (antes del
partido) y se resuelven con resolve_predictions.py (después) — este script es
solo para no perder el primer punto de calibración del proyecto, que se hizo antes
de que existiera el registro unificado.

Uso:
    python -m src.build_backtest_log
"""
from __future__ import annotations

import pandas as pd

from . import config
from .prediction_log import append_predictions, favored_side, result_letter

# Predicciones cualitativas publicadas en el dashboard de la Jornada 1 (2026-09-07),
# antes de que se jugaran los partidos.
JORNADA1_DASHBOARD = [
    ("AEK Athens", "LASK", 50, 28, 22),
    ("Club Brugge", "Aston Villa", 38, 27, 35),
    ("Borussia Dortmund", "Villarreal", 52, 24, 24),
    ("Porto", "Manchester City", 22, 23, 55),
    ("Lille", "Real Betis", 39, 27, 34),
    ("Real Madrid", "Inter", 50, 27, 23),
    ("Barcelona", "Feyenoord", 68, 19, 13),
    ("Stuttgart", "Viking", 66, 20, 14),
    ("Liverpool", "Atlético de Madrid", 48, 28, 24),
    ("Paris Saint-Germain", "Slovan Bratislava", 62, 20, 18),
    ("Sporting CP", "Galatasaray", 40, 26, 34),
    ("Napoli", "Arsenal", 32, 27, 41),
    ("Fenerbahçe", "Roma", 37, 28, 35),
    ("PSV Eindhoven", "Shakhtar Donetsk", 51, 27, 22),
    ("Como", "Leipzig", 33, 28, 39),
    ("Bayern München", "Bodø/Glimt", 78, 14, 8),
    ("Manchester United", "Sabah", 80, 13, 7),
    ("Slavia Praha", "Lens", 34, 28, 38),
]


def main():
    actual = pd.read_csv(config.PROCESSED_DIR / "matches_CL_2026.csv")
    actual = actual[actual["matchday"] == 1]

    rows = []
    for home, away, h, d, a in JORNADA1_DASHBOARD:
        match = actual[
            actual["home_team"].str.contains(home.split()[0], case=False, na=False)
            & actual["away_team"].str.contains(away.split()[0], case=False, na=False)
        ]
        if match.empty:
            print(f"(sin match real encontrado para {home} vs {away}, se omite)")
            continue
        m = match.iloc[0]
        actual_result = result_letter(m["home_goals"], m["away_goals"])
        favored = favored_side(h, d, a)
        rows.append({
            "logged_at": "2026-09-07T00:00:00Z",
            "resolved_at": "2026-09-11T00:00:00Z",
            "competition": "CL",
            "season": 2026,
            "matchday": 1,
            "home_team": home,
            "away_team": away,
            "source": "dashboard_v1_qualitative",
            "pred_home_pct": h,
            "pred_draw_pct": d,
            "pred_away_pct": a,
            "home_xg": "",
            "away_xg": "",
            "over_2_5_pct": "",
            "btts_pct": "",
            "low_confidence": False,
            "status": "resolved",
            "actual_home_goals": int(m["home_goals"]),
            "actual_away_goals": int(m["away_goals"]),
            "actual_result": actual_result,
            "favored_side": favored,
            "hit": favored == actual_result,
        })

    added, skipped = append_predictions(rows)
    print(f"{added} predicciones de la Jornada 1 de Champions migradas al log unificado "
          f"({skipped} ya estaban).")


if __name__ == "__main__":
    main()
