"""
Exporta a JSON las predicciones de la jornada de fin de semana de las 5 grandes ligas,
para alimentar el dashboard (evita retranscribir números a mano).

Uso:
    python -m src.export_weekend_predictions > data/tracking/weekend_predictions.json
"""
from __future__ import annotations

import pandas as pd

from .matchday_predictions import matchday_predictions_df

WEEKEND_MATCHDAYS = {
    "PL": 4,
    "PD": 5,
    "SA": 4,
    "BL1": 3,
    "FL1": 4,
}


def main():
    frames = [matchday_predictions_df(code, 2026, matchday) for code, matchday in WEEKEND_MATCHDAYS.items()]
    frames = [f for f in frames if not f.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(combined.to_json(orient="records", force_ascii=False, indent=2))


if __name__ == "__main__":
    main()

