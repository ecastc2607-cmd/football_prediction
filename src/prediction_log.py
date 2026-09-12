"""
Esquema y utilidades compartidas del registro de calibración
(data/tracking/predictions_log.csv).

Flujo: se loguea una predicción ANTES de que se juegue el partido (status=pending),
y se resuelve DESPUÉS con el resultado real (status=resolved). Loguear antes evita
el sesgo retrospectivo de "ya sabía que iba a ganar".
"""
from __future__ import annotations

import pandas as pd

from . import config

TRACKING_DIR = config.ROOT_DIR / "data" / "tracking"
LOG_PATH = TRACKING_DIR / "predictions_log.csv"

COLUMNS = [
    "logged_at", "resolved_at", "competition", "season", "matchday",
    "home_team", "away_team", "source",
    "pred_home_pct", "pred_draw_pct", "pred_away_pct",
    "home_xg", "away_xg", "over_2_5_pct", "btts_pct", "low_confidence",
    "status", "actual_home_goals", "actual_away_goals", "actual_result",
    "favored_side", "hit",
]

KEY_COLUMNS = ["competition", "season", "matchday", "home_team", "away_team"]


def favored_side(h: float, d: float, a: float) -> str:
    return max((("H", h), ("D", d), ("A", a)), key=lambda x: x[1])[0]


def result_letter(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(LOG_PATH)


def save_log(df: pd.DataFrame) -> None:
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    df = df.reindex(columns=COLUMNS)
    df.to_csv(LOG_PATH, index=False, encoding="utf-8")


def append_predictions(new_rows: list[dict]) -> tuple[int, int]:
    """Agrega filas nuevas al log, sin duplicar una predicción ya logueada para el
    mismo partido (misma competición+temporada+jornada+local+visitante).
    Devuelve (agregadas, saltadas_por_duplicado).
    """
    log = load_log()
    existing_keys = set(
        tuple(row) for row in log[KEY_COLUMNS].itertuples(index=False)
    ) if not log.empty else set()

    added, skipped = 0, 0
    rows_to_add = []
    for row in new_rows:
        key = tuple(row[c] for c in KEY_COLUMNS)
        if key in existing_keys:
            skipped += 1
            continue
        rows_to_add.append(row)
        existing_keys.add(key)
        added += 1

    if rows_to_add:
        log = pd.concat([log, pd.DataFrame(rows_to_add)], ignore_index=True)
        save_log(log)
    return added, skipped
