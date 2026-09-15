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

# Mercados de corners/faltas/tarjetas: se logean con la MISMA tendencia que ve
# el usuario en "Detalle por partido" (match_tendencies.py), usando el promedio
# histórico disponible en el momento de loguear — por eso solo existen desde que
# se empezaron a registrar (no hay forma honesta de "reconstruir" qué tendencia
# se habría mostrado para un partido de antes de que existiera esta columna).
STAT_MARKETS = ["corners", "faltas", "tarjetas"]

COLUMNS = [
    "logged_at", "resolved_at", "competition", "season", "matchday",
    "home_team", "away_team", "source",
    "pred_home_pct", "pred_draw_pct", "pred_away_pct",
    "home_xg", "away_xg", "over_2_5_pct", "btts_pct", "low_confidence",
    "status", "actual_home_goals", "actual_away_goals", "actual_result",
    "favored_side", "hit",
    # Resultado se resuelve arriba (hit); estos dos se resuelven en
    # resolve_predictions.py a partir de datos que YA se logueaban
    # (over_2_5_pct/btts_pct + el resultado real), sin necesitar nada nuevo.
    "actual_over_2_5", "hit_over_2_5",
    "actual_btts", "hit_btts",
] + [
    f"pred_{market}_{campo}"
    for market in STAT_MARKETS
    for campo in ("expected", "line", "probable")
] + [
    f"actual_{market}_total" for market in STAT_MARKETS
] + [
    f"hit_{market}" for market in STAT_MARKETS
]

KEY_COLUMNS = ["competition", "season", "matchday", "home_team", "away_team"]

# Un mercado por fila, para poder agrupar "qué tan bien acierta cada mercado,
# por liga" en un solo groupby — ver app.py, sección "Tendencia por liga".
MARKETS = [
    ("Resultado (1X2)", "hit"),
    ("Goles (Over/Under 2.5)", "hit_over_2_5"),
    ("Ambos anotan", "hit_btts"),
    ("Corners", "hit_corners"),
    ("Faltas", "hit_faltas"),
    ("Tarjetas", "hit_tarjetas"),
]


def favored_side(h: float, d: float, a: float) -> str:
    return max((("H", h), ("D", d), ("A", a)), key=lambda x: x[1])[0]


def result_letter(home_goals: float, away_goals: float) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def over_2_5_hit(pred_over_2_5_pct: float, actual_home_goals: float, actual_away_goals: float) -> tuple[bool, bool]:
    """(acierto, resultado_real) del mercado Over/Under 2.5 — a partir de datos
    que YA se logueaban (la predicción se guarda desde siempre en over_2_5_pct),
    así que este mercado se puede resolver incluso para partidos antiguos."""
    actual_over = (actual_home_goals + actual_away_goals) > 2.5
    predicted_over = pred_over_2_5_pct >= 50
    return predicted_over == actual_over, actual_over


def btts_hit(pred_btts_pct: float, actual_home_goals: float, actual_away_goals: float) -> tuple[bool, bool]:
    """(acierto, resultado_real) del mercado Ambos anotan — mismo caso que
    over_2_5_hit: resoluble con datos que ya estaban en el log."""
    actual_btts_ = actual_home_goals > 0 and actual_away_goals > 0
    predicted_btts = pred_btts_pct >= 50
    return predicted_btts == actual_btts_, actual_btts_


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
