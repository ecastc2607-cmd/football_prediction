"""
Carga única de estadísticas finales (corners, faltas, tarjetas...) de la Liga
BetPlay ya jugada, desde 365scores, a data/tracking/match_stats_log.csv —
para que el modelo de corners/tarjetas (totals_model.py) tenga base en
Colombia. Se salta lo que ya esté guardado; después lo mantiene al día la
captura automática (capture_live_odds.capture_final_stats).

Uso:
    python -m src.col_stats_backfill
"""
from __future__ import annotations

import time

import pandas as pd

from . import colombia, scores365
from .match_stats_log import append_rows, build_rows, load_log

PAUSE_SECONDS = 0.4  # servicio no oficial: sin apuro


def main() -> None:
    hist = colombia._load_history()
    hist = hist[hist["status_group"] == 4]
    log = load_log()
    ya = set(zip(log["competition"], log["utc_date"].astype(str).str[:16], log["home_team"])) if not log.empty else set()
    pendientes = [r for r in hist.itertuples() if ("COL", str(r.start_utc)[:16], r.home_team) not in ya]
    print(f"{len(pendientes)} partidos sin estadísticas guardadas")
    filas, hechos = [], 0
    for i, r in enumerate(pendientes, 1):
        stats = scores365.live_stats(int(r.id), r.home_id, r.away_id)
        if stats:
            season = int(str(r.start_utc)[:4])
            filas += build_rows("COL", season, 0, r.start_utc, r.home_team, r.away_team,
                                {r.home_team: stats["home"], r.away_team: stats["away"]})
            hechos += 1
        if i % 50 == 0 or i == len(pendientes):
            append_rows(filas)
            filas = []
            print(f"  {i}/{len(pendientes)} consultados, {hechos} con estadísticas")
        time.sleep(PAUSE_SECONDS)


if __name__ == "__main__":
    main()
