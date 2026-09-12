"""
Resuelve las predicciones "pending" del registro de calibración contra resultados
reales: vuelve a descargar los partidos de cada competición/temporada involucrada,
busca los que ya terminaron y completa resultado real + acierto.

Uso:
    python -m src.resolve_predictions
    python -m src.resolve_predictions --no-fetch   # usa los CSV ya descargados, sin llamar a la API
"""
from __future__ import annotations

import argparse

from . import config
from .fetch_football_data import FootballDataClient, fetch_competition
from .prediction_log import favored_side, load_log, result_letter, save_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-fetch", action="store_true",
                         help="No llamar a la API; usa los CSV procesados ya existentes.")
    args = parser.parse_args()

    log = load_log()
    pending = log[log["status"] == "pending"]
    if pending.empty:
        print("No hay predicciones pendientes por resolver.")
        return

    needed = pending[["competition", "season"]].drop_duplicates()
    if not args.no_fetch:
        client = FootballDataClient()
        for _, row in needed.iterrows():
            fetch_competition(client, row["competition"], int(row["season"]))

    import pandas as pd
    resolved_count = 0
    for _, need in needed.iterrows():
        code, season = need["competition"], int(need["season"])
        path = config.PROCESSED_DIR / f"matches_{code}_{season}.csv"
        if not path.exists():
            continue
        matches = pd.read_csv(path)
        finished = matches[matches["status"] == "FINISHED"]

        mask = (
            (log["competition"] == code)
            & (log["season"] == season)
            & (log["status"] == "pending")
        )
        for idx in log[mask].index:
            row = log.loc[idx]
            game = finished[
                (finished["home_team"] == row["home_team"])
                & (finished["away_team"] == row["away_team"])
                & (finished["matchday"] == row["matchday"])
            ]
            if game.empty:
                continue
            g = game.iloc[0]
            actual = result_letter(g["home_goals"], g["away_goals"])
            log.loc[idx, "actual_home_goals"] = int(g["home_goals"])
            log.loc[idx, "actual_away_goals"] = int(g["away_goals"])
            log.loc[idx, "actual_result"] = actual
            log.loc[idx, "hit"] = row["favored_side"] == actual
            log.loc[idx, "status"] = "resolved"
            log.loc[idx, "resolved_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            resolved_count += 1

    save_log(log)

    still_pending = (log["status"] == "pending").sum()
    print(f"\n{resolved_count} predicciones resueltas. {still_pending} siguen pendientes "
          "(el partido aún no se jugó o no se encontró en los datos frescos).")

    resolved = log[log["status"] == "resolved"]
    if not resolved.empty:
        hit_rate = resolved["hit"].mean()
        print(f"\nAcierto histórico acumulado: {resolved['hit'].sum():.0f}/{len(resolved)} ({hit_rate:.0%})")
        print("\nPor competición:")
        print(resolved.groupby("competition")["hit"].agg(["sum", "count"]))
        print("\nBaja confianza vs. resto:")
        print(resolved.groupby("low_confidence")["hit"].agg(["mean", "count"]))


if __name__ == "__main__":
    main()
