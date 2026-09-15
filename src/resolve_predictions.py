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
import sys

import pandas as pd

from . import config, europa_league
from .fetch_football_data import FootballDataClient, fetch_competition
from .match_stats_log import get_cached_stats
from .prediction_log import MARKETS, STAT_MARKETS, load_log, result_letter, save_log


def _refresh_competition(client: FootballDataClient, code: str, season: int) -> None:
    """Descarga los partidos frescos de una competición/temporada, sea cual sea
    su fuente. La Europa League corre sobre Goal API (no está en el plan gratis
    de football-data.org — ver src/europa_league.py); si SU descarga falla, no
    debe impedir que se resuelvan las predicciones pendientes de las otras 6."""
    if code in config.GOAL_API_COMPETITIONS:
        europa_league.fetch_competition(season)
    else:
        fetch_competition(client, code, season)


def _resolve_stat_markets(code: str, home_team: str, away_team: str, utc_date: str, row) -> dict:
    """Corners/faltas/tarjetas reales para este partido (si ya se consultaron
    alguna vez vía "En vivo"/"Resultados ya jugados"/el backfill — no siempre
    están), comparados contra la tendencia que se logueó ANTES de jugarse.

    Devuelve solo las claves que sí se pudieron resolver: si falta el dato real
    o no se había logueado una tendencia para ese mercado, ese mercado
    simplemente no aparece (queda NaN en el log, no se inventa un resultado)."""
    stats = get_cached_stats(code, home_team, away_team, utc_date)
    if not stats or home_team not in stats or away_team not in stats:
        return {}

    home_s, away_s = stats[home_team], stats[away_team]
    totales = {
        "corners": (home_s.get("corners") or 0) + (away_s.get("corners") or 0),
        "faltas": (home_s.get("faltas") or 0) + (away_s.get("faltas") or 0),
        "tarjetas": (
            (home_s.get("tarjetas_amarillas") or 0) + (home_s.get("tarjetas_rojas") or 0)
            + (away_s.get("tarjetas_amarillas") or 0) + (away_s.get("tarjetas_rojas") or 0)
        ),
    }

    resultado = {}
    for market in STAT_MARKETS:
        linea, probable = row.get(f"pred_{market}_line"), row.get(f"pred_{market}_probable")
        if pd.isna(linea) or pd.isna(probable) or probable == "":
            continue  # no se había logueado tendencia para este mercado en este partido
        total_real = totales[market]
        actual_probable = "Sí" if total_real >= linea else "No"
        resultado[f"actual_{market}_total"] = total_real
        resultado[f"hit_{market}"] = (probable == actual_probable)
    return resultado


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

    # Estas columnas nacen vacías (NaN, dtype float64) al reindexar contra
    # COLUMNS; pasarlas a "object" antes de meterles booleanos/strings evita el
    # FutureWarning de pandas por cambiar de dtype en una asignación .loc.
    for market in STAT_MARKETS:
        log[f"actual_{market}_total"] = log[f"actual_{market}_total"].astype(object)
        log[f"hit_{market}"] = log[f"hit_{market}"].astype(object)

    needed = pending[["competition", "season"]].drop_duplicates()
    if not args.no_fetch:
        client = FootballDataClient()
        for _, row in needed.iterrows():
            try:
                _refresh_competition(client, row["competition"], int(row["season"]))
            except Exception as e:
                # Aislado a propósito: si UNA competición falla al refrescar (ej.
                # Europa League vía Goal API), las demás igual se resuelven.
                print(f"  ERROR refrescando {row['competition']} {row['season']}: {e}", file=sys.stderr)

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
            home_goals, away_goals = g["home_goals"], g["away_goals"]
            actual = result_letter(home_goals, away_goals)
            log.loc[idx, "actual_home_goals"] = int(home_goals)
            log.loc[idx, "actual_away_goals"] = int(away_goals)
            log.loc[idx, "actual_result"] = actual
            log.loc[idx, "hit"] = row["favored_side"] == actual
            # Goles (O/U 2.5) y Ambos anotan se calculan más abajo, de una vez
            # para todos los partidos resueltos (nuevos y viejos) — ver el
            # bloque vectorizado justo antes de save_log().

            # Corners/faltas/tarjetas: solo si además hay estadística real
            # guardada para este partido (no todos los partidos terminados la
            # tienen — depende de si se consultó alguna vez en "En vivo").
            for campo, valor in _resolve_stat_markets(
                code, row["home_team"], row["away_team"], g["utc_date"], row
            ).items():
                log.loc[idx, campo] = valor

            log.loc[idx, "status"] = "resolved"
            log.loc[idx, "resolved_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            resolved_count += 1

    # Goles (O/U 2.5) y Ambos anotan se pueden resolver retroactivamente para
    # partidos YA resueltos antes de que existieran estas columnas: los goles
    # reales y la predicción (over_2_5_pct/btts_pct) siempre se guardaron, así
    # que no hace falta esperar a que se jueguen partidos nuevos para tener
    # calibración de estos dos mercados. Corners/faltas/tarjetas no se pueden
    # reconstruir así (la tendencia no se logueaba en ese momento).
    resolved_mask = log["status"] == "resolved"
    if resolved_mask.any():
        # Estas columnas nacen vacías (NaN, dtype float64) al reindexar contra
        # COLUMNS; pasarlas a "object" antes de meterles booleanos evita el
        # FutureWarning de pandas por cambiar de dtype en una asignación .loc.
        for campo in ("actual_over_2_5", "hit_over_2_5", "actual_btts", "hit_btts"):
            log[campo] = log[campo].astype(object)

        home_g = pd.to_numeric(log.loc[resolved_mask, "actual_home_goals"], errors="coerce")
        away_g = pd.to_numeric(log.loc[resolved_mask, "actual_away_goals"], errors="coerce")
        over_pct = pd.to_numeric(log.loc[resolved_mask, "over_2_5_pct"], errors="coerce")
        btts_pct = pd.to_numeric(log.loc[resolved_mask, "btts_pct"], errors="coerce")

        # Se recalcula desde cero (no se acumula sobre lo que ya hubiera en el
        # CSV) para que quede idempotente: correr esto dos veces da el mismo
        # resultado. Un puñado de filas viejas (el primer lote "cualitativo" de
        # Champions League, de antes del modelo de Poisson) nunca tuvo
        # over_2_5_pct/btts_pct -- sin la predicción original no hay nada que
        # comparar, así que esas quedan explícitamente en None (NaN >= 50 daría
        # un "acierto" falso si no se descartan a propósito).
        log.loc[resolved_mask, "actual_over_2_5"] = None
        log.loc[resolved_mask, "hit_over_2_5"] = None
        log.loc[resolved_mask, "actual_btts"] = None
        log.loc[resolved_mask, "hit_btts"] = None

        con_prediccion_over = over_pct.notna()
        con_prediccion_btts = btts_pct.notna()

        actual_over = (home_g + away_g) > 2.5
        idx_over = resolved_mask[resolved_mask].index[con_prediccion_over.values]
        log.loc[idx_over, "actual_over_2_5"] = actual_over[con_prediccion_over]
        log.loc[idx_over, "hit_over_2_5"] = (over_pct[con_prediccion_over] >= 50) == actual_over[con_prediccion_over]

        actual_bt = (home_g > 0) & (away_g > 0)
        idx_btts = resolved_mask[resolved_mask].index[con_prediccion_btts.values]
        log.loc[idx_btts, "actual_btts"] = actual_bt[con_prediccion_btts]
        log.loc[idx_btts, "hit_btts"] = (btts_pct[con_prediccion_btts] >= 50) == actual_bt[con_prediccion_btts]

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

        print("\nPor mercado (todas las competiciones):")
        for label, campo in MARKETS:
            sub = resolved.dropna(subset=[campo]) if campo in resolved.columns else resolved.iloc[0:0]
            if sub.empty:
                print(f"  {label:25} sin datos suficientes todavía")
                continue
            tasa = sub[campo].mean()
            print(f"  {label:25} {sub[campo].sum():.0f}/{len(sub)} ({tasa:.0%})")


if __name__ == "__main__":
    main()
