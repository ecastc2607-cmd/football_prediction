"""
Deja listos los CSV de partidos (data/processed/, que no se versiona) de todas
las competiciones, temporada en curso y anterior — lo que necesitan el modelo y
la captura automática. Pensado para la captura en GitHub Actions
(.github/workflows/capture.yml), cuyo servidor arranca vacío en cada corrida:
se corre una vez por día y el resultado queda en la caché de Actions.

Gasta poco: las temporadas cerradas salen de data/tracking/closed_seasons/,
las copas y Colombia de sus historiales versionados (solo se pide lo nuevo) y
las selecciones de nt_history.csv. Cada competición va aislada: si una falla,
las demás siguen.

Uso:
    python -m src.refresh_data
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone

from . import config, season_store
from .fetch_football_data import FootballDataClient, fetch_competition

FOOTBALL_DATA = ["PL", "PD", "SA", "BL1", "FL1", "CL"]


def european_season(now: datetime) -> int:
    """Temporada europea en curso (año de inicio): desde julio, la del año."""
    return now.year if now.month >= 7 else now.year - 1


def main() -> int:
    now = datetime.now(timezone.utc)
    temporada = european_season(now)
    errores = 0
    client = FootballDataClient()
    for code in FOOTBALL_DATA:
        for season in (temporada, temporada - 1):
            if season_store.closed_path(code, season).exists():
                continue  # cerrada y versionada: no se vuelve a pedir
            try:
                df = fetch_competition(client, code, season)
                season_store.save_if_closed(df, code, season)
            except Exception:
                errores += 1
                print(f"ERROR {code} {season}: {traceback.format_exc().splitlines()[-1]}", file=sys.stderr)
    for code in sorted(config.GOAL_API_COMPETITIONS):
        modulo = config.goal_api_module(code)
        try:
            actual = modulo.current_season()
            for season in (actual, actual - 1):
                modulo.fetch_competition(season)
        except Exception:
            errores += 1
            print(f"ERROR {code}: {traceback.format_exc().splitlines()[-1]}", file=sys.stderr)
    try:
        from .support_leagues import ensure_support_leagues
        ensure_support_leagues(temporada)
    except Exception:
        errores += 1
        print(f"ERROR ligas de apoyo: {traceback.format_exc().splitlines()[-1]}", file=sys.stderr)
    print(f"Datos listos ({errores} competición(es) con error).")
    return 0  # nunca corta la captura: con lo que haya, se sigue


if __name__ == "__main__":
    sys.exit(main())
