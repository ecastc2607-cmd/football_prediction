"""
Descarga y mantiene las ligas "de apoyo" (config.SUPPORT_LEAGUES: Eredivisie y
Primeira Liga) que solo sirven de respaldo de fuerza doméstica para equipos de
Europa League — ver cross_competition_strength.py.

Se guardan en data/tracking/support_leagues/ (versionado en git) para que:
  - sobrevivan a los reinicios de Streamlit Cloud sin volver a pedirlas;
  - una consulta ya hecha no se repita: la temporada anterior (cerrada) se
    descarga una sola vez y nunca más; la actual solo si su copia tiene más
    de MAX_AGE_DAYS (2 peticiones a football-data.org por liga, como mucho).

Uso:
    python -m src.support_leagues --season 2026
"""
from __future__ import annotations

import argparse
import time

from . import config
from .fetch_football_data import FootballDataClient, flatten_matches

MAX_AGE_DAYS = 7


def _needs_download(code: str, season: int, current_season: int) -> bool:
    path = config.SUPPORT_LEAGUES_DIR / f"matches_{code}_{season}.csv"
    if not path.exists():
        return True
    if season < current_season:
        return False  # temporada cerrada: lo guardado ya es definitivo
    edad_dias = (time.time() - path.stat().st_mtime) / 86400
    return edad_dias > MAX_AGE_DAYS


def ensure_support_leagues(current_season: int, seasons_back: int = 1,
                           api_key: str = "") -> list[str]:
    """Descarga solo lo que falta o está viejo. Devuelve una línea por archivo
    descargado (vacía si no hizo falta pedir nada). Una liga que falle no
    impide la otra."""
    pendientes = [
        (code, current_season - i)
        for code in config.SUPPORT_LEAGUES
        for i in range(seasons_back + 1)
        if _needs_download(code, current_season - i, current_season)
    ]
    if not pendientes:
        return []

    client = FootballDataClient(api_key=api_key or config.FOOTBALL_DATA_API_KEY)
    config.SUPPORT_LEAGUES_DIR.mkdir(parents=True, exist_ok=True)
    hechos = []
    for code, season in pendientes:
        try:
            df = flatten_matches(client.get_matches(code, season))
        except Exception as e:
            hechos.append(f"{config.SUPPORT_LEAGUES[code]} {season}: no se pudo descargar ({e})")
            continue
        if df.empty:
            continue
        path = config.SUPPORT_LEAGUES_DIR / f"matches_{code}_{season}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        terminados = int((df["status"] == "FINISHED").sum())
        hechos.append(f"{config.SUPPORT_LEAGUES[code]} {season}: {len(df)} partidos ({terminados} terminados)")
    return hechos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    hechos = ensure_support_leagues(args.season)
    print("\n".join(hechos) if hechos else "Ligas de apoyo al día: no hizo falta descargar nada.")


if __name__ == "__main__":
    main()
