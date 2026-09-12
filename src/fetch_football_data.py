"""
Cliente para la API gratuita de football-data.org.

Descarga partidos (jugados y programados) y tablas de posiciones de las 5 grandes
ligas europeas y la Champions League, y los guarda en:
  - data/raw/{codigo}_matches_{temporada}.json   (respuesta cruda de la API)
  - data/processed/matches_{codigo}.csv          (partidos aplanados, listos para el modelo)

Uso:
    python -m src.fetch_football_data --comp PL --season 2025
    python -m src.fetch_football_data --comp all --season 2025
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from . import config


class FootballDataClient:
    def __init__(self, api_key: str = config.FOOTBALL_DATA_API_KEY):
        if not api_key:
            raise RuntimeError(
                "Falta FOOTBALL_DATA_API_KEY. Consigue una key gratuita en "
                "https://www.football-data.org/client/register, cópiala en un "
                "archivo .env (ver .env.example) y vuelve a intentarlo."
            )
        self.session = requests.Session()
        self.session.headers.update({"X-Auth-Token": api_key})
        self._last_request_ts = 0.0

    def _throttle(self):
        # Plan gratuito: máx. 10 peticiones/minuto -> dejamos ~6.5s entre llamadas.
        min_interval = 60.0 / config.REQUESTS_PER_MINUTE + 0.5
        elapsed = time.time() - self._last_request_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def _get(self, path: str, params: dict | None = None) -> dict:
        self._throttle()
        url = f"{config.FOOTBALL_DATA_BASE_URL}{path}"
        resp = self.session.get(url, params=params, timeout=20)
        self._last_request_ts = time.time()
        if resp.status_code == 429:
            print("  -> límite de tasa alcanzado, esperando 60s...", file=sys.stderr)
            time.sleep(60)
            return self._get(path, params)
        resp.raise_for_status()
        return resp.json()

    def get_matches(self, competition_code: str, season: int | None = None) -> dict:
        params = {"season": season} if season else {}
        return self._get(f"/competitions/{competition_code}/matches", params)

    def get_standings(self, competition_code: str, season: int | None = None) -> dict:
        params = {"season": season} if season else {}
        return self._get(f"/competitions/{competition_code}/standings", params)


def flatten_matches(raw: dict) -> pd.DataFrame:
    """Convierte la respuesta cruda de /matches en un DataFrame plano, una fila por partido."""
    rows = []
    for m in raw.get("matches", []):
        score = m.get("score", {}).get("fullTime", {})
        rows.append(
            {
                "match_id": m["id"],
                "competition": raw.get("competition", {}).get("code"),
                "season_start_year": m.get("season", {}).get("startDate", "")[:4],
                "matchday": m.get("matchday"),
                "utc_date": m.get("utcDate"),
                "status": m.get("status"),
                "home_team": m.get("homeTeam", {}).get("name"),
                "away_team": m.get("awayTeam", {}).get("name"),
                "home_goals": score.get("home"),
                "away_goals": score.get("away"),
                "winner": m.get("score", {}).get("winner"),
            }
        )
    return pd.DataFrame(rows)


def save_processed(df: pd.DataFrame, code: str, season: int) -> Path:
    csv_path = config.PROCESSED_DIR / f"matches_{code}_{season}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


def fetch_competition(client: FootballDataClient, code: str, season: int | None):
    stamp = season or datetime.now(timezone.utc).year
    print(f"Descargando {config.COMPETITIONS.get(code, code)} (temporada {stamp})...")
    raw = client.get_matches(code, season)

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = config.RAW_DIR / f"{code}_matches_{stamp}.json"
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    df = flatten_matches(raw)
    csv_path = save_processed(df, code, stamp)

    finished = (df["status"] == "FINISHED").sum()
    print(f"  {len(df)} partidos guardados en {csv_path.relative_to(config.ROOT_DIR)} "
          f"({finished} finalizados, {len(df) - finished} pendientes/en curso).")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comp", default="all",
                         help="Código de competición (PL, PD, SA, BL1, FL1, CL) o 'all'.")
    parser.add_argument("--season", type=int, default=None,
                         help="Año de inicio de temporada, ej. 2025 para 2025/26.")
    args = parser.parse_args()

    client = FootballDataClient()
    codes = list(config.COMPETITIONS) if args.comp == "all" else [args.comp.upper()]

    for code in codes:
        try:
            fetch_competition(client, code, args.season)
        except requests.HTTPError as e:
            print(f"  ERROR en {code}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
