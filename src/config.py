"""Configuración central del pipeline: credenciales y códigos de competición."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# La consola de Windows suele quedar en cp1252, que no puede imprimir acentos ni
# símbolos como "⚠" y hace crashear los prints. Forzamos UTF-8 en stdout/stderr.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# Códigos de competición de football-data.org para las 5 grandes ligas + Champions.
COMPETITIONS = {
    "PL": "Premier League",
    "PD": "LaLiga",
    "SA": "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
    "CL": "Champions League",
}

# Plan gratuito de football-data.org: 10 peticiones/minuto.
REQUESTS_PER_MINUTE = 10
