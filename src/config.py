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
# "EL" (Europa League) es la excepción: football-data.org no la incluye en su
# plan gratis (verificado contra su API y su tabla de cobertura — hace falta su
# plan de €49/mes), así que corre sobre Goal API en vez de football-data.org.
# Ver src/europa_league.py: ahí vive TODO lo específico de esa fuente, aislado
# a propósito para que un problema con ella nunca afecte a las otras 6.
COMPETITIONS = {
    "PL": "Premier League",
    "PD": "LaLiga",
    "SA": "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
    "CL": "Champions League",
    "EL": "Europa League",
}

# Competiciones que NO vienen de football-data.org (ver comentario arriba).
GOAL_API_COMPETITIONS = {"EL"}

# Competiciones tipo copa donde, si un equipo no tiene NINGÚN partido propio
# todavía (rotación normal del torneo), se usa su fuerza calculada en su liga
# doméstica como respaldo en vez de omitir el partido — ver
# src/cross_competition_strength.py. Distinto a GOAL_API_COMPETITIONS a
# propósito: hoy coinciden en "EL", pero son dos decisiones independientes
# (de dónde vienen los partidos vs. qué hacer si falta la fuerza de un equipo).
CUP_STYLE_COMPETITIONS = {"EL"}

# Plan gratuito de football-data.org: 10 peticiones/minuto.
REQUESTS_PER_MINUTE = 10
