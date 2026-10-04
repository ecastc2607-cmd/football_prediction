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
    "NT": "Selecciones (Nations League)",
}

# Competiciones que NO vienen de football-data.org (ver comentario arriba).
# "NT" (selecciones) tampoco: football-data.org solo tiene Mundial/Eurocopa, y
# nada de eso corre fuera de esos torneos — ver src/national_teams.py.
GOAL_API_COMPETITIONS = {"EL", "NT"}

# Qué módulo aislado resuelve cada competición fuera de football-data.org. Los
# puntos de integración (app.py, resolve_predictions.py, fetch_football_data.py)
# despachan por acá en vez de tener el nombre del módulo repetido/hardcodeado
# en cada uno — agregar una fuente nueva es sumar una entrada acá.
def goal_api_module(code: str):
    if code == "EL":
        from . import europa_league
        return europa_league
    if code == "NT":
        from . import national_teams
        return national_teams
    raise KeyError(f"'{code}' no es una competición de Goal API conocida.")

# Ligas "de apoyo": NO se muestran ni se predicen en el dashboard; solo aportan
# la fuerza doméstica de equipos de Europa League que no juegan en las 5
# grandes (AZ, NEC, Benfica...) — ver cross_competition_strength.py. Están en el
# plan gratis de football-data.org. Se guardan versionadas en git (no en
# data/processed/, que no se versiona y se pierde en cada reinicio de
# Streamlit Cloud) — ver support_leagues.py.
SUPPORT_LEAGUES = {"DED": "Eredivisie", "PPL": "Primeira Liga"}
SUPPORT_LEAGUES_DIR = ROOT_DIR / "data" / "tracking" / "support_leagues"


def matches_path(code: str, season: int) -> Path:
    """CSV de partidos de una competición/temporada: data/processed/ si existe,
    si no la copia versionada de las ligas de apoyo (si existe)."""
    procesado = PROCESSED_DIR / f"matches_{code}_{season}.csv"
    if procesado.exists():
        return procesado
    versionado = SUPPORT_LEAGUES_DIR / f"matches_{code}_{season}.csv"
    return versionado if versionado.exists() else procesado


# Competiciones tipo copa donde, si un equipo no tiene NINGÚN partido propio
# todavía (rotación normal del torneo), se usa su fuerza calculada en su liga
# doméstica como respaldo en vez de omitir el partido — ver
# src/cross_competition_strength.py. Distinto a GOAL_API_COMPETITIONS a
# propósito: hoy coinciden en "EL", pero son dos decisiones independientes
# (de dónde vienen los partidos vs. qué hacer si falta la fuerza de un equipo).
CUP_STYLE_COMPETITIONS = {"EL"}

# Plan gratuito de football-data.org: 10 peticiones/minuto.
REQUESTS_PER_MINUTE = 10
