"""
Cliente de Goal API (https://goal-api.com) — reemplaza a Highlightly como fuente
de corners, faltas y tarjetas por partido.

Por qué se migró (medido, no supuesto):
  - Plan gratis de 1.000 peticiones/día contra las 100 de Highlightly.
  - 1 petición por partido en vez de 2, y el mapa de IDs de una jornada entera
    se resuelve con UNA sola llamada por liga y fecha (que además se cachea).
  - 5 temporadas de histórico disponibles, lo que permite alimentar el modelo
    sin esperar a acumular datos partido por partido.

Devuelve el MISMO formato que highlightly_client.get_match_stats
({nombre_equipo: {corners, faltas, ...}}) para que el resto del proyecto
—match_stats_log, la tabla del dashboard— siga funcionando sin cambios.

Requiere GOAL_API_KEY en el .env (o en los secrets de Streamlit Cloud).
"""
from __future__ import annotations

import os
import re
import unicodedata
from difflib import SequenceMatcher

import requests

from . import config  # noqa: F401  -- importarlo corre load_dotenv() antes del os.getenv de abajo

BASE_URL = "https://api.goal-api.com/v1"
GOAL_API_KEY = os.getenv("GOAL_API_KEY", "")


class GoalApiRateLimited(Exception):
    """Se agotaron las 1.000 peticiones/día del plan gratis. Se distingue de un
    error cualquiera para poder avisarlo en la UI en vez de mostrar
    silenciosamente "sin datos"."""


# IDs fijos de Goal API por competición. Se fijan a mano y NO se buscan por
# nombre en caliente: hay 54 ligas llamadas "Premier League" en su catálogo
# (Rusia, Ucrania, Kenia, Maldivas...) y 9 "Ligue 1" (Argelia, Túnez, Senegal...),
# así que buscar por nombre traería la liga equivocada. Verificados uno por uno
# contra /leagues junto con su país.
LEAGUE_IDS = {
    "PL":  "cmr77dvkr005nrx06lp7rvp49",  # Premier League (England)
    "PD":  "cmr77dvnt006nrx063v3w622e",  # La Liga (Spain)
    "SA":  "cmr77dvpd006yrx06zig7907g",  # Serie A (Italy)
    "BL1": "cmr77dvgm0002rx06rt2uqxii",  # Bundesliga (Germany)
    "FL1": "cmr77dvqg007crx06q1kaceyo",  # Ligue 1 (France)
    "CL":  "cmr77dw3900f5rx06j05wgzv4",  # UEFA Champions League (Europe)
    "EL":  "cmr77dw3900f6rx06tuqwft2d",  # UEFA Europa League (Europe)
}

# Nombres exactos de las filas de /statistics, verificados contra partidos reales
# de Premier League. Ojo: NO coinciden con los de Highlightly ("Yellow cards" allá,
# "Yellow Cards" acá), que es justo el tipo de detalle que ya nos costó una vez.
_STAT_FIELDS = {
    "corners": "Corners",
    "faltas": "Fouls",
    "tarjetas_amarillas": "Yellow Cards",
    "tarjetas_rojas": "Red Cards",
    "posesion": "Ball Possession",
    "remates_a_puerta": "Shots On Goal",
    "remates_totales": "Shots Total",
    "fueras_de_juego": "Offsides",
}

# Estadísticas que, si no aparecen en la respuesta, es porque fueron cero y el
# API simplemente omite la fila (verificado: en un partido sin rojas no viene
# "Red Cards"). Para el resto, ausente significa "no lo sabemos" -> None.
_ZERO_WHEN_MISSING = {"tarjetas_amarillas", "tarjetas_rojas", "fueras_de_juego"}


# Palabras que solo dicen "esto es un club" y estorban al comparar nombres:
# "Stade Rennais FC 1901" y "Rennes" son el mismo equipo, igual que
# "Real Racing Club de Santander" y "Racing Santander".
_NOISE_TOKENS = {
    "fc", "cf", "afc", "ac", "sc", "ss", "as", "us", "cd", "ca", "rc", "rcd",
    "sv", "tsg", "vfb", "vfl", "fsv", "bsc", "sd", "ud", "club", "calcio",
    "de", "del", "la", "el", "football", "futbol", "stade", "olympique",
}


def normalize_team_name(name: str) -> str:
    """Minúsculas, sin acentos, sin puntuación, sin números de fundación ni
    palabras genéricas de club. Público (sin "_"): también lo reutiliza
    cross_competition_strength.py para emparejar nombres de Goal API contra
    los de football-data.org."""
    txt = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    txt = re.sub(r"[^A-Za-z0-9]+", " ", txt).lower().strip()
    tokens = [t for t in txt.split() if t not in _NOISE_TOKENS and not t.isdigit()]
    return " ".join(tokens) or txt


def team_name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_team_name(a), normalize_team_name(b)).ratio()


def _same_kickoff(fixture_kickoff: str, our_utc_date: str) -> bool:
    """Compara la hora de inicio al minuto. Goal API entrega
    '2026-09-11T18:45:00.000Z' y football-data.org '2026-09-11T18:45:00Z':
    el mismo instante escrito distinto."""
    if not fixture_kickoff or not our_utc_date:
        return False
    return str(fixture_kickoff)[:16] == str(our_utc_date)[:16]


def _to_number(value):
    """'40%' -> 40, '4' -> 4, None/'' -> None."""
    if value is None:
        return None
    text = str(value).strip().rstrip("%")
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return None


def _pick_rows(rows: list[dict]) -> dict[str, tuple]:
    """Convierte la lista de filas {type, home, away} en {type: (home, away)}.

    Maneja las dos rarezas reales del API (comprobadas en varios partidos):
      - 'Corners' viene DUPLICADO, siempre con el mismo valor -> da igual cuál.
      - 'Ball Possession' viene duplicado con valores DISTINTOS, y a veces la
        primera copia es '0%'/'0%' basura -> nos quedamos con la que no es cero.
    """
    out: dict[str, tuple] = {}
    for row in rows:
        if not isinstance(row, dict) or "type" not in row:
            continue
        tipo = row["type"]
        home, away = _to_number(row.get("home")), _to_number(row.get("away"))
        if tipo not in out:
            out[tipo] = (home, away)
            continue
        # Ya había una copia: solo la reemplazamos si la vieja era el 0/0 basura
        # y la nueva trae algo real.
        viejo = out[tipo]
        if viejo in ((0, 0), (None, None)) and (home, away) not in ((0, 0), (None, None)):
            out[tipo] = (home, away)
    return out


def _build_team_stats(rows: list[dict], home_team: str, away_team: str) -> dict:
    valores = _pick_rows(rows)
    stats = {home_team: {}, away_team: {}}
    for clave, nombre_api in _STAT_FIELDS.items():
        if nombre_api in valores:
            local, visita = valores[nombre_api]
        elif clave in _ZERO_WHEN_MISSING:
            local, visita = 0, 0  # el API omite la fila cuando fue cero
        else:
            local, visita = None, None
        stats[home_team][clave] = local
        stats[away_team][clave] = visita
    return stats


class GoalApiClient:
    def __init__(self, api_key: str = ""):
        api_key = api_key or GOAL_API_KEY
        if not api_key:
            raise RuntimeError(
                "Falta GOAL_API_KEY. Regístrate gratis (sin tarjeta) en "
                "https://goal-api.com/signup, copia tu key del dashboard, y agrégala al .env."
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def _get(self, path: str, params: dict | None = None):
        resp = self.session.get(f"{BASE_URL}{path}", params=params or None, timeout=20)
        if resp.status_code == 429:
            raise GoalApiRateLimited(
                "Se agotaron las 1.000 peticiones/día del plan gratis de Goal API."
            )
        resp.raise_for_status()
        return resp.json().get("data")

    def fixtures_by_date(self, date_iso: str, competition_code: str | None = None) -> list[dict]:
        """Todos los partidos de una fecha, opcionalmente acotados a una liga.

        UNA llamada resuelve la jornada entera de esa liga, así que el que la use
        debería cachear el resultado en vez de pedirlo partido por partido.
        """
        params = {"limit": 100}
        league_id = LEAGUE_IDS.get(competition_code) if competition_code else None
        if league_id:
            params["leagueId"] = league_id
        data = self._get(f"/fixtures/date/{date_iso[:10]}", params)
        return data if isinstance(data, list) else []

    def find_fixture_id(self, home_team: str, away_team: str, date_iso: str,
                         competition_code: str | None = None,
                         fixtures: list[dict] | None = None) -> str | None:
        """Encuentra el partido equivalente en Goal API.

        Ancla primero en la HORA DE INICIO, no en el nombre: ambas fuentes dan el
        mismo instante en UTC (verificado), mientras que los nombres pueden ser
        irreconocibles entre sí ("Stade Rennais FC 1901" contra "Rennes"). Con la
        hora ya filtrada quedan uno o dos candidatos, y ahí el nombre solo tiene
        que desempatar, así que se puede exigir menos parecido sin arriesgar un
        emparejamiento equivocado.

        Si ninguna hora coincide (datos corridos, partido aplazado), cae al
        cotejo por nombre sobre todos los partidos del día, pero exigiendo un
        parecido alto.

        `fixtures`: pasa la lista ya descargada para no gastar otra petición
        cuando emparejas varios partidos de la misma liga y fecha.
        """
        if fixtures is None:
            fixtures = self.fixtures_by_date(date_iso, competition_code)
        if not fixtures:
            return None

        def score(f) -> float:
            return (team_name_similarity(f.get("homeTeamName", ""), home_team)
                    + team_name_similarity(f.get("awayTeamName", ""), away_team))

        misma_hora = [f for f in fixtures if _same_kickoff(f.get("kickoffUtc"), date_iso)]
        if misma_hora:
            mejor = max(misma_hora, key=score)
            if score(mejor) >= 1.0:
                return mejor.get("id")
            # Coincidía la hora pero ningún nombre pega: puede ser un partido
            # reprogramado o una hora corregida en una sola de las dos fuentes.
            # No nos rendimos: se reintenta por nombre sobre todo el día.

        mejor = max(fixtures, key=score)
        return mejor.get("id") if score(mejor) >= 1.3 else None

    def match_statistics(self, fixture_id: str, home_team: str, away_team: str,
                          period: str = "fullTime") -> dict | None:
        """Corners, faltas y tarjetas por equipo. `period` puede ser 'fullTime' o
        'firstHalf' (el API expone 'secondHalf' pero llega siempre vacío).

        Devuelve None si el partido todavía no tiene estadísticas — el propio API
        lo dice en el flag `hasStatistics`, así no confundimos "sin datos" con
        "todo en cero".
        """
        data = self._get(f"/fixtures/{fixture_id}/statistics")
        if not isinstance(data, dict) or not data.get("hasStatistics"):
            return None
        rows = (data.get("match") or {}).get(period) or []
        if not rows:
            return None
        return _build_team_stats(rows, home_team, away_team)


def get_match_stats(home_team: str, away_team: str, date_iso: str,
                     competition_code: str | None = None, api_key: str = "",
                     fixtures: list[dict] | None = None) -> dict | None:
    """Punto de entrada simple, con la misma firma que el de Highlightly para que
    el dashboard pueda cambiar de proveedor sin tocar el resto del código.

    Nunca revienta el dashboard: devuelve None si falta la key, no se encuentra
    el partido o el API falla. La excepción es GoalApiRateLimited, que sí se
    propaga para poder avisar en la UI.

    `api_key`: pásala explícita cuando venga de st.secrets (Streamlit Cloud),
    porque os.getenv no ve los secrets de Streamlit.
    """
    key = api_key or GOAL_API_KEY
    if not key:
        return None
    client = GoalApiClient(api_key=key)
    try:
        fixture_id = client.find_fixture_id(
            home_team, away_team, date_iso, competition_code, fixtures=fixtures
        )
        if fixture_id is None:
            return None
        return client.match_statistics(fixture_id, home_team, away_team)
    except GoalApiRateLimited:
        raise
    except Exception:
        return None
