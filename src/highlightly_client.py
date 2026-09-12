"""
Cliente mínimo de Highlightly (https://highlightly.net) — la fuente para lo que
football-data.org no trae: corners, faltas y tarjetas por partido.

Plan gratis: 100 peticiones/día, sin tarjeta de crédito. Regístrate directo en
https://highlightly.net (no hace falta pasar por RapidAPI) y copia tu key desde
el dashboard.

Se usa bajo demanda, partido por partido — no para reconstruir histórico completo.
Requiere HIGHLIGHTLY_API_KEY en el .env.
"""
from __future__ import annotations

import os
from difflib import SequenceMatcher

import requests

BASE_URL = "https://soccer.highlightly.net"
HIGHLIGHTLY_API_KEY = os.getenv("HIGHLIGHTLY_API_KEY", "")

# football-data.org usa nombres oficiales completos ("Real Madrid CF",
# "FC Internazionale Milano"); Highlightly usa nombres cortos ("Real Madrid",
# "Inter") y un nombre de liga propio para filtrar — verificado a mano por liga.
LEAGUE_NAME_MAP = {
    "PL": "Premier League",
    "PD": "La Liga",
    "SA": "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
    "CL": "UEFA Champions League",
}


class HighlightlyClient:
    def __init__(self, api_key: str = HIGHLIGHTLY_API_KEY):
        if not api_key:
            raise RuntimeError(
                "Falta HIGHLIGHTLY_API_KEY. Regístrate gratis (sin tarjeta) en "
                "https://highlightly.net, copia tu key del dashboard, y agrégala al .env."
            )
        self.session = requests.Session()
        self.session.headers.update({"x-rapidapi-key": api_key})

    def _get(self, path: str, params: dict) -> dict | list:
        resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def find_match_id(self, home_team: str, away_team: str, date_iso: str,
                       league_name: str | None = None) -> int | None:
        """Busca el matchId de Highlightly por fecha (+ liga, si se conoce) y
        emparejamiento aproximado de nombres: football-data.org usa nombres
        oficiales completos ("Real Madrid CF") y Highlightly nombres cortos
        ("Real Madrid"), así que un match exacto de nombre casi nunca funciona.
        """
        date = date_iso[:10]  # YYYY-MM-DD
        params = {"date": date}
        if league_name:
            params["leagueName"] = league_name
        data = self._get("/matches", params)
        matches = data.get("data", data if isinstance(data, list) else [])
        if not matches:
            return None

        def score(m):
            h = m.get("homeTeam", {}).get("name", "")
            a = m.get("awayTeam", {}).get("name", "")
            return (
                SequenceMatcher(None, h.lower(), home_team.lower()).ratio()
                + SequenceMatcher(None, a.lower(), away_team.lower()).ratio()
            )

        best = max(matches, key=score)
        return best["id"] if score(best) > 1.1 else None

    def match_statistics(self, match_id: int) -> dict:
        """Corners, faltas, tarjetas por equipo."""
        data = self._get(f"/statistics/{match_id}", {})
        out = {}
        for team_stats in data if isinstance(data, list) else []:
            team_name = team_stats.get("team", {}).get("name", "?")
            # displayName real de Highlightly: minúsculas en "cards" (verificado con datos reales).
            stat_map = {s.get("displayName"): s.get("value") for s in team_stats.get("statistics", [])}
            out[team_name] = {
                "corners": stat_map.get("Corners"),
                "faltas": stat_map.get("Fouls"),
                "tarjetas_amarillas": stat_map.get("Yellow cards"),
                "tarjetas_rojas": stat_map.get("Red cards"),
                # datos extra que ya vienen en la misma llamada, por si se quieren mostrar después:
                "posesion": stat_map.get("Possession"),
                "remates_a_puerta": stat_map.get("Shots on target"),
                "xg": stat_map.get("Expected Goals"),
            }
        return out


def get_match_stats(home_team: str, away_team: str, date_iso: str,
                     competition_code: str | None = None) -> dict | None:
    """Punto de entrada simple: nunca revienta el dashboard — devuelve None si
    falta la key, no se encuentra el partido, o la API falla.

    `competition_code` es el código de football-data.org (PL, PD, SA, BL1, FL1,
    CL) — se traduce al nombre de liga de Highlightly para acotar la búsqueda.
    """
    if not HIGHLIGHTLY_API_KEY:
        return None
    try:
        client = HighlightlyClient()
        league_name = LEAGUE_NAME_MAP.get(competition_code) if competition_code else None
        match_id = client.find_match_id(home_team, away_team, date_iso, league_name)
        if match_id is None:
            return None
        return client.match_statistics(match_id)
    except Exception:
        return None
