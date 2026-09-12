"""
Contexto de partido que no depende del modelo de goles: posición en la liga
(football-data.org, gratis e ilimitado dentro del plan normal) y rivalidad/
derbi/clásico (lista curada a mano — no hay API confiable para esto).

La localía ya está incorporada al modelo de Poisson (fuerza de ataque/defensa
separada por local/visitante), así que no se repite acá como factor aparte.
"""
from __future__ import annotations

from .fetch_football_data import FootballDataClient

# Rivalidades históricas conocidas, por nombre corto (el "shortName" real que da
# la API — verificado contra los datos descargados, no adivinado: por ejemplo es
# "Barça" y no "Barcelona", "Atleti" y no "Atlético Madrid", "Dortmund" y no
# "Borussia Dortmund"). Lista curada a mano, no exhaustiva — se amplía con el tiempo.
RIVALRIES: list[tuple[str, str, str]] = [
    ("Real Madrid", "Barça", "🔥 El Clásico"),
    ("Real Madrid", "Atleti", "🔥 Derbi madrileño"),
    ("Barça", "Espanyol", "🔥 Derbi barcelonés"),
    ("Sevilla FC", "Real Betis", "🔥 Derbi sevillano"),
    ("Arsenal", "Tottenham", "🔥 Derbi del norte de Londres"),
    ("Chelsea", "Arsenal", "🔥 Derbi londinense"),
    ("Liverpool", "Everton", "🔥 Derbi de Merseyside"),
    ("Liverpool", "Man United", "🔥 Rivalidad histórica"),
    ("Man United", "Man City", "🔥 Derbi de Mánchester"),
    ("Man United", "Leeds United", "🔥 Rivalidad histórica"),
    ("Inter", "Milan", "🔥 Derbi della Madonnina"),
    ("Roma", "Lazio", "🔥 Derbi della Capitale"),
    ("Juventus", "Torino", "🔥 Derbi della Mole"),
    ("Juventus", "Inter", "🔥 Derbi d'Italia"),
    ("Bayern", "Dortmund", "🔥 Der Klassiker"),
    ("Dortmund", "Schalke", "🔥 Revierderby"),
    ("PSG", "Marseille", "🔥 Le Classique"),
    ("Olympique Lyon", "Saint-Étienne", "🔥 Derbi del Ródano"),
    ("Porto", "Benfica", "🔥 O Clássico"),
    ("Benfica", "Sporting CP", "🔥 Derbi de Lisboa"),
]


def rivalry_label(home_short: str, away_short: str) -> str | None:
    """Devuelve la etiqueta de rivalidad si el cruce coincide con la lista
    curada (en cualquier orden de local/visitante), o None si no aplica."""
    for a, b, label in RIVALRIES:
        if {a, b} <= {home_short, away_short}:
            return label
    return None


def get_standings_map(client: FootballDataClient, competition_code: str, season: int) -> dict[str, int]:
    """{nombre_equipo: posición en la tabla}. Devuelve {} si la competición no
    tiene tabla de posiciones (ej. Champions League en fase de grupos/liga usa
    un formato distinto que football-data.org no expone igual)."""
    try:
        data = client.get_standings(competition_code, season)
    except Exception:
        return {}
    positions = {}
    for table in data.get("standings", []):
        if table.get("type") != "TOTAL":
            continue
        for row in table.get("table", []):
            positions[row["team"]["name"]] = row["position"]
    return positions
