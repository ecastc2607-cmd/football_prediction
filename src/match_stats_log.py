"""
Acumula, para siempre, cada resultado real de corners/faltas/tarjetas que
consultamos en Highlightly — sin importar si fue desde la vista en vivo o desde
otro lado. La idea: no hay cuota para reconstruir el histórico de golpe (100
peticiones/día), pero si cada consulta real queda guardada, en unas semanas hay
suficiente muestra por equipo para promedios propios.

data/tracking/match_stats_log.csv — una fila por equipo por partido.
"""
from __future__ import annotations

import pandas as pd

from . import config

LOG_PATH = config.ROOT_DIR / "data" / "tracking" / "match_stats_log.csv"

COLUMNS = [
    "competition", "season", "matchday", "utc_date",
    "home_team", "away_team", "team", "is_home",
    "corners", "faltas", "tarjetas_amarillas", "tarjetas_rojas",
    "posesion", "remates_a_puerta", "remates_totales", "fueras_de_juego",
]

# Campos "bonus" que trae Goal API además de corners/faltas/tarjetas — se
# guardan también para que una consulta cacheada (get_cached_stats) se vea
# igual de completa que una fresca, y porque sirven igual para calibración.
EXTRA_FIELDS = ["posesion", "remates_a_puerta", "remates_totales", "fueras_de_juego"]
CORE_FIELDS = ["corners", "faltas", "tarjetas_amarillas", "tarjetas_rojas"]

MIN_SAMPLE_FOR_CONFIDENCE = 5  # partidos mínimos por equipo antes de confiar en el promedio


def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(LOG_PATH)


def match_key(competition: str, season, home_team: str, away_team: str) -> tuple:
    """Identidad de un partido dentro del log. `season` se normaliza a texto
    porque al releer el CSV puede volver como int o como str."""
    return (str(competition), str(season), str(home_team), str(away_team))


def existing_keys(log: pd.DataFrame | None = None) -> set[tuple]:
    """Partidos ya guardados. Sirve para saltárselos sin releer el CSV en cada
    iteración (el backfill recorre miles)."""
    log = load_log() if log is None else log
    if log.empty:
        return set()
    return {
        match_key(r.competition, r.season, r.home_team, r.away_team)
        for r in log.itertuples()
    }


def build_rows(competition: str, season: int, matchday: int, utc_date: str,
                home_team: str, away_team: str, stats: dict) -> list[dict]:
    """Convierte el dict de stats en filas del log (una por equipo)."""
    rows = []
    for team_name, team_stats in stats.items():
        is_home = team_name == home_team or team_name in home_team or home_team in team_name
        fila = {
            "competition": competition, "season": season, "matchday": matchday,
            "utc_date": utc_date, "home_team": home_team, "away_team": away_team,
            "team": team_name, "is_home": is_home,
        }
        for campo in CORE_FIELDS + EXTRA_FIELDS:
            fila[campo] = team_stats.get(campo)
        rows.append(fila)
    return rows


def append_rows(rows: list[dict]) -> int:
    """Añade filas ya construidas al CSV de una sola escritura. Devuelve cuántas
    escribió. Pensado para el backfill: guardar partido por partido reescribiría
    el archivo entero miles de veces."""
    if not rows:
        return 0
    log = load_log()
    config.ROOT_DIR.joinpath("data", "tracking").mkdir(parents=True, exist_ok=True)
    nuevas = pd.DataFrame(rows)
    if log.empty:
        combinado = nuevas
    else:
        # No siempre las nuevas filas traen los campos "bonus" (Highlightly no
        # da remates_totales/fueras_de_juego, y un partido guardado antes de
        # agregar estos campos tampoco los tiene), así que esa columna puede
        # llegar completamente en NaN de un lado del concat. Fijar el dtype a
        # float64 en ambos lados evita que pandas tenga que "adivinar" el tipo
        # de una columna totalmente vacía (y avise de un cambio futuro por eso).
        columnas = sorted(set(log.columns) | set(nuevas.columns), key=COLUMNS.index)
        log = log.reindex(columns=columnas)
        nuevas = nuevas.reindex(columns=columnas)
        for campo in CORE_FIELDS + EXTRA_FIELDS:
            log[campo] = pd.to_numeric(log[campo], errors="coerce")
            nuevas[campo] = pd.to_numeric(nuevas[campo], errors="coerce")
        combinado = pd.concat([log, nuevas], ignore_index=True)
    combinado.to_csv(LOG_PATH, index=False, encoding="utf-8")
    return len(rows)


def log_match_stats(competition: str, season: int, matchday: int, utc_date: str,
                     home_team: str, away_team: str, stats: dict) -> None:
    """Guarda las stats de un partido (dict como lo devuelven goal_api_client o
    highlightly_client: {nombre_equipo: {corners, faltas, ...}}).
    No duplica si ese partido ya estaba guardado.
    """
    if not stats:
        return
    log = load_log()
    key_mask = (
        (log["competition"] == competition) & (log["season"] == season)
        & (log["home_team"] == home_team) & (log["away_team"] == away_team)
    ) if not log.empty else pd.Series(dtype=bool)
    if key_mask.any():
        return  # ya lo teníamos

    rows = build_rows(competition, season, matchday, utc_date, home_team, away_team, stats)
    append_rows(rows)


def get_cached_stats(competition: str, home_team: str, away_team: str, utc_date: str) -> dict | None:
    """Estadísticas ya guardadas para ESTE partido exacto, en el mismo formato
    que devuelven goal_api_client/highlightly_client ({equipo: {corners, ...}}),
    o None si no está en el log.

    Identifica el partido por competición + equipos + fecha/hora de inicio, NO
    por season/jornada: un mismo partido puede haber quedado guardado con
    season=0 si se vio primero desde "En vivo" (ahí no siempre se conoce la
    jornada de todas las competiciones a la vez), y esta función debe encontrarlo
    igual, sin importar con qué season se guardó.

    Pensado para PARTIDOS YA TERMINADOS ("Resultados ya jugados"): antes, cada
    vez que se abría esa sección se volvía a pedir el dato al API sin importar
    si ya se tenía guardado, lo que agotó la cuota diaria de Goal API en un
    solo día de uso normal. Con esto, un partido consultado una vez no vuelve
    a gastar cuota — y de paso esta misma tabla queda lista para comparar,
    más adelante, corners/faltas/tarjetas reales contra lo que predijo el
    modelo (calibración), igual que ya se hace con el resultado 1X2.
    """
    log = load_log()
    if log.empty:
        return None
    # Compara solo hasta el minuto: nuestra propia fuente siempre guarda
    # 'YYYY-MM-DDTHH:MM:SSZ', pero por si acaso llega con milisegundos u otro
    # sufijo no vale la pena que eso rompa el match.
    fecha = str(utc_date)[:16]
    mask = (
        (log["competition"] == competition)
        & (log["home_team"] == home_team) & (log["away_team"] == away_team)
        & (log["utc_date"].astype(str).str[:16] == fecha)
    )
    rows = log[mask]
    if rows.empty:
        return None
    return {
        r.team: {campo: getattr(r, campo, None) for campo in CORE_FIELDS + EXTRA_FIELDS}
        for r in rows.itertuples()
    }


def team_averages(competition: str | None = None) -> pd.DataFrame:
    """Promedio de corners/faltas/tarjetas por equipo, local y visitante por
    separado, con cuántos partidos hay detrás de cada promedio (para marcar
    baja confianza cuando la muestra es chica)."""
    log = load_log()
    if competition:
        log = log[log["competition"] == competition]
    if log.empty:
        return pd.DataFrame(columns=[
            "team", "partidos", "corners_prom", "faltas_prom",
            "tarjetas_amarillas_prom", "tarjetas_rojas_prom", "baja_confianza",
        ])
    grouped = log.groupby("team").agg(
        partidos=("team", "count"),
        corners_prom=("corners", "mean"),
        faltas_prom=("faltas", "mean"),
        tarjetas_amarillas_prom=("tarjetas_amarillas", "mean"),
        tarjetas_rojas_prom=("tarjetas_rojas", "mean"),
    ).reset_index()
    grouped["baja_confianza"] = grouped["partidos"] < MIN_SAMPLE_FOR_CONFIDENCE
    return grouped
