"""
Cuotas de 365scores (https://www.365scores.com) — prepartido y en vivo.

No es una API pública documentada: es el servicio que usa su propia web
(webws.365scores.com), que responde sin clave. Con país Colombia
(userCountryId=109) trae las cuotas de BWIN, la casa que 365scores muestra en
Colombia, en ~19 mercados por partido (1X2, doble oportunidad, goles, ambos
anotan, 1er/2do tiempo, corners, tarjetas...), con la cuota actual, la
anterior y la de APERTURA. Verificado oct-2026.

Al ser un servicio interno puede cambiar o bloquearse sin aviso, así que:
  - nunca lanza hacia el dashboard (devuelve [] ante cualquier problema);
  - odds_api.get_odds_table la usa primero y, si no trae nada, cae a The Odds
    API como antes.

Las filas salen en el MISMO formato que odds_api._event_row (co_home,
co_over, ...), así que combinadas, Tips y el ancla de mercado en vivo las
usan sin cambios.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

BASE_URL = "https://webws.365scores.com/web"
BOOKMAKER = "BWIN (365scores)"
SOURCE = "365scores"
# langId=1 (inglés): los nombres en inglés se parecen mucho más a los de
# football-data.org/Goal API que los traducidos ("Friburgo", "Estrella Roja").
_PARAMS = {"appTypeId": 5, "langId": 1, "timezoneName": "UTC", "userCountryId": 109}
_HEADERS = {
    "User-Agent": "Mozilla/5.0", "Accept": "application/json",
    "Origin": "https://www.365scores.com", "Referer": "https://www.365scores.com/",
}

# IDs de competición en 365scores, verificados contra su /search/ (oct-2026).
COMPETITION_IDS = {
    "PL": 7, "PD": 11, "SA": 17, "BL1": 25, "FL1": 35,
    "CL": 572, "EL": 573, "ECL": 7685, "NT": 7016,
}

# Tipos de línea de /bets/lines/ y el orden ("num") de sus opciones.
LINE_1X2, LINE_GOALS, LINE_BTTS, LINE_DOUBLE_CHANCE = 1, 3, 12, 14
TOTALS_LINE = "2.5"
# statusGroup de 365scores: 2 = por jugar, 3 = en juego, 4 = terminado.
STATUS_SCHEDULED, STATUS_LIVE, STATUS_FINISHED = 2, 3, 4


def _get(path: str, params: dict) -> dict | None:
    try:
        resp = requests.get(f"{BASE_URL}{path}", params={**_PARAMS, **params}, headers=_HEADERS, timeout=20)
        if not resp.ok:
            return None
        return resp.json()
    except Exception:
        return None


def _utc(start_time: str) -> str:
    """'2026-10-15T16:45:00+00:00' -> '2026-10-15T16:45:00Z'."""
    try:
        return datetime.fromisoformat(start_time).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return ""


def _decimal(option: dict, key: str = "rate") -> float | None:
    try:
        valor = float((option.get(key) or {}).get("decimal"))
    except (TypeError, ValueError):
        return None
    return valor if valor > 1 else None


def _score(competitor: dict | None) -> int | None:
    """Goles del equipo; 365scores manda -1 si el partido no empezó."""
    try:
        valor = int((competitor or {}).get("score"))
    except (TypeError, ValueError):
        return None
    return valor if valor >= 0 else None


def games(code: str, date_from_utc: str, date_to_utc: str) -> list[dict]:
    """Partidos de la competición cuyo inicio cae en [desde, hasta) UTC."""
    comp = COMPETITION_IDS.get(code)
    if comp is None:
        return []
    desde = datetime.strptime(date_from_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    hasta = datetime.strptime(date_to_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    data = _get("/games/", {
        "competitions": comp, "showOdds": "true", "topBookmaker": 4,
        "startDate": desde.strftime("%d/%m/%Y"),
        "endDate": (hasta - timedelta(seconds=1)).strftime("%d/%m/%Y"),
    })
    salida = []
    for g in (data or {}).get("games", []):
        inicio = _utc(g.get("startTime"))
        if inicio and date_from_utc <= inicio < date_to_utc:
            salida.append(g)
    return salida


def _prices_from_lines(lines: list[dict]) -> dict[str, tuple[float | None, float | None]]:
    """{clave: (cuota_actual, cuota_de_apertura)} con las claves de odds_api.PRICE_KEYS."""
    precios: dict[str, tuple[float | None, float | None]] = {}

    def anotar(clave, opcion):
        if opcion:
            precios[clave] = (_decimal(opcion), _decimal(opcion, "originalRate"))

    for line in lines:
        tipo = line.get("lineTypeId") or (line.get("lineType") or {}).get("id")
        por_num = {o.get("num"): o for o in line.get("options", [])}
        if tipo == LINE_1X2:
            anotar("home", por_num.get(1)); anotar("draw", por_num.get(2)); anotar("away", por_num.get(3))
        elif tipo == LINE_GOALS and str(line.get("internalOptionValue")) == TOTALS_LINE:
            anotar("under", por_num.get(1)); anotar("over", por_num.get(2))
        elif tipo == LINE_BTTS:
            anotar("btts_yes", por_num.get(1)); anotar("btts_no", por_num.get(2))
        elif tipo == LINE_DOUBLE_CHANCE:
            anotar("dc_1x", por_num.get(1)); anotar("dc_x2", por_num.get(3))
    return precios


def game_lines(game_id: int) -> list[dict]:
    """Todas las líneas de BWIN de UN partido (el endpoint no acepta varios)."""
    data = _get("/bets/lines/", {"games": game_id})
    return [l for l in (data or {}).get("lines", []) if l.get("gameId") == game_id]


def event_rows(code: str, date_from_utc: str, date_to_utc: str, meta: dict,
               price_keys: list[str]) -> list[dict]:
    """Filas en el formato de odds_api._event_row, más la cuota de apertura
    (open_<clave>), el estado y el minuto (para las fotos en vivo)."""
    filas = []
    for g in games(code, date_from_utc, date_to_utc):
        precios = _prices_from_lines(game_lines(g["id"]))
        if not precios and g.get("odds"):  # sin líneas: al menos el 1X2 principal
            precios = _prices_from_lines([g["odds"]])
        if not precios:
            continue
        fila = dict(
            meta, source=SOURCE, event_id=f"365:{g['id']}", commence_time=_utc(g.get("startTime")),
            home_team=(g.get("homeCompetitor") or {}).get("name"),
            away_team=(g.get("awayCompetitor") or {}).get("name"),
            status_group=g.get("statusGroup"), game_minute=g.get("gameTime"),
            home_score=_score(g.get("homeCompetitor")),
            away_score=_score(g.get("awayCompetitor")),
        )
        for clave in price_keys:
            actual, apertura = precios.get(clave, (None, None))
            for filtro in ("co", "all"):  # BWIN opera en Colombia: misma cuota en ambos
                fila[f"{filtro}_{clave}"] = actual
                fila[f"{filtro}_{clave}_book"] = BOOKMAKER if actual else ""
            fila[f"open_{clave}"] = apertura
        filas.append(fila)
    return filas


# --- En vivo ---

def live_rows(codes: list[str], price_keys: list[str]) -> list[dict]:
    """Partidos EN JUEGO ahora de esas competiciones, con su cuota en vivo
    (mismo formato que event_rows). UNA consulta para todas las competiciones
    + una por partido en juego."""
    ids = {COMPETITION_IDS[c]: c for c in codes if c in COMPETITION_IDS}
    if not ids:
        return []
    hoy = datetime.now(timezone.utc)
    data = _get("/games/", {
        "competitions": ",".join(str(i) for i in ids), "showOdds": "true", "topBookmaker": 4,
        # desde ayer: un partido que empezó antes de medianoche UTC sigue en juego
        "startDate": (hoy - timedelta(days=1)).strftime("%d/%m/%Y"), "endDate": hoy.strftime("%d/%m/%Y"),
    })
    filas = []
    ahora = hoy.strftime("%Y-%m-%dT%H:%M:%SZ")
    for g in (data or {}).get("games", []):
        if g.get("statusGroup") != STATUS_LIVE:
            continue
        code = ids.get(g.get("competitionId"))
        meta = {"fetched_at": ahora, "competition": code}
        precios = _prices_from_lines(game_lines(g["id"])) or _prices_from_lines([g["odds"]] if g.get("odds") else [])
        fila = dict(
            meta, source=SOURCE, event_id=f"365:{g['id']}", commence_time=_utc(g.get("startTime")),
            home_team=(g.get("homeCompetitor") or {}).get("name"),
            away_team=(g.get("awayCompetitor") or {}).get("name"),
            status_group=g.get("statusGroup"), game_minute=g.get("gameTime"),
            home_score=_score(g.get("homeCompetitor")), away_score=_score(g.get("awayCompetitor")),
        )
        for clave in price_keys:
            actual, apertura = precios.get(clave, (None, None))
            fila[f"co_{clave}"] = fila[f"all_{clave}"] = actual
            fila[f"co_{clave}_book"] = fila[f"all_{clave}_book"] = BOOKMAKER if actual else ""
            fila[f"open_{clave}"] = apertura
        filas.append(fila)
    return filas
