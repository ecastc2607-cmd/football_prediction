"""
Eventos de partido (goles con minuto y jugador, penales, tarjetas) y cuotas de
mercados especiales de BWIN, de los partidos YA JUGADOS, desde 365scores.

Para qué (pedido del usuario, oct-2026): analizar mercados más allá de 1X2 /
goles / ambos anotan — primer gol, penal en el partido, goles de jugador,
hándicap asiático, resultado por tiempo, tiros al arco, fueras de juego... — y
mostrar en la vista "Avanzada" los que de verdad aguantan.

Se guarda en el repo (el usuario lo pidió así, para no volver a pedirlo):
  data/tracking/events_365/games.csv        un renglón por partido descargado
  data/tracking/events_365/events.csv       un renglón por evento (sin cambios)
  data/tracking/events_365/special_odds.csv una opción de cuota por renglón

Ojo con las cuotas: 365scores borra los mercados especiales ~10 días después
del partido (quedan solo 1X2 y goles). Por eso la captura de GitHub Actions
llama a `record_finished` en cada corrida: si se pierde una semana, esas
cuotas ya no se recuperan (los eventos sí).

Uso (descarga inicial, ambas temporadas, todas las competiciones):
    python -m src.events365 --desde 2025-07-01
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config, scores365

DIR = config.ROOT_DIR / "data" / "tracking" / "events_365"
GAMES_PATH = DIR / "games.csv"
EVENTS_PATH = DIR / "events.csv"
ODDS_PATH = DIR / "special_odds.csv"

GAME_COLUMNS = ["game_id", "competition", "start_utc", "home_team", "away_team", "home_goals", "away_goals",
                "ht_home_goals", "ht_away_goals", "has_odds", "fetched_at"]
EVENT_COLUMNS = ["game_id", "side", "minute", "added_time", "half", "type", "subtype", "player"]
ODDS_COLUMNS = ["game_id", "line_type_id", "market", "line", "option", "odds", "opening_odds", "won"]
SKIP_EVENTS = {"Substitution"}  # no sirven para ningún mercado y son la mitad de los eventos
STAGE_HALF = {7: 1, 9: 2}  # stageId de 365: 7 = 1er tiempo, 9 = 2do (verificado oct-2026)
ODDS_KEEP_DAYS = 9  # después 365 ya no tiene los mercados especiales


def _read(path, columns) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


def load_games() -> pd.DataFrame:
    return _read(GAMES_PATH, GAME_COLUMNS)


def load_events() -> pd.DataFrame:
    return _read(EVENTS_PATH, EVENT_COLUMNS)


def load_odds() -> pd.DataFrame:
    return _read(ODDS_PATH, ODDS_COLUMNS)


def _game_detail(game_id: int) -> tuple[dict | None, list[dict], list[dict]]:
    """(partido, eventos, cuotas) de un partido terminado."""
    data = scores365._get("/game/", {"gameId": game_id}) or {}
    g = data.get("game")
    if not g:
        return None, [], []
    local, visita = g.get("homeCompetitor") or {}, g.get("awayCompetitor") or {}
    lado = {local.get("id"): "home", visita.get("id"): "away"}
    jugadores = {}
    for m in g.get("members") or []:
        jugadores[m.get("id")] = m.get("name")
    ht = next((s for s in g.get("stages") or [] if s.get("id") == 7), {})
    partido = {
        "game_id": game_id, "start_utc": scores365._utc(g.get("startTime")),
        "home_team": local.get("name"), "away_team": visita.get("name"),
        "home_goals": scores365._score(local), "away_goals": scores365._score(visita),
        "ht_home_goals": ht.get("homeCompetitorScore"), "ht_away_goals": ht.get("awayCompetitorScore"),
    }
    eventos = []
    for e in g.get("events") or []:
        tipo = (e.get("eventType") or {})
        if tipo.get("name") in SKIP_EVENTS:
            continue
        eventos.append({
            "game_id": game_id, "side": lado.get(e.get("competitorId")),
            "minute": e.get("gameTime"), "added_time": e.get("addedTime"),
            "half": STAGE_HALF.get(e.get("stageId")), "type": tipo.get("name"),
            "subtype": tipo.get("subTypeName"), "player": jugadores.get(e.get("playerId")),
        })
    lineas = (scores365._get("/bets/lines/", {"games": game_id}) or {}).get("lines", [])
    cuotas = line_rows(game_id, lineas)
    # Solo 1X2 y goles = ya se borraron los especiales: no se guardan (ya están en odds_snapshots).
    especiales = {c["line_type_id"] for c in cuotas} - {scores365.LINE_1X2, scores365.LINE_GOALS}
    if not especiales:
        cuotas = []
    partido["has_odds"] = bool(cuotas)
    return partido, eventos, cuotas


def line_rows(game_id: int, lineas: list[dict]) -> list[dict]:
    """Líneas de BWIN de 365scores -> renglones con las columnas de special_odds."""
    filas = []
    for ln in lineas:
        tipo = ln.get("lineType") or {}
        for o in ln.get("options", []):
            filas.append({
                "game_id": game_id, "line_type_id": ln.get("lineTypeId"), "market": tipo.get("name"),
                "line": ln.get("internalOptionValue"), "option": o.get("name"),
                "odds": scores365._decimal(o), "opening_odds": scores365._decimal(o, "originalRate"),
                "won": bool(o.get("isWon")),
            })
    return filas


def _finished_ids(code: str, desde: datetime, hasta: datetime) -> list[tuple[int, str]]:
    """(id, competición) de los partidos terminados, en ventanas de ≤ 1 mes."""
    salida = []
    inicio = desde
    while inicio < hasta:
        fin = min(inicio + timedelta(days=28), hasta)
        # Sin showOdds/topBookmaker (los usa scores365.games): con esos filtros 365
        # devolvía solo parte de los partidos (Colombia: 199 de ~800).
        data = scores365._get("/games/", {"competitions": scores365.COMPETITION_IDS[code],
                                          "startDate": inicio.strftime("%d/%m/%Y"),
                                          "endDate": (fin - timedelta(seconds=1)).strftime("%d/%m/%Y")}) or {}
        for g in data.get("games", []):
            if g.get("statusGroup") == scores365.STATUS_FINISHED:
                salida.append((int(g["id"]), code))
        inicio = fin
    return salida


def _append(path, columns, rows: list[dict]) -> None:
    if not rows:
        return
    DIR.mkdir(parents=True, exist_ok=True)
    nuevo = pd.DataFrame(rows, columns=columns)
    nuevo.to_csv(path, mode="a", header=not path.exists(), index=False)


def fetch(pendientes: list[tuple[int, str]], workers: int = 4) -> int:
    """Descarga y guarda los partidos pendientes. Devuelve cuántos guardó."""
    ahora = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def uno(item):
        gid, code = item
        try:
            return code, _game_detail(gid)
        except Exception:
            return code, (None, [], [])

    guardados = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        lote_p, lote_e, lote_o = [], [], []
        for code, (partido, eventos, cuotas) in pool.map(uno, pendientes):
            if not partido or partido["home_goals"] is None:
                continue
            lote_p.append({**partido, "competition": code, "fetched_at": ahora})
            lote_e += eventos
            lote_o += cuotas
            guardados += 1
            if len(lote_p) >= 100:  # se guarda por tandas: si se corta, no se pierde todo
                _append(GAMES_PATH, GAME_COLUMNS, lote_p); _append(EVENTS_PATH, EVENT_COLUMNS, lote_e)
                _append(ODDS_PATH, ODDS_COLUMNS, lote_o)
                print(f"  {guardados} partidos guardados", flush=True)
                lote_p, lote_e, lote_o = [], [], []
        _append(GAMES_PATH, GAME_COLUMNS, lote_p); _append(EVENTS_PATH, EVENT_COLUMNS, lote_e)
        _append(ODDS_PATH, ODDS_COLUMNS, lote_o)
    return guardados


def record_finished(games: list[dict]) -> int:
    """Para la captura automática: guarda los partidos terminados de `games`
    (la lista de 365 que la captura ya tiene) que todavía no estén."""
    ids = {v: k for k, v in scores365.COMPETITION_IDS.items()}
    ya = set(load_games()["game_id"].astype(int))
    pendientes = [(int(g["id"]), ids[g["competitionId"]]) for g in games
                  if g.get("statusGroup") == scores365.STATUS_FINISHED and g.get("competitionId") in ids
                  and int(g["id"]) not in ya]
    return fetch(pendientes, workers=2) if pendientes else 0


def backfill(desde: datetime, hasta: datetime | None = None, codes: list[str] | None = None) -> int:
    hasta = hasta or datetime.now(timezone.utc)
    ya = set(load_games()["game_id"].astype(int))
    pendientes = []
    for code in codes or list(scores365.COMPETITION_IDS):
        ids = [x for x in _finished_ids(code, desde, hasta) if x[0] not in ya]
        print(f"{code}: {len(ids)} partidos por bajar", flush=True)
        pendientes += ids
    # Primero los más recientes: son los únicos que todavía tienen cuotas especiales.
    pendientes.sort(key=lambda x: -x[0])
    return fetch(pendientes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="2025-07-01")
    ap.add_argument("--comp", nargs="*")
    a = ap.parse_args()
    desde = datetime.strptime(a.desde, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    n = backfill(desde, codes=a.comp)
    print(f"Listo: {n} partidos nuevos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
