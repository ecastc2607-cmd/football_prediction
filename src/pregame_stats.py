"""
Estadísticas previas de 365scores por partido (pedido del usuario, nov-2026):
lo mismo que muestra 365 en la ficha del partido — "Últimos 15 partidos" de
cada equipo en todas las competiciones: partidos ganados, ambos marcaron, más
de 2.5 goles, marcó primero, valla invicta... y promedios (goles, xG, remates,
corners, tarjetas).

Fuente: webws.365scores.com/web/stats/preGame/?game=<id> (sin clave).

Se guardan en data/tracking/pregame_stats.csv (decisión del usuario): la
captura automática (capture_live_odds) toma UNA foto por partido en la
ventana previa (las cifras cambian con cada fecha que juega el equipo). La
Jornada lee el archivo y, si el partido todavía no está, consulta en vivo.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import config, scores365

PATH = config.ROOT_DIR / "data" / "tracking" / "pregame_stats.csv"
COLUMNS = ["fetched_at", "competition", "game_id", "utc_date", "home_team", "away_team", "side",
           "stat_id", "group", "name", "value", "games_text"]
LANG_ES = 29  # nombres de las estadísticas en español, como en la web de 365

# Lo que va en la columna compacta de la Jornada: (stat_id, etiqueta corta).
COMPACT = [(141, "Gan"), (142, "Ambos"), (143, "+2.5"), (150, "Invicta"), (171, "Corners"), (174, "Tarj")]


def fetch(game_id: int, competition: str = "", utc_date: str = "", home: str = "", away: str = "") -> pd.DataFrame:
    """Estadísticas previas de un partido de 365 (vacío si no hay)."""
    data = scores365._get("/stats/preGame/", {"game": int(game_id), "langId": LANG_ES})
    if not data or not data.get("statistics"):
        return pd.DataFrame(columns=COLUMNS)
    juego = next((g for g in data.get("games") or [] if g.get("id") == int(game_id)), {})
    home_id = (juego.get("homeCompetitor") or {}).get("id")
    away_id = (juego.get("awayCompetitor") or {}).get("id")
    if home_id is None or away_id is None:  # sin el partido en la respuesta: el primero que aparece es el local
        ids = list(dict.fromkeys(s.get("competitorId") for s in data["statistics"]))
        home_id, away_id = (ids + [None, None])[:2]
    textos = data.get("statisticGamesPlayed") or {}
    ahora = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    filas = []
    for s in data["statistics"]:
        lado = "home" if s.get("competitorId") == home_id else "away" if s.get("competitorId") == away_id else None
        if lado is None:
            continue
        filas.append({
            "fetched_at": ahora, "competition": competition, "game_id": int(game_id),
            "utc_date": utc_date or scores365._utc(juego.get("startTime")) or "",
            "home_team": home or (juego.get("homeCompetitor") or {}).get("name", ""),
            "away_team": away or (juego.get("awayCompetitor") or {}).get("name", ""),
            "side": lado, "stat_id": s.get("id"), "group": s.get("statisticGroup"),
            "name": s.get("name", ""), "value": str(s.get("value", "")),
            "games_text": textos.get("homeText" if lado == "home" else "awayText", ""),
        })
    return pd.DataFrame(filas, columns=COLUMNS)


def load() -> pd.DataFrame:
    if not PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(PATH, dtype={"value": str})


def record(games: list[dict], code_by_competition_id: dict[int, str]) -> int:
    """Guarda la foto de los partidos de `games` (de /games/ de 365) que aún no
    estén en el archivo. Devuelve cuántos partidos guardó."""
    hist = load()
    ya = set(pd.to_numeric(hist["game_id"], errors="coerce").dropna().astype(int)) if not hist.empty else set()
    nuevas = []
    for g in games:
        gid = g.get("id")
        if gid is None or int(gid) in ya:
            continue
        df = fetch(int(gid), code_by_competition_id.get(g.get("competitionId"), ""),
                   scores365._utc(g.get("startTime")) or "",
                   (g.get("homeCompetitor") or {}).get("name", ""), (g.get("awayCompetitor") or {}).get("name", ""))
        if not df.empty:
            nuevas.append(df)
    if not nuevas:
        return 0
    todo = pd.concat([hist, *nuevas], ignore_index=True) if not hist.empty else pd.concat(nuevas, ignore_index=True)
    PATH.parent.mkdir(parents=True, exist_ok=True)
    todo.to_csv(PATH, index=False)
    return len(nuevas)


def for_match(stats: pd.DataFrame, game_id: int | None) -> pd.DataFrame:
    """Filas del partido (la última foto) del archivo o de una consulta ya hecha."""
    if game_id is None or stats.empty:
        return pd.DataFrame(columns=COLUMNS)
    d = stats[pd.to_numeric(stats["game_id"], errors="coerce") == int(game_id)]
    if d.empty:
        return d
    return d[d["fetched_at"] == d["fetched_at"].max()]


def compact(d: pd.DataFrame) -> str:
    """'Gan 60%/48% · Ambos 47%/24% · +2.5 47%/29% · Corners 7.2/5.2 ...' (local/visitante)."""
    if d.empty:
        return "Sin datos"
    partes = []
    for sid, etiqueta in COMPACT:
        v = d[pd.to_numeric(d["stat_id"], errors="coerce") == sid].set_index("side")["value"]
        if "home" not in v or "away" not in v:
            continue

        def corto(x):  # "9 (60%)" -> "60%"
            x = str(x)
            return x[x.find("(") + 1:x.find(")")] if "(" in x else x
        partes.append(f"{etiqueta} {corto(v['home'])}/{corto(v['away'])}")
    return " · ".join(partes) if partes else "Sin datos"


def detail_table(d: pd.DataFrame, home_label: str, away_label: str) -> pd.DataFrame:
    """Tabla local | estadística | visitante, en el orden de 365."""
    if d.empty:
        return pd.DataFrame()
    orden = d.drop_duplicates("stat_id")[["stat_id", "group", "name"]]
    local = d[d["side"] == "home"].set_index("stat_id")["value"]
    visita = d[d["side"] == "away"].set_index("stat_id")["value"]
    def nombre(sid, g, n):  # el grupo 2 son promedios por partido, salvo penales (total convertidos/ejecutados)
        return f"Promedio: {n}" if g == 2 and sid != 175 else n
    return pd.DataFrame({
        home_label: [local.get(s, "") for s in orden["stat_id"]],
        "Estadística": [nombre(int(s), int(g) if pd.notna(g) else 0, n)
                        for s, g, n in zip(orden["stat_id"], orden["group"], orden["name"])],
        away_label: [visita.get(s, "") for s in orden["stat_id"]],
    })
