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


# --- Alarma de Tips: el pick del modelo contra lo que dicen las cifras de 365 ---
# Pedido del usuario (nov-2026), caso Kairat–Mjällby: el modelo daba de favorito
# al visitante (27/25/48) y en 365 Kairat gana el 54% de sus partidos y Mjällby
# el 8%. No cambia el estado del pick: es un aviso para revisarlo antes de apostar.
# Umbrales (pts de diferencia en % de partidos ganados del equipo del pick
# contra su rival): ganar / empate no válido / hándicap negativo 20, doble
# oportunidad 30, hándicap positivo 40 (ahí el pick no dice que gane).
ALARM_WIN_GAP = {"gana": 0.20, "doble": 0.30, "positivo": 0.40}
# Goles: promedio de "más de 2.5" / "ambos marcaron" de los dos equipos en 365.
ALARM_GOALS = {("Más", 1.5): ("<", 0.25), ("Más", 2.5): ("<", 0.40), ("Menos", 2.5): (">", 0.60),
               ("Menos", 3.5): (">", 0.70)}
ALARM_BTTS = {"Sí": ("<", 0.35), "No": (">", 0.65)}
# Estadísticas: suma de los promedios por partido de cada equipo en 365.
ALARM_STAT_ID = {"Corners": 171, "Tarjetas": 174, "Tiros al arco": 168, "Remates totales": 165}


def _numero(valor) -> float | None:
    """'9 (60%)' -> 0.60 · '17/26 (65%)' -> 0.65 · '5.28' -> 5.28."""
    import re

    s = str(valor)
    m = re.search(r"\(([0-9.]+)%\)", s)
    if m:
        return float(m.group(1)) / 100
    try:
        return float(s)
    except ValueError:
        return None


def _valores(d: pd.DataFrame) -> dict:
    v = {"home": {}, "away": {}}
    for r in d.itertuples():
        if r.side in v:
            n = _numero(r.value)
            if n is not None:
                v[r.side][int(r.stat_id)] = n
    return v


def alarm(d: pd.DataFrame, mercado: str, pick: str, home: str, away: str,
          home_label: str = "", away_label: str = "") -> str:
    """Texto de alarma si las cifras de 365 contradicen el pick; '' si no."""
    import re

    if d is None or d.empty:
        return ""
    v = _valores(d)
    nombres = {"home": home_label or home, "away": away_label or away}
    pick = str(pick)

    def lado_de(equipo: str) -> str | None:
        e = equipo.strip().lower()
        for lado, nombre in (("home", home), ("away", away), ("home", home_label), ("away", away_label)):
            if nombre and e == str(nombre).strip().lower():
                return lado
        return None

    def por_ganar(lado: str | None, tipo: str) -> str:
        if lado is None:
            return ""
        otro = "away" if lado == "home" else "home"
        g, r = v[lado].get(141), v[otro].get(141)
        if g is None or r is None or r - g < ALARM_WIN_GAP[tipo]:
            return ""
        return (f"⚠ 365: {nombres[lado]} gana {g:.0%} de sus partidos y {nombres[otro]} {r:.0%}, "
                f"pero el pick va con {nombres[lado]}")

    if mercado == "Resultado" and pick in ("Local", "Visitante"):
        return por_ganar("home" if pick == "Local" else "away", "gana")
    if mercado == "Doble oportunidad":
        return por_ganar("home" if pick.startswith("1X") else "away", "doble")
    if mercado in ("Empate no válido", "Primer gol"):
        if mercado == "Primer gol":
            lado = lado_de(pick)
            if lado is None:
                return ""
            otro = "away" if lado == "home" else "home"
            p, r = v[lado].get(144), v[otro].get(144)
            if p is not None and r is not None and r - p >= ALARM_WIN_GAP["gana"]:
                return f"⚠ 365: {nombres[lado]} marca primero {p:.0%} y {nombres[otro]} {r:.0%}"
            return ""
        return por_ganar(lado_de(pick), "gana")
    h = re.match(r"^(.+) ([+-][0-9]+(?:\.[0-9]+)?)$", pick)
    if mercado == "Hándicap asiático" and h:
        return por_ganar(lado_de(h.group(1)), "positivo" if float(h.group(2)) > 0 else "gana")

    if mercado == "Más tarjetas":  # el equipo del pick promedia menos tarjetas que el rival en 365
        lado = lado_de(pick.split(" (más tarjetas")[0])
        if lado is None:
            return ""
        otro = "away" if lado == "home" else "home"
        a, b = v[lado].get(174), v[otro].get(174)
        if a is not None and b is not None and b - a >= 0.3:
            return f"⚠ 365: {nombres[lado]} promedia {a:.2f} tarjetas por partido y {nombres[otro]} {b:.2f}"
        return ""
    m = re.match(r"^\s*(?:(.+?): )?(Más|Menos) de ([0-9]+(?:\.[0-9]+)?)", pick)
    if mercado in ("Goles", "Goles (otra línea)") and m:
        regla = ALARM_GOALS.get((m.group(2), float(m.group(3))))
        a, b = v["home"].get(143), v["away"].get(143)
        if regla is None or a is None or b is None:
            return ""
        prom = (a + b) / 2
        if (regla[0] == "<" and prom < regla[1]) or (regla[0] == ">" and prom > regla[1]):
            return (f"⚠ 365: más de 2.5 goles en {a:.0%} de los partidos de {nombres['home']} y {b:.0%} "
                    f"de los de {nombres['away']}")
        return ""
    if mercado == "Ambos anotan":
        regla = ALARM_BTTS.get("Sí" if pick.lower().startswith("s") else "No")
        a, b = v["home"].get(142), v["away"].get(142)
        if a is None or b is None:
            return ""
        prom = (a + b) / 2
        if (regla[0] == "<" and prom < regla[1]) or (regla[0] == ">" and prom > regla[1]):
            return f"⚠ 365: ambos marcaron en {a:.0%} / {b:.0%} de sus partidos"
        return ""
    if m and (mercado in ALARM_STAT_ID or mercado == "Tiros al arco por equipo"):
        sid = ALARM_STAT_ID.get(mercado, 168)
        mas, linea = m.group(2) == "Más", float(m.group(3))
        if m.group(1):  # por equipo: lo del equipo del pick
            lado = lado_de(m.group(1))
            if lado is None or v[lado].get(sid) is None:
                return ""
            total, detalle = v[lado][sid], f"{nombres[lado]} promedia {v[lado][sid]:.1f}"
        else:
            a, b = v["home"].get(sid), v["away"].get(sid)
            if a is None or b is None:
                return ""
            total, detalle = a + b, f"{nombres['home']} {a:.1f} + {nombres['away']} {b:.1f} = {a + b:.1f}"
        if (mas and total < linea) or (not mas and total > linea):
            return f"⚠ 365: {detalle} por partido, {'debajo' if mas else 'encima'} de la línea {linea:g}"
    return ""
