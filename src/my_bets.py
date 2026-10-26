"""
Mis apuestas: los cupones reales del usuario (copiados del historial de bwin),
resueltos pierna por pierna con datos de 365scores y comparados con el modelo.

Pedido del usuario (oct-2026): pegar el historial de bwin en Reportes y ver
su acierto real por mercado, por tamaño de cupón y frente a lo que decía el
modelo, semana a semana.

Privacidad: son montos personales y el repositorio de GitHub es público, así
que data/tracking/my_bets.csv NO se versiona (.gitignore): queda solo en el PC.

Formato que se pega (tal cual lo muestra bwin al copiar el historial):
    TriplePerdida
    24 de oct de 2026 • 10:42 a. m.
    ID del cupón: 13270243509
    Apuesta sin empate: Villarreal
    Valencia - Villarreal
    ...
    Cuotas:
    3.06
    Apuesta:
    $30,000
    Pago:                (si hubo)
    $54,626
y las sencillas: "Sencilla / @ / 1.18 / Ganadas / fecha / ID / pierna / partido / Apuesta / Pago".
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import config, scores365

PATH = config.ROOT_DIR / "data" / "tracking" / "my_bets.csv"
COLUMNS = [
    "coupon_id", "placed_at", "tipo", "n_piernas", "estado", "cuota", "monto", "pago",
    "pierna", "mercado_bwin", "pick_bwin", "partido_bwin", "local", "visita",
    "mercado", "lado", "linea", "game_id", "liga", "inicio", "marcador", "dato",
    "resultado", "prob_modelo", "resuelto_at",
]
TIPOS = {"Sencilla": 1, "Doble": 2, "Triple": 3, "Cuádruple": 4, "Quíntuple": 5, "Séxtuple": 6,
         "Séptuple": 7, "Óctuple": 8}
ESTADOS = {"Ganadas": "Ganada", "Ganada": "Ganada", "Perdida": "Perdida", "Perdidas": "Perdida", "Nula": "Nula",
           "Cobro de apuesta confirmado": "Cobro anticipado", "Abierta": "Abierta", "Abiertas": "Abierta",
           "Pendiente": "Abierta"}
MESES = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8, "sept": 9, "sep": 9,
         "oct": 10, "nov": 11, "dic": 12}
ACIERTO, FALLO, DEVUELTO, SIN_DATO, PENDIENTE = "acierto", "fallo", "devuelto", "sin resultado", "pendiente"
BOGOTA = timezone(timedelta(hours=-5))
# Una pierna sin partido encontrado se reintenta hasta estos días después de apostada.
RETRY_DAYS = 5


# --- Lectura del texto de bwin ----------------------------------------------------

def _num(texto: str) -> float | None:
    t = str(texto).strip().strip("()").replace("$", "").replace(",", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def _fecha(linea: str) -> str | None:
    """'25 de oct de 2026 • 05:23 p. m.' -> ISO UTC (la hora de bwin es de Colombia)."""
    m = re.match(r"\s*(\d{1,2}) de (\w+)\.? de (\d{4})\s*•\s*(\d{1,2}):(\d{2})\s*([ap])\.?\s*m\.?", linea, re.I)
    if not m:
        return None
    dia, mes, anio, hh, mm, ampm = m.groups()
    mes_n = MESES.get(mes.lower()[:4].rstrip(".")) or MESES.get(mes.lower()[:3])
    if not mes_n:
        return None
    h = int(hh) % 12 + (12 if ampm.lower() == "p" else 0)
    local = datetime(int(anio), mes_n, int(dia), h, int(mm), tzinfo=BOGOTA)
    return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cabecera(linea: str) -> tuple[str, str | None] | None:
    """'TriplePerdida' -> ('Triple', 'Perdida'); 'Sencilla' -> ('Sencilla', None)."""
    for tipo in sorted(TIPOS, key=len, reverse=True):
        if linea.startswith(tipo):
            resto = linea[len(tipo):].strip()
            if not resto:
                return tipo, None
            for e in sorted(ESTADOS, key=len, reverse=True):
                if resto.startswith(e):
                    return tipo, ESTADOS[e]
    return None


def parse_coupons(texto: str) -> list[dict]:
    """Cupones del texto pegado: [{coupon_id, placed_at, tipo, estado, cuota,
    monto, pago, piernas: [(mercado, pick, partido)]}]."""
    lineas = [l.strip() for l in str(texto).replace("\r", "").split("\n")]
    lineas = [l for l in lineas if l]
    bloques, actual = [], None
    for l in lineas:
        cab = _cabecera(l)
        if cab:
            actual = {"tipo": cab[0], "estado": cab[1], "lineas": []}
            bloques.append(actual)
        elif actual is not None:
            actual["lineas"].append(l)
    cupones = []
    for b in bloques:
        ls = b["lineas"]
        c = {"tipo": b["tipo"], "estado": b["estado"], "coupon_id": None, "placed_at": None, "cuota": None,
             "monto": None, "pago": None, "piernas": []}
        i = 0
        while i < len(ls):
            l = ls[i]
            sig = ls[i + 1] if i + 1 < len(ls) else ""
            if l == "@":
                c["cuota"] = _num(sig); i += 2; continue
            if l in ESTADOS and c["estado"] is None:
                c["estado"] = ESTADOS[l]; i += 1; continue
            if _fecha(l):
                c["placed_at"] = _fecha(l); i += 1; continue
            if l.startswith("ID del cupón"):
                c["coupon_id"] = l.split(":", 1)[1].strip(); i += 1; continue
            if l.rstrip(":") in ("Cuotas", "Apuesta", "Pago"):
                clave = {"Cuotas": "cuota", "Apuesta": "monto", "Pago": "pago"}[l.rstrip(":")]
                c[clave] = _num(sig); i += 2; continue
            if ":" in l and " - " in sig and not sig.startswith(("Cuotas", "Apuesta")):
                mercado, pick = [x.strip() for x in l.split(":", 1)]
                c["piernas"].append((mercado, pick, sig)); i += 2; continue
            if ":" in l and sig.upper() == "VICTORIA ANTICIPADA" and i + 2 < len(ls) and " - " in ls[i + 2]:
                mercado, pick = [x.strip() for x in l.split(":", 1)]
                c["piernas"].append((mercado, pick, ls[i + 2])); i += 3; continue
            i += 1
        if c["coupon_id"] and c["piernas"]:
            cupones.append(c)
    return cupones


# --- Mercados ---------------------------------------------------------------------

def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    quitar = {"fc", "cf", "ac", "as", "sc", "afc", "club", "deportivo", "sa", "s", "a", "cd", "rc", "ssc", "1907"}
    return " ".join(w for w in s.split() if w not in quitar)


def _sim(a, b) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    return r + (0.3 if a in b or b in a else 0.0)


def classify(mercado: str, pick: str, local: str, visita: str) -> tuple[str, str | None, float | None]:
    """(mercado normalizado, lado, línea). Lado: 'local'/'visita'/'empate', 'Sí'/'No', 'Más'/'Menos'."""
    m = mercado.lower()
    lm = re.match(r"(Más|Menos) de ([0-9]+(?:[.,][0-9]+)?)", pick)
    linea = float(lm.group(2).replace(",", ".")) if lm else None
    lado_total = lm.group(1) if lm else None

    def equipo(p):
        if _norm(p) in ("empate", "x", "draw"):
            return "empate"
        return "local" if _sim(p, local) >= _sim(p, visita) else "visita"

    if m.startswith("resultado final") or m in ("1x2", "resultado del partido"):
        return "1X2", equipo(pick), None
    if "sin empate" in m:
        return "Empate no válido", equipo(pick), None
    if m.startswith("doble oportunidad"):
        partes = [p.strip() for p in re.split(r"\s+o\s+|/", pick)]
        lados = {equipo(p) for p in partes}
        lado = "1X" if lados == {"local", "empate"} else "X2" if lados == {"visita", "empate"} else "12"
        return "Doble oportunidad", lado, None
    if m.startswith("ambos equipos"):
        return "Ambos marcan", "Sí" if pick.lower().startswith("s") else "No", None
    if "esquina" in m:
        return "Corners", lado_total, linea
    if "tarjeta" in m:
        return "Tarjetas", lado_total, linea
    if "goles" in m and ("1.ª parte" in m or "1a parte" in m or "primera parte" in m or "1er tiempo" in m):
        return "Goles 1T", lado_total, linea
    if "goles" in m:
        return "Goles", lado_total, linea
    return "Otro", None, None


# --- Resolución con 365scores -------------------------------------------------------

_DAY_CACHE: dict = {}
EXCLUIR = ("(W)", "(F)", "U19", "U20", "U21", "U23", "Women", "Femenino", "Fem.")


def _games_del_dia(dia: str) -> list[dict]:
    """Partidos de fútbol de ese día (UTC) con nombres en inglés y en español."""
    if dia not in _DAY_CACHE:
        fecha = datetime.strptime(dia, "%Y-%m-%d").strftime("%d/%m/%Y")
        por_id: dict = {}
        # La lista general del día viene recortada (~700 partidos): las ligas del
        # proyecto se piden aparte para que nunca falten (ej. Liga BetPlay).
        nuestras = ",".join(str(i) for i in scores365.COMPETITION_IDS.values())
        consultas = [(lang, extra) for lang in (1, 14) for extra in ({"sports": 1}, {"competitions": nuestras})]
        for lang, extra in consultas:
            data = scores365._get("/games/", {**extra, "startDate": fecha, "endDate": fecha, "langId": lang}) or {}
            for g in data.get("games", []):
                h, a = (g.get("homeCompetitor") or {}), (g.get("awayCompetitor") or {})
                if any(x in (h.get("name", "") + a.get("name", "")) for x in EXCLUIR):
                    continue
                e = por_id.setdefault(g["id"], {"game": g, "names": []})
                e["names"].append((h.get("name", ""), a.get("name", "")))
                if lang == 1:
                    e["game"] = g
        _DAY_CACHE[dia] = list(por_id.values())
    return _DAY_CACHE[dia]


def find_game(local: str, visita: str, placed_at: str) -> dict | None:
    """El partido de 365 que mejor coincide, entre 1 día antes y 4 después de apostar."""
    base = datetime.strptime(placed_at[:10], "%Y-%m-%d")
    mejor, puntaje = None, 0.0
    for delta in range(-1, 5):
        dia = (base + timedelta(days=delta)).strftime("%Y-%m-%d")
        for e in _games_del_dia(dia):
            for h, a in e["names"]:
                sh, sa = _sim(local, h), _sim(visita, a)
                p = sh + sa
                if p > puntaje and max(sh, sa) >= 0.8 and min(sh, sa) >= 0.45:
                    mejor, puntaje = e["game"], p
    return mejor if puntaje >= 1.4 else None


def _stats(game_id: int) -> dict:
    from .stats365_backfill import _stats as stats365
    return stats365(game_id) or {}


def _medio_tiempo(game_id: int) -> tuple | None:
    g = (scores365._get("/game/", {"gameId": game_id}) or {}).get("game", {})
    ht = next((s for s in g.get("stages") or [] if s.get("id") == 7), None)
    return (ht.get("homeCompetitorScore"), ht.get("awayCompetitorScore")) if ht else None


def settle_leg(mercado: str, lado, linea, hg: int, ag: int, game_id: int) -> tuple[str, str]:
    """(resultado, dato usado)."""
    if mercado == "1X2":
        real = "local" if hg > ag else "visita" if ag > hg else "empate"
        return (ACIERTO if real == lado else FALLO), ""
    if mercado == "Empate no válido":
        d = hg - ag if lado == "local" else ag - hg
        return (DEVUELTO if d == 0 else ACIERTO if d > 0 else FALLO), ""
    if mercado == "Doble oportunidad":
        ok = {"1X": hg >= ag, "X2": ag >= hg, "12": hg != ag}.get(lado)
        return (SIN_DATO if ok is None else ACIERTO if ok else FALLO), ""
    if mercado == "Ambos marcan":
        ambos = hg > 0 and ag > 0
        return (ACIERTO if ambos == (lado == "Sí") else FALLO), ""
    if linea is None or lado not in ("Más", "Menos"):
        return SIN_DATO, ""
    if mercado == "Goles":
        total = hg + ag
    elif mercado == "Goles 1T":
        ht = _medio_tiempo(game_id)
        if not ht or None in ht:
            return SIN_DATO, ""
        total = int(ht[0]) + int(ht[1])
    elif mercado in ("Corners", "Tarjetas"):
        st = _stats(game_id)
        if not st:
            return SIN_DATO, ""
        if mercado == "Corners":
            total = sum((st.get(s) or {}).get("corners", 0) for s in ("home", "away"))
        else:
            total = sum((st.get(s) or {}).get("tarjetas_amarillas", 0) + (st.get(s) or {}).get("tarjetas_rojas", 0)
                        for s in ("home", "away"))
    else:
        return SIN_DATO, ""
    ok = total > linea if lado == "Más" else total < linea
    return (ACIERTO if ok else FALLO), f"{total:g}"


# --- Probabilidad del modelo para la pierna ---------------------------------------

def model_prob(local: str, visita: str, inicio: str | None, mercado: str, lado, linea=None,
               otros_nombres: tuple | None = None) -> float | None:
    """Lo que decía el modelo (predictions_log, %) para 1X2, empate no válido,
    doble oportunidad, ambos marcan y goles 2.5. None si no hay predicción."""
    from .prediction_log import load_log

    log = load_log()
    if log.empty or not inicio:
        return None
    d = log[log["utc_date"].astype(str).str[:10] == str(inicio)[:10]]
    if d.empty:
        return None
    pares = [(local, visita)] + ([otros_nombres] if otros_nombres else [])
    sc = d.apply(lambda r: max(_sim(h, r["home_team"]) + _sim(a, r["away_team"]) for h, a in pares), axis=1)
    if sc.max() < 1.3:
        return None
    r = d.loc[sc.idxmax()]
    num = lambda c: pd.to_numeric(r.get(c), errors="coerce")  # noqa: E731
    ph, px, pa = num("pred_home_pct"), num("pred_draw_pct"), num("pred_away_pct")
    if mercado == "1X2":
        return {"local": ph, "empate": px, "visita": pa}.get(lado)
    if mercado == "Empate no válido":
        w, l = (ph, pa) if lado == "local" else (pa, ph)
        return 100 * w / (w + l) if w + l else None
    if mercado == "Doble oportunidad":
        return {"1X": ph + px, "X2": pa + px, "12": ph + pa}.get(lado)
    if mercado == "Ambos marcan":
        b = num("btts_pct")
        return b if lado == "Sí" else 100 - b
    if mercado == "Goles" and linea == 2.5 and lado in ("Más", "Menos"):
        o = num("over_2_5_pct")
        return o if lado == "Más" else 100 - o
    return None


# --- Guardar / cargar / resolver -----------------------------------------------------

def load() -> pd.DataFrame:
    if not PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(PATH, dtype={"coupon_id": str})


def _save(df: pd.DataFrame) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(PATH, index=False)


def add_coupons(texto: str) -> tuple[int, int]:
    """Agrega los cupones pegados (los ya guardados se actualizan: estado y
    pago pueden cambiar). Devuelve (nuevos, actualizados)."""
    cupones = parse_coupons(texto)
    df = load()
    existentes = set(df["coupon_id"].astype(str)) if not df.empty else set()
    filas, nuevos, actualizados = [], 0, 0
    for c in cupones:
        if c["coupon_id"] in existentes:
            m = df["coupon_id"].astype(str) == c["coupon_id"]
            df.loc[m, "estado"] = c["estado"]
            df.loc[m, "pago"] = c["pago"]
            actualizados += 1
            continue
        nuevos += 1
        for n, (mercado, pick, partido) in enumerate(c["piernas"], 1):
            local, _, visita = partido.partition(" - ")
            merc, lado, linea = classify(mercado, pick, local.strip(), visita.strip())
            filas.append({
                "coupon_id": c["coupon_id"], "placed_at": c["placed_at"], "tipo": c["tipo"],
                "n_piernas": len(c["piernas"]), "estado": c["estado"], "cuota": c["cuota"], "monto": c["monto"],
                "pago": c["pago"], "pierna": n, "mercado_bwin": mercado, "pick_bwin": pick, "partido_bwin": partido,
                "local": local.strip(), "visita": visita.strip(), "mercado": merc, "lado": lado, "linea": linea,
                "resultado": PENDIENTE,
            })
    if filas:
        nuevas = pd.DataFrame(filas, columns=COLUMNS)
        df = nuevas if df.empty else pd.concat([df, nuevas], ignore_index=True)
    _save(df)
    return nuevos, actualizados


def resolve() -> int:
    """Resuelve las piernas pendientes. Devuelve cuántas resolvió."""
    df = load()
    if df.empty:
        return 0
    for col in ("resultado", "marcador", "dato", "liga", "inicio", "resuelto_at"):
        df[col] = df[col].astype(object)
    ahora = datetime.now(timezone.utc)
    hechas = 0
    for i in df.index[df["resultado"] == PENDIENTE]:
        r = df.loc[i]
        g = find_game(r["local"], r["visita"], str(r["placed_at"] or ahora.isoformat()))
        if g is None:
            if ahora - pd.Timestamp(r["placed_at"]).to_pydatetime() > timedelta(days=RETRY_DAYS):
                df.at[i, "resultado"] = SIN_DATO
            continue
        df.at[i, "game_id"] = g["id"]
        df.at[i, "liga"] = g.get("competitionDisplayName")
        df.at[i, "inicio"] = scores365._utc(g.get("startTime"))
        if g.get("statusGroup") != scores365.STATUS_FINISHED:
            continue
        hg, ag = scores365._score(g.get("homeCompetitor")), scores365._score(g.get("awayCompetitor"))
        if hg is None or ag is None:
            continue
        linea = pd.to_numeric(r["linea"], errors="coerce")
        res, dato = settle_leg(r["mercado"], r["lado"], None if pd.isna(linea) else float(linea), hg, ag, g["id"])
        df.at[i, "marcador"] = f"{hg}-{ag}"
        df.at[i, "dato"] = dato
        df.at[i, "resultado"] = res
        try:
            p = model_prob(r["local"], r["visita"], df.at[i, "inicio"], r["mercado"], r["lado"],
                           None if pd.isna(linea) else float(linea),
                           ((g.get("homeCompetitor") or {}).get("name", ""), (g.get("awayCompetitor") or {}).get("name", "")))
            df.at[i, "prob_modelo"] = None if p is None or pd.isna(p) else round(float(p), 1)
        except Exception:
            pass
        df.at[i, "resuelto_at"] = ahora.strftime("%Y-%m-%dT%H:%M:%SZ")
        hechas += 1
    _save(df)
    return hechas


# --- Resúmenes para Reportes ------------------------------------------------------------

def tickets(df: pd.DataFrame) -> pd.DataFrame:
    """Un renglón por cupón con su ganancia y cuántas piernas fallaron."""
    if df.empty:
        return pd.DataFrame()
    t = df.groupby("coupon_id").agg(
        placed_at=("placed_at", "first"), tipo=("tipo", "first"), n_piernas=("n_piernas", "first"),
        estado=("estado", "first"), cuota=("cuota", "first"), monto=("monto", "first"), pago=("pago", "first"),
        fallos=("resultado", lambda s: int((s == FALLO).sum())),
        pendientes=("resultado", lambda s: int((s == PENDIENTE).sum())),
    ).reset_index()
    monto = pd.to_numeric(t["monto"], errors="coerce")
    pago = pd.to_numeric(t["pago"], errors="coerce").fillna(0)
    t["ganancia"] = (pago - monto).where(t["estado"] != "Abierta")
    t.loc[t["estado"] == "Nula", "ganancia"] = 0.0
    return t.sort_values("placed_at", ascending=False)
