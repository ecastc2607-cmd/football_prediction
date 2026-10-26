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

# Abreviaturas que usan las casas y que no se parecen al nombre completo.
ALIASES = {"psg": "paris saint germain", "man utd": "manchester united", "man city": "manchester city",
           "spurs": "tottenham hotspur", "barca": "barcelona", "atleti": "atletico madrid", "bvb": "borussia dortmund",
           "inter": "internazionale inter", "lyon": "olympique lyonnais lyon", "marsella": "olympique marseille",
           "napoles": "napoli", "om": "olympique marseille"}


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = ALIASES.get(" ".join(s.split()), s)
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
                if h.get("longName") or a.get("longName"):  # "PSG" -> "Paris Saint-Germain"
                    e["names"].append((h.get("longName") or h.get("name", ""), a.get("longName") or a.get("name", "")))
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


# --- Plan de apuestas: chequeo por cupón y simulación de alternativas -------------
# Reglas acordadas con el usuario (oct-2026), sacadas de su propio historial:
# con 3-4 piernas perdía casi todo; las piernas con el modelo >= 65% acertaban
# 83%; "Ambos marcan: Sí" 42%; corners solo rendían donde Reportes los marca fuertes.
PLAN_MAX_LEGS = 2
PLAN_MIN_PROB = 65.0
PLAN_DAILY_BUDGET = 50_000
PLAN_MIN_STAT_STRENGTH = 5.0  # puntos (fuerza de Reportes, encogida) para aceptar corners/tarjetas
MODEL_MARKETS = {"1X2", "Empate no válido", "Doble oportunidad", "Ambos marcan", "Goles"}
STAT_MARKETS_REPORT = {"Corners": "Corners", "Tarjetas": "Tarjetas"}
LIGA_A_CODIGO = [("Bundesliga", "BL1"), ("Premier", "PL"), ("LaLiga", "PD"), ("Serie A", "SA"), ("Ligue 1", "FL1"),
                 ("Champions", "CL"), ("Europa League", "EL"), ("Conference", "ECL"), ("BetPlay", "COL"),
                 ("Nations League", "NT")]


def _codigo_liga(liga) -> str | None:
    for clave, codigo in LIGA_A_CODIGO:
        if isinstance(liga, str) and clave.lower() in liga.lower():
            return codigo
    return None


def _league_strength() -> pd.DataFrame:
    """Liga × mercado de Reportes (diferencia acierto − esperado)."""
    from . import reports
    from .prediction_log import load_log

    log = load_log()
    if log.empty:
        return pd.DataFrame()
    liga, _ = reports.strength_tables(reports.predictions_long(log, include_backtest=True))
    return liga


def leg_issues(r, fuerza: pd.DataFrame | None = None) -> list[str]:
    """Reglas del plan que rompe una pierna (vacío = cumple)."""
    from .reports import STRENGTH_PRIOR

    motivos = []
    prob = pd.to_numeric(r.get("prob_modelo"), errors="coerce")
    if r.get("mercado") == "Ambos marcan" and r.get("lado") == "Sí":
        motivos.append("Ambos marcan: Sí")
    elif r.get("mercado") in MODEL_MARKETS:
        if pd.isna(prob):
            motivos.append("sin predicción del modelo")
        elif prob < PLAN_MIN_PROB:
            motivos.append(f"modelo {prob:.1f}% (< {PLAN_MIN_PROB:.0f}%)")
    elif r.get("mercado") in STAT_MARKETS_REPORT:
        codigo = _codigo_liga(r.get("liga"))
        clave = (codigo, STAT_MARKETS_REPORT[r["mercado"]])
        if fuerza is None or fuerza.empty or clave not in fuerza.index:
            motivos.append(f"{r['mercado'].lower()} sin historial en la liga")
        else:
            # Misma fuerza que colorea Jornada: acercada a neutro con poca muestra.
            fila = fuerza.loc[clave]
            n = float(fila["predicciones"])
            f = float(fila["diferencia"]) * n / (n + STRENGTH_PRIOR)
            if f < PLAN_MIN_STAT_STRENGTH:
                motivos.append(f"{r['mercado'].lower()} no es fortaleza en la liga ({f:+.0f} pts)")
    else:
        motivos.append("mercado sin respaldo del modelo")
    return motivos


def plan_check(df: pd.DataFrame, presupuesto: float = PLAN_DAILY_BUDGET) -> pd.DataFrame:
    """Por cupón: si cumple el plan y qué reglas rompe."""
    if df.empty:
        return pd.DataFrame(columns=["coupon_id", "cumple", "motivos"])
    fuerza = _league_strength()
    df = df.copy()
    df["_motivos"] = [leg_issues(r, fuerza) for r in df.to_dict("records")]
    t = tickets(df)
    t["_dia"] = pd.to_datetime(t["placed_at"], utc=True, errors="coerce").dt.tz_convert(BOGOTA).dt.date
    t = t.sort_values("placed_at")
    t["_acumulado"] = pd.to_numeric(t["monto"], errors="coerce").fillna(0).groupby(t["_dia"]).cumsum()
    filas = []
    for _, c in t.iterrows():
        motivos = []
        if c["n_piernas"] > PLAN_MAX_LEGS:
            motivos.append(f"{int(c['n_piernas'])} selecciones (máx. {PLAN_MAX_LEGS})")
        if c["_acumulado"] > presupuesto:
            motivos.append(f"supera el presupuesto del día (${c['_acumulado']:,.0f})")
        for n, m in zip(df.loc[df["coupon_id"] == c["coupon_id"], "pierna"],
                        df.loc[df["coupon_id"] == c["coupon_id"], "_motivos"]):
            motivos += [f"selección {int(n)}: {x}" for x in m]
        filas.append({"coupon_id": c["coupon_id"], "cumple": not motivos, "motivos": motivos})
    return pd.DataFrame(filas)


def simulate(df: pd.DataFrame, stake: float = 10_000) -> pd.DataFrame:
    """Qué habría pasado con TUS mismas piernas jugadas de otra forma. La cuota
    de cada pierna se estima como la raíz n-ésima de la cuota del cupón (bwin
    no la muestra por separado): es aproximada."""
    if df.empty:
        return pd.DataFrame()
    fuerza = _league_strength()
    d = df[df["estado"] != "Nula"].copy()
    d["cuota_c"] = pd.to_numeric(d["cuota"], errors="coerce")
    d["cuota_pierna"] = d["cuota_c"] ** (1 / d["n_piernas"].astype(float))
    d = d[d["resultado"].isin([ACIERTO, FALLO, DEVUELTO]) & d["cuota_pierna"].notna()]
    d["cumple"] = [not leg_issues(r, fuerza) for r in d.to_dict("records")]
    d["prob"] = pd.to_numeric(d["prob_modelo"], errors="coerce")
    t = tickets(df)
    real = t[t["estado"].isin(["Ganada", "Perdida", "Cobro anticipado"])]
    apostado_real = pd.to_numeric(real["monto"], errors="coerce").sum()
    filas = [{"estrategia": "Lo que hiciste (combinadas)", "apuestas": len(real),
              "acierto_pct": 100 * real["estado"].isin(["Ganada", "Cobro anticipado"]).mean() if len(real) else None,
              "apostado": apostado_real, "ganancia": pd.to_numeric(real["ganancia"], errors="coerce").sum()}]

    def sencillas(nombre, sub):
        if sub.empty:
            return
        g = sub.apply(lambda r: stake * (r["cuota_pierna"] - 1) if r["resultado"] == ACIERTO
                      else 0.0 if r["resultado"] == DEVUELTO else -stake, axis=1).sum()
        filas.append({"estrategia": nombre, "apuestas": len(sub),
                      "acierto_pct": 100 * (sub["resultado"] == ACIERTO).mean(),
                      "apostado": stake * len(sub), "ganancia": g})

    sencillas("Todas tus selecciones como sencillas", d)
    sencillas("Solo selecciones que cumplen el plan (sencillas)", d[d["cumple"]])
    sencillas(f"Solo selecciones con modelo ≥ {PLAN_MIN_PROB:.0f}% (sencillas)", d[d["prob"] >= PLAN_MIN_PROB])
    sencillas("Sin 'Ambos marcan: Sí' (sencillas)", d[~((d["mercado"] == "Ambos marcan") & (d["lado"] == "Sí"))])
    out = pd.DataFrame(filas)
    out["retorno_pct"] = 100 * out["ganancia"] / out["apostado"]
    return out


# --- Analizar un cupón ANTES de apostarlo ----------------------------------------
# Pedido del usuario (oct-2026): pegar el boleto que está armando (BetPlay o bwin)
# y ver si el modelo lo ve factible antes de confirmarlo.
VALUE_MODEL_WEIGHT = 0.5   # misma regla de valor que Tips (value_tips.MODEL_WEIGHT)
VALUE_MIN_EDGE = 0.03


def parse_slip(texto: str) -> list[dict]:
    """Piernas de un boleto sin confirmar. Acepta el formato de BetPlay
    (pick / mercado / partido / cuota, con o sin viñetas y links) y el de bwin
    ("Mercado: pick" / partido / cuota)."""
    lineas = []
    for l in str(texto).replace("\r", "").split("\n"):
        l = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", l)       # [PSG - Lyon](url) -> PSG - Lyon
        l = re.sub(r"^\s*[*•·\-–]\s*", "", l).strip()          # viñetas
        if l:
            lineas.append(l)

    def es_partido(l):
        return " - " in l and _num(l) is None and ":" not in l

    piernas = []
    for i, l in enumerate(lineas):
        if not es_partido(l):
            continue
        antes = [x for x in lineas[max(0, i - 2):i] if _num(x) is None and not es_partido(x)]
        if antes and ":" in antes[-1]:
            mercado, pick = [x.strip() for x in antes[-1].split(":", 1)]
        elif len(antes) >= 2:
            pick, mercado = antes[-2], antes[-1]
        else:
            continue
        cuota = next((_num(x) for x in lineas[i + 1:i + 3] if _num(x) is not None), None)
        local, _, visita = l.partition(" - ")
        piernas.append({"mercado_txt": mercado, "pick": pick, "local": local.strip(), "visita": visita.strip(),
                        "cuota": cuota})
    return piernas


def _season_for(code: str, fecha: datetime) -> int:
    if code == "COL":
        return fecha.year
    return fecha.year if fecha.month >= 7 else fecha.year - 1


def _model_now(code: str | None, local: str, visita: str, fecha: datetime, mercado: str, lado, linea):
    """Probabilidad (%) del modelo calculada en el momento, cuando el partido no
    está en predictions_log (o para corners y tarjetas)."""
    if not code:
        return None
    import numpy as np

    from . import totals_model
    from .poisson_model import predict_match
    from .team_strength import team_strength_for_competition

    temporada = _season_for(code, fecha)
    try:
        fuerza = team_strength_for_competition(code, [temporada, temporada - 1])
    except Exception:
        return None
    equipos = list(fuerza.index)
    if not equipos:
        return None
    h = max(equipos, key=lambda e: _sim(local, e))
    a = max(equipos, key=lambda e: _sim(visita, e))
    if _sim(local, h) < 0.6 or _sim(visita, a) < 0.6:
        return None
    if mercado in ("Corners", "Tarjetas") and linea is not None:
        r = totals_model.predict(code, h, a, "corners" if mercado == "Corners" else "tarjetas", float(linea))
        if not r:
            return None
        return 100 * (r[1] if lado == "Más" else 1 - r[1])
    try:
        pr = predict_match(fuerza, h, a)
    except Exception:
        return None
    m = pr.score_matrix
    g = np.arange(m.shape[0])
    tot = g[:, None] + g[None, :]
    if mercado == "1X2":
        return 100 * {"local": pr.home_win, "empate": pr.draw, "visita": pr.away_win}.get(lado, np.nan)
    if mercado == "Empate no válido":
        w, l = (pr.home_win, pr.away_win) if lado == "local" else (pr.away_win, pr.home_win)
        return 100 * w / (w + l)
    if mercado == "Doble oportunidad":
        return 100 * {"1X": pr.home_win + pr.draw, "X2": pr.away_win + pr.draw,
                      "12": pr.home_win + pr.away_win}.get(lado, np.nan)
    if mercado == "Ambos marcan":
        return 100 * (pr.btts if lado == "Sí" else 1 - pr.btts)
    if mercado == "Goles" and linea is not None:
        mas = float(m[tot > linea].sum())
        return 100 * (mas if lado == "Más" else 1 - mas)
    return None


def _texto_opcion(mercado, lado, linea, local, visita) -> str | None:
    equipo = {"local": local, "visita": visita, "empate": "Empate"}.get(lado, lado)
    if mercado == "1X2":
        return f"Resultado final: {equipo}"
    if mercado == "Empate no válido":
        return f"Empate no válido: {equipo}"
    if mercado == "Doble oportunidad":
        return f"Doble oportunidad: {lado}"
    if mercado == "Ambos marcan":
        return f"Ambos marcan: {lado}"
    if mercado in ("Goles", "Goles 1T", "Corners", "Tarjetas") and linea is not None:
        nombre = {"Goles": "Total de goles", "Goles 1T": "Goles 1er tiempo"}.get(mercado, mercado)
        return f"{nombre}: {lado} de {linea:g}"
    return None


def analyze_slip(texto: str) -> tuple[pd.DataFrame, dict]:
    """(piernas analizadas, resumen del combinado) de un boleto sin confirmar."""
    piernas = parse_slip(texto)
    ahora = datetime.now(timezone.utc)
    fuerza = _league_strength()
    filas = []
    for p in piernas:
        merc, lado, linea = classify(p["mercado_txt"], p["pick"], p["local"], p["visita"])
        g = find_game(p["local"], p["visita"], (ahora - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"))
        liga = g.get("competitionDisplayName") if g else None
        inicio = scores365._utc(g.get("startTime")) if g else None
        nombres_365 = (((g.get("homeCompetitor") or {}).get("longName") or (g.get("homeCompetitor") or {}).get("name", ""),
                        (g.get("awayCompetitor") or {}).get("longName") or (g.get("awayCompetitor") or {}).get("name", ""))
                       if g else None)
        prob = model_prob(p["local"], p["visita"], inicio, merc, lado, linea, nombres_365) if inicio else None
        fuente = "predicción del día" if prob is not None and not pd.isna(prob) else None
        if prob is None or pd.isna(prob):
            prob = _model_now(_codigo_liga(liga), p["local"], p["visita"], ahora, merc, lado, linea)
            fuente = "modelo (calculado ahora)" if prob is not None and not pd.isna(prob) else "sin modelo"
        fila = {"pick": p["pick"], "mercado_txt": p["mercado_txt"], "partido": f"{p['local']} - {p['visita']}",
                "mercado": merc, "lado": lado, "linea": linea, "liga": liga, "inicio": inicio, "cuota": p["cuota"],
                "prob_modelo": None if prob is None or pd.isna(prob) else round(float(prob), 1), "fuente": fuente,
                "prob_casa": None, "cuota_justa": None, "valor_pct": None}
        q = fila["prob_modelo"] / 100 if fila["prob_modelo"] is not None else None
        c = p["cuota"]
        if c:
            fila["prob_casa"] = round(100 / c, 1)
        if q and c:
            ajustada = VALUE_MODEL_WEIGHT * q + (1 - VALUE_MODEL_WEIGHT) / c
            fila["cuota_justa"] = round(1 / q, 2)
            fila["valor_pct"] = round(100 * (ajustada * c - 1), 1)
        fila["motivos"] = leg_issues(fila, fuerza)
        fila["recomendada"] = (not fila["motivos"] and fila["valor_pct"] is not None
                               and fila["valor_pct"] >= 100 * VALUE_MIN_EDGE)
        fila["alternativas"] = pd.DataFrame()
        if not fila["recomendada"] and g:
            try:
                fila["alternativas"] = alternatives(
                    _codigo_liga(liga), g.get("id"), p["local"], p["visita"], p["cuota"],
                    excluir=_texto_opcion(merc, lado, linea, p["local"], p["visita"]), fecha=ahora)
            except Exception:
                pass
        filas.append(fila)
    df = pd.DataFrame(filas)
    resumen: dict = {}
    if df.empty:
        return df, resumen
    cuotas = pd.to_numeric(df["cuota"], errors="coerce")
    probs = pd.to_numeric(df["prob_modelo"], errors="coerce") / 100
    cuota_total = float(cuotas.prod()) if cuotas.notna().all() else None
    prob_total = float(probs.prod()) if probs.notna().all() else None
    motivos = []
    if len(df) > PLAN_MAX_LEGS:
        motivos.append(f"{len(df)} selecciones (máx. {PLAN_MAX_LEGS})")
    if df["partido"].duplicated().any():
        motivos.append("dos selecciones del mismo partido (están correlacionadas)")
    for n, m in enumerate(df["motivos"], 1):
        motivos += [f"selección {n}: {x}" for x in m]
    valor_total = None
    if cuota_total and prob_total:
        ajustadas = [VALUE_MODEL_WEIGHT * q + (1 - VALUE_MODEL_WEIGHT) / c for q, c in zip(probs, cuotas)]
        prod = 1.0
        for x in ajustadas:
            prod *= x
        valor_total = 100 * (prod * cuota_total - 1)
    valores = pd.to_numeric(df["valor_pct"], errors="coerce")
    if (valores < 100 * VALUE_MIN_EDGE).any() or valores.isna().any():
        motivos.append("alguna selección no tiene valor contra la cuota")
    sencillas = df[(valores >= 100 * VALUE_MIN_EDGE) & df["motivos"].map(len).eq(0)]
    sugerido = []
    for _, f in df.iterrows():
        if f["recomendada"]:
            sugerido.append({"opcion": f"{f['mercado_txt']}: {f['pick']}", "partido": f["partido"],
                             "cuota": f["cuota"], "prob_modelo": f["prob_modelo"], "cambio": False})
        elif isinstance(f["alternativas"], pd.DataFrame) and not f["alternativas"].empty                 and f["alternativas"].iloc[0]["nivel"].startswith("✅"):
            mejor = f["alternativas"].iloc[0]
            sugerido.append({"opcion": mejor["opcion"], "partido": f["partido"], "cuota": mejor["cuota"],
                             "prob_modelo": mejor["prob_modelo"], "cambio": True})
    sugerido = pd.DataFrame(sugerido)
    if len(sugerido) != len(df) or not sugerido["cambio"].any():
        sugerido = pd.DataFrame()  # solo si TODAS las selecciones quedan cubiertas y algo cambió
    resumen = {"piernas": len(df), "cuota": cuota_total, "prob": prob_total,
               "cuota_justa": 1 / prob_total if prob_total else None, "valor_pct": valor_total,
               "motivos": motivos, "sencillas_ok": sencillas["pick"].tolist(), "sugerido": sugerido,
               "factible": bool(valor_total is not None and valor_total >= 100 * VALUE_MIN_EDGE and not motivos)}
    return df, resumen


# --- Alternativas del mismo partido para una selección no recomendada -------------
# Pedido del usuario (oct-2026): si una selección no se recomienda, proponer otra
# opción del MISMO partido, de lo que ofrece la casa, con una cuota parecida.
ALT_MAX = 3
ALT_WEIGHT_MAIN = 0.5      # 1X2, doble oportunidad, empate no válido, goles, ambos marcan
ALT_WEIGHT_OTHER = 0.35    # corners, tarjetas, hándicap, 1er tiempo... (igual que Tips)


def _match_model(code: str | None, local: str, visita: str, fecha: datetime):
    """(predicción de goles, nombre local, nombre visita) del modelo, con los
    nombres del proyecto; None si no se puede."""
    if not code:
        return None
    from .poisson_model import predict_match
    from .team_strength import team_strength_for_competition

    temporada = _season_for(code, fecha)
    try:
        fuerza = team_strength_for_competition(code, [temporada, temporada - 1])
    except Exception:
        return None
    equipos = list(fuerza.index)
    if not equipos:
        return None
    h = max(equipos, key=lambda e: _sim(local, e))
    a = max(equipos, key=lambda e: _sim(visita, e))
    if _sim(local, h) < 0.6 or _sim(visita, a) < 0.6:
        return None
    try:
        return predict_match(fuerza, h, a), h, a
    except Exception:
        return None


def alternatives(code: str | None, game_id, local: str, visita: str, cuota_ref: float | None,
                 excluir: str | None = None, fecha: datetime | None = None) -> pd.DataFrame:
    """Opciones del mismo partido (líneas de BWIN en 365scores) con valor y
    modelo >= PLAN_MIN_PROB, ordenadas por cercanía a `cuota_ref`."""
    import math

    import numpy as np

    from . import events365, special_markets, totals_model

    if not game_id:
        return pd.DataFrame()
    fecha = fecha or datetime.now(timezone.utc)
    modelo = _match_model(code, local, visita, fecha)
    if modelo is None:
        return pd.DataFrame()
    pr, h, a = modelo
    lineas = pd.DataFrame(events365.line_rows(int(game_id), scores365.game_lines(int(game_id))))
    if lineas.empty:
        return pd.DataFrame()
    m = pr.score_matrix
    g = np.arange(m.shape[0])
    H, A = np.meshgrid(g, g, indexing="ij")
    hx, ax = pr.home_xg, pr.away_xg
    share = special_markets.DEFAULT_FIRST_HALF_SHARE
    try:
        share = pd.read_csv(special_markets.SHARE_PATH, index_col=0).iloc[:, 0].to_dict().get(code, share)
    except Exception:
        pass
    nombre = {"Home": local, "Away": visita, "1": local, "2": visita, "X": "Empate"}
    filas = []
    for r in lineas.itertuples(index=False):
        c = r.odds
        if not c or c <= 1.01:
            continue
        tipo, op = int(r.line_type_id), str(r.option)
        try:
            ln = float(r.line) if r.line is not None and str(r.line) not in ("", "nan", "None") else None
        except ValueError:
            ln = None
        q, texto, peso = None, None, ALT_WEIGHT_OTHER
        if tipo == 1 and op in ("1", "X", "2"):
            q = {"1": pr.home_win, "X": pr.draw, "2": pr.away_win}[op]
            texto, peso = f"Resultado final: {nombre[op]}", ALT_WEIGHT_MAIN
        elif tipo == 14 and op in ("1X", "12", "X2"):
            q = {"1X": pr.home_win + pr.draw, "12": pr.home_win + pr.away_win, "X2": pr.away_win + pr.draw}[op]
            texto, peso = f"Doble oportunidad: {op}", ALT_WEIGHT_MAIN
        elif tipo == 15 and op in ("Home", "Away"):
            w, l = (pr.home_win, pr.away_win) if op == "Home" else (pr.away_win, pr.home_win)
            q, texto, peso = w / (w + l), f"Empate no válido: {nombre[op]}", ALT_WEIGHT_MAIN
        elif tipo == 12 and op in ("Yes", "No"):
            q = pr.btts if op == "Yes" else 1 - pr.btts
            texto, peso = f"Ambos marcan: {'Sí' if op == 'Yes' else 'No'}", ALT_WEIGHT_MAIN
        elif tipo == 3 and ln is not None and op in ("Over", "Under"):
            mas = float(m[(H + A) > ln].sum())
            q = mas if op == "Over" else 1 - mas
            texto, peso = f"Total de goles: {'Más' if op == 'Over' else 'Menos'} de {ln:g}", ALT_WEIGHT_MAIN
        elif tipo == 9 and ln is not None and op in ("Over", "Under"):
            lam = (hx + ax) * share
            from scipy.stats import poisson
            mas = 1 - float(poisson.cdf(math.floor(ln), lam))
            q = mas if op == "Over" else 1 - mas
            texto = f"Goles 1er tiempo: {'Más' if op == 'Over' else 'Menos'} de {ln:g}"
        elif tipo == 11 and ln is not None and op in ("Home", "Away") and abs(ln * 2 - round(ln * 2)) < 1e-9 \
                and round(ln * 2) % 2:  # solo líneas .5 (sin devolución)
            margen = (H - A) if op == "Home" else (A - H)
            linea_lado = ln if op == "Home" else -ln
            q = float(m[(margen + linea_lado) > 0].sum())
            texto = f"Hándicap asiático: {nombre[op]} {linea_lado:+g}"
        elif tipo in (137, 141) and ln is not None and op in ("Over", "Under"):
            res = totals_model.predict(code, h, a, "corners" if tipo == 137 else "tarjetas", ln)
            if res:
                q = res[1] if op == "Over" else 1 - res[1]
                texto = f"{'Corners' if tipo == 137 else 'Tarjetas'}: {'Más' if op == 'Over' else 'Menos'} de {ln:g}"
        elif tipo in (139, 140) and ln is not None and op in ("Over", "Under"):
            st = special_markets.stat_prediction(code, h, a, "sot" if tipo == 139 else "off", ln)
            if st:
                q = st["p_mas"] if op == "Over" else 1 - st["p_mas"]
                texto = (f"{'Tiros al arco' if tipo == 139 else 'Fueras de juego'}: "
                         f"{'Más' if op == 'Over' else 'Menos'} de {ln:g}")
        elif tipo in (144, 145) and op in ("Yes", "No"):
            recibe = A if tipo == 144 else H  # 144 = valla invicta del local: el visitante no marca
            invicta = float(m[recibe == 0].sum())
            q = invicta if op == "Yes" else 1 - invicta
            equipo = local if tipo == 144 else visita
            texto = f"Valla invicta {equipo}: {'Sí' if op == 'Yes' else 'No'}"
        elif tipo == 7 and op in ("Home", "Away"):
            lam = hx + ax
            q = (hx if op == "Home" else ax) / lam * (1 - math.exp(-lam)) if lam > 0 else None
            texto = f"Primer gol: {nombre[op]}"
        if q is None or texto is None or (excluir and texto.lower() == excluir.lower()):
            continue
        ajustada = peso * q + (1 - peso) / c
        filas.append({"opcion": texto, "cuota": c, "prob_modelo": round(100 * q, 1), "prob_casa": round(100 / c, 1),
                      "valor_pct": round(100 * (ajustada * c - 1), 1),
                      "distancia": abs(math.log(c / cuota_ref)) if cuota_ref else 0.0})
    alt = pd.DataFrame(filas)
    if alt.empty:
        return alt
    # Dos niveles: con valor claro (regla de Tips) y "margen justo" (no pierde,
    # pero la ventaja es chica). Nunca por debajo del mínimo del plan.
    alt = alt[(alt["prob_modelo"] >= PLAN_MIN_PROB) & (alt["valor_pct"] >= 0)].copy()
    alt["nivel"] = alt["valor_pct"].map(lambda v: "✅ Recomendada" if v >= 100 * VALUE_MIN_EDGE else "🟡 Margen justo")
    alt["_n"] = (alt["valor_pct"] < 100 * VALUE_MIN_EDGE).astype(int)
    alt = alt.sort_values(["_n", "distancia", "valor_pct"], ascending=[True, True, False]).head(ALT_MAX)
    return alt.drop(columns=["distancia", "_n"])[["nivel", "opcion", "cuota", "prob_modelo", "prob_casa", "valor_pct"]]
