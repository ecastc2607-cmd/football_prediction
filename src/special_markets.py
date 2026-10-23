"""
Mercados especiales (vista "Avanzada"): hándicap asiático y de 3 vías, empate
no válido, mercados por tiempo, goles por equipo, valla invicta, primer gol,
penal, tarjeta roja, tiros al arco y fueras de juego.

Pedido del usuario (oct-2026): buscar mercados con acierto > 75% y "fuerza
alta". Dos trampas que este módulo evita:
  1. Un acierto alto no dice nada solo: "más de 0.5 goles" acierta 92%... y
     paga 1.07. Lo que importa es (a) si el modelo eleva el acierto por encima
     de la frecuencia normal del mercado y (b) si la cuota paga más que la
     probabilidad real. Por eso cada mercado se compara contra su frecuencia
     base y, cuando hay cuotas guardadas de BWIN, contra la cuota.
  2. Buscar patrones en muchos equipos × mercados encuentra "rachas" que son
     azar. Todo se mide walk-forward (cada partido se predice solo con los
     anteriores) y los patrones por equipo se validan en una segunda mitad.

Datos:
  - goles finales y al descanso (data/processed + historiales versionados);
  - eventos y cuotas especiales de 365scores (events365.py);
  - estadísticas finales (match_stats_log.csv).

Uso (regenera el backtest que lee la vista):
    python -m src.special_markets
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

from . import config

BACKTEST_PATH = config.ROOT_DIR / "data" / "tracking" / "special_markets_backtest.csv.gz"  # comprimido: ~10x menos
SHARE_PATH = config.ROOT_DIR / "data" / "tracking" / "special_markets_share.csv"
MIN_PROB = 0.75          # umbral del pedido del usuario
MIN_PREV_MATCHES = 5     # cada equipo con al menos estos partidos previos
DEFAULT_FIRST_HALF_SHARE = 0.44
RHO = -0.10              # Dixon-Coles, igual que poisson_model
_G = np.arange(9)
_H, _A = np.meshgrid(_G, _G, indexing="ij")


def score_matrix(home_xg: float, away_xg: float) -> np.ndarray:
    m = np.outer(poisson.pmf(_G, home_xg), poisson.pmf(_G, away_xg))
    m[0, 0] *= max(1 - home_xg * away_xg * RHO, 0)
    m[0, 1] *= max(1 + home_xg * RHO, 0)
    m[1, 0] *= max(1 + away_xg * RHO, 0)
    m[1, 1] *= max(1 - RHO, 0)
    return m / m.sum()


# --- Hándicap asiático con líneas de cuarto (.25 / .75) ---------------------

def ah_settle(margin: int, line: float) -> float:
    """Fracción de la apuesta que GANA (1), DEVUELVE (0) o PIERDE (-1), con
    medias para las líneas de cuarto. `margin` = goles a favor − en contra del
    equipo apostado; `line` = su hándicap (+1.5, -0.25...)."""
    if abs(line * 4 - round(line * 4)) < 1e-9 and round(line * 4) % 2:  # .25 / .75
        return (ah_settle(margin, line - 0.25) + ah_settle(margin, line + 0.25)) / 2
    x = margin + line
    return 1.0 if x > 0 else (0.0 if x == 0 else -1.0)


def ah_return(margin: int, line: float, odds: float) -> float:
    """Ganancia neta por unidad apostada."""
    if abs(line * 4 - round(line * 4)) < 1e-9 and round(line * 4) % 2:
        return (ah_return(margin, line - 0.25, odds) + ah_return(margin, line + 0.25, odds)) / 2
    x = margin + line
    return odds - 1 if x > 0 else (0.0 if x == 0 else -1.0)


def ah_expected_return(m: np.ndarray, side_is_home: bool, line: float, odds: float) -> float:
    margen = (_H - _A) if side_is_home else (_A - _H)
    return float(sum(m[i, j] * ah_return(int(margen[i, j]), line, odds)
                     for i in range(len(_G)) for j in range(len(_G))))


def ah_win_prob(m: np.ndarray, side_is_home: bool, line: float) -> tuple[float, float]:
    """(P(gana), P(devuelve)) para líneas .0 y .5."""
    margen = (_H - _A) if side_is_home else (_A - _H)
    x = margen + line
    return float(m[x > 0].sum()), float(m[x == 0].sum())


# --- Catálogo de mercados de goles ------------------------------------------

@dataclass
class Match:
    home_xg: float
    away_xg: float
    share1: float = DEFAULT_FIRST_HALF_SHARE  # proporción de goles en el 1er tiempo (por liga)


def _e0(lam: float) -> float:
    return 1 - math.exp(-lam)


def _p_win_half(fav: float, und: float) -> float:
    """P(el de goles esperados `fav` gana esa mitad), Poisson independiente."""
    return float(sum(poisson.pmf(i, und) * (1 - poisson.cdf(i, fav)) for i in range(10)))


def goal_market_probs(x: Match) -> list[tuple[str, str, float, float]]:
    """(familia, mercado, P(gana sin contar devolución), P(devuelve)) para un
    partido. Todo se expresa desde el FAVORITO (más goles esperados), así un
    mismo mercado junta partidos parecidos."""
    m = score_matrix(x.home_xg, x.away_xg)
    fav_local = x.home_xg >= x.away_xg
    fx, ux = (x.home_xg, x.away_xg) if fav_local else (x.away_xg, x.home_xg)
    gf = _H if fav_local else _A
    gu = _A if fav_local else _H
    d = gf - gu
    tot = _H + _A
    l1, l2 = (fx + ux) * x.share1, (fx + ux) * (1 - x.share1)
    f1, u1, f2, u2 = fx * x.share1, ux * x.share1, fx * (1 - x.share1), ux * (1 - x.share1)
    s = lambda mask: float(m[mask].sum())  # noqa: E731
    pw, pp = s(d > 0), s(d == 0)
    pg1, pg2 = _p_win_half(f1, u1), _p_win_half(f2, u2)
    return [
        ("Hándicap asiático", "Favorito empate no válido (HA 0)", pw / (1 - pp), pp),
        ("Hándicap asiático", "Favorito -0.5 (gana)", pw, 0.0),
        ("Hándicap asiático", "Favorito -1", s(d >= 2) / max(1 - s(d == 1), 1e-9), s(d == 1)),
        ("Hándicap asiático", "No favorito +1", s(d <= 0) / max(1 - s(d == 1), 1e-9), s(d == 1)),
        ("Hándicap asiático", "No favorito +1.5", s(d <= 1), 0.0),
        ("Hándicap asiático", "No favorito +2.5", s(d <= 2), 0.0),
        ("Hándicap 3 vías", "Favorito -1 (gana por 2+)", s(d >= 2), 0.0),
        ("Hándicap 3 vías", "Empate -1 (favorito gana por 1)", s(d == 1), 0.0),
        ("Goles por equipo", "Favorito anota", s(gf >= 1), 0.0),
        ("Goles por equipo", "Favorito más de 1.5", s(gf >= 2), 0.0),
        ("Goles por equipo", "No favorito menos de 1.5", s(gu <= 1), 0.0),
        ("Valla invicta", "Favorito deja valla invicta", s(gu == 0), 0.0),
        ("Valla invicta", "No favorito NO deja valla invicta", s(gf >= 1), 0.0),
        ("Total de goles", "Más de 1.5 goles", s(tot >= 2), 0.0),
        ("Total de goles", "Menos de 3.5 goles", s(tot <= 3), 0.0),
        ("Por tiempo", "1T más de 0.5", _e0(l1), 0.0),
        ("Por tiempo", "1T menos de 1.5", math.exp(-l1) * (1 + l1), 0.0),
        ("Por tiempo", "1T menos de 2.5", float(poisson.cdf(2, l1)), 0.0),
        ("Por tiempo", "2T más de 0.5", _e0(l2), 0.0),
        ("Por tiempo", "Gol en ambos tiempos", _e0(l1) * _e0(l2), 0.0),
        ("Por tiempo", "Ambos anotan 1T: No", 1 - _e0(f1) * _e0(u1), 0.0),
        ("Por tiempo", "Ambos anotan 2T: No", 1 - _e0(f2) * _e0(u2), 0.0),
        ("Por tiempo", "No favorito no gana el 1T", 1 - _p_win_half(u1, f1), 0.0),
        ("Por tiempo", "Favorito gana algún tiempo", 1 - (1 - pg1) * (1 - pg2), 0.0),
        ("Primer gol", "Favorito marca primero", fx / (fx + ux) * _e0(fx + ux), 0.0),
        ("Primer gol", "No favorito no marca primero", 1 - ux / (fx + ux) * _e0(fx + ux), 0.0),
    ]


def goal_market_outcome(mercado: str, hg: int, ag: int, hg1, ag1, fav_local: bool,
                        first: str | None) -> float | None:
    """1 gana, 0 pierde, None devuelve / sin dato. `first` = 'home'/'away'/'none'
    (quién marcó primero) o None si no hay eventos."""
    gf, gu = (hg, ag) if fav_local else (ag, hg)
    d, tot = gf - gu, hg + ag
    ht = pd.notna(hg1) and pd.notna(ag1)
    if ht:
        f1, u1 = (int(hg1), int(ag1)) if fav_local else (int(ag1), int(hg1))
        f2, u2 = gf - f1, gu - u1
    sin_ht = mercado.startswith(("1T", "2T", "Gol en ambos", "Ambos anotan", "No favorito no gana el 1T",
                                 "Favorito gana algún"))
    if sin_ht and not ht:
        return None
    lado_fav = "home" if fav_local else "away"
    r = {
        "Favorito empate no válido (HA 0)": None if d == 0 else d > 0,
        "Favorito -0.5 (gana)": d > 0,
        "Favorito -1": None if d == 1 else d >= 2,
        "No favorito +1": None if d == 1 else d <= 0,
        "No favorito +1.5": d <= 1,
        "No favorito +2.5": d <= 2,
        "Favorito -1 (gana por 2+)": d >= 2,
        "Empate -1 (favorito gana por 1)": d == 1,
        "Favorito anota": gf >= 1,
        "Favorito más de 1.5": gf >= 2,
        "No favorito menos de 1.5": gu <= 1,
        "Favorito deja valla invicta": gu == 0,
        "No favorito NO deja valla invicta": gf >= 1,
        "Más de 1.5 goles": tot >= 2,
        "Menos de 3.5 goles": tot <= 3,
        "Favorito marca primero": None if first is None else first == lado_fav,
        "No favorito no marca primero": None if first is None else first != ("away" if fav_local else "home"),
    }
    if ht:
        h1, h2 = f1 + u1, f2 + u2
        r.update({
            "1T más de 0.5": h1 >= 1, "1T menos de 1.5": h1 <= 1, "1T menos de 2.5": h1 <= 2,
            "2T más de 0.5": h2 >= 1, "Gol en ambos tiempos": h1 >= 1 and h2 >= 1,
            "Ambos anotan 1T: No": not (f1 >= 1 and u1 >= 1), "Ambos anotan 2T: No": not (f2 >= 1 and u2 >= 1),
            "No favorito no gana el 1T": u1 <= f1, "Favorito gana algún tiempo": f1 > u1 or f2 > u2,
        })
    v = r.get(mercado)
    return None if v is None else float(v)


# --- Backtest walk-forward ----------------------------------------------------

def walk_forward_xg(codes: list[str] | None = None) -> pd.DataFrame:
    """Goles esperados del modelo para cada partido jugado de la temporada en
    curso y la anterior, calculados SOLO con partidos previos a ese día."""
    from .cross_competition_strength import fill_missing_with_domestic_strength
    from .poisson_model import expected_goals
    from .team_strength import _load_season_file, build_team_strength

    filas = []
    for code in codes or list(config.COMPETITIONS):
        temporadas = sorted({int(p.stem.rsplit("_", 1)[1]) for p in config.PROCESSED_DIR.glob(f"matches_{code}_*.csv")})
        for season in temporadas[-2:]:
            frames = []
            for s in (season, season - 1):
                try:
                    frames.append(_load_season_file(code, s))
                except Exception:
                    pass
            todos = pd.concat(frames, ignore_index=True)
            if "match_id" in todos.columns:
                todos = todos.drop_duplicates("match_id")
            todos = todos[todos["status"] == "FINISHED"].dropna(subset=["home_goals", "away_goals"]).copy()
            todos["home_goals"] = todos["home_goals"].astype(int)
            todos["away_goals"] = todos["away_goals"].astype(int)
            todos["_inicio"] = pd.to_datetime(todos["utc_date"], utc=True, errors="coerce")
            jugados = todos[todos["season_start_year"] == season]
            for dia, del_dia in jugados.groupby(jugados["_inicio"].dt.floor("D")):
                previos = todos[todos["_inicio"] < dia]
                if len(previos) < 30:
                    continue
                try:
                    fuerza = build_team_strength(previos, as_of=dia)
                except Exception:
                    continue
                for _, m in del_dia.iterrows():
                    f = fuerza
                    if code in config.CUP_STYLE_COMPETITIONS and not {m["home_team"], m["away_team"]} <= set(f.index):
                        try:
                            f, _ = fill_missing_with_domestic_strength(
                                f, {m["home_team"], m["away_team"]}, [season, season - 1], before=dia)
                        except Exception:
                            pass
                    try:
                        hx, ax = expected_goals(f, m["home_team"], m["away_team"])
                    except KeyError:
                        continue
                    previos_de = lambda t: int(f.loc[t, "home_played"] + f.loc[t, "away_played"])  # noqa: E731
                    filas.append({
                        "competition": code, "season": season, "utc_date": m["utc_date"],
                        "home_team": m["home_team"], "away_team": m["away_team"],
                        "home_xg": hx, "away_xg": ax,
                        "min_prev": min(previos_de(m["home_team"]), previos_de(m["away_team"])),
                        "hg": int(m["home_goals"]), "ag": int(m["away_goals"]),
                        "hg1": m.get("home_ht_goals"), "ag1": m.get("away_ht_goals"),
                    })
    return pd.DataFrame(filas)


def first_half_share(wf: pd.DataFrame) -> dict[str, float]:
    d = wf.dropna(subset=["hg1", "ag1"])
    tot = d.groupby("competition")[["hg", "ag", "hg1", "ag1"]].sum()
    return ((tot["hg1"] + tot["ag1"]) / (tot["hg"] + tot["ag"]).clip(lower=1)).to_dict()


def link_365(wf: pd.DataFrame) -> pd.DataFrame:
    """Agrega a cada partido el game_id de 365scores (hora + nombres + marcador)."""
    from . import events365
    from .odds_api import find_event

    games = events365.load_games()
    wf = wf.copy()
    wf["game_id"] = pd.NA
    if games.empty:
        return wf
    games["_dia"] = games["start_utc"].astype(str).str[:10]
    por_dia = {k: g.to_dict("records") for k, g in games.groupby(["competition", "_dia"])}
    ids = []
    for r in wf.itertuples(index=False):
        dia = str(r.utc_date)[:10]
        cand = []
        for delta in (-1, 0, 1):
            k = (pd.Timestamp(dia) + pd.Timedelta(days=delta)).strftime("%Y-%m-%d")
            cand += por_dia.get((r.competition, k), [])
        cand = [dict(c, commence_time=c["start_utc"]) for c in cand
                if c["home_goals"] == r.hg and c["away_goals"] == r.ag]
        ev = find_event(cand, r.home_team, r.away_team, r.utc_date) if cand else None
        ids.append(int(ev["game_id"]) if ev else pd.NA)
    wf["game_id"] = ids
    return wf


def first_scorer(events: pd.DataFrame) -> dict[int, str]:
    """{game_id: 'home'|'away'} del primer gol (en contra cuenta para el rival,
    como lo informa 365: el gol se asigna al equipo que suma)."""
    goles = events[events["type"] == "Goal"].copy()
    goles["_t"] = pd.to_numeric(goles["minute"], errors="coerce") + pd.to_numeric(goles["added_time"],
                                                                                  errors="coerce").fillna(0) / 100
    primero = goles.sort_values("_t").drop_duplicates("game_id")
    return dict(zip(primero["game_id"].astype(int), primero["side"]))


def build_backtest(wf: pd.DataFrame | None = None) -> pd.DataFrame:
    """Un renglón por partido × mercado de goles (y primer gol cuando hay eventos)."""
    from . import events365

    wf = walk_forward_xg() if wf is None else wf
    wf = link_365(wf[wf["min_prev"] >= MIN_PREV_MATCHES])
    share = first_half_share(wf)
    ev = events365.load_events()
    games = events365.load_games()
    primero = first_scorer(ev) if not ev.empty else {}
    sin_goles = set(games.loc[(games["home_goals"] == 0) & (games["away_goals"] == 0), "game_id"].astype(int))
    filas = []
    for r in wf.itertuples(index=False):
        fav_local = r.home_xg >= r.away_xg
        first = None
        if pd.notna(r.game_id):
            gid = int(r.game_id)
            first = primero.get(gid) or ("none" if gid in sin_goles else None)
        for familia, mercado, prob, p_dev in goal_market_probs(Match(r.home_xg, r.away_xg,
                                                                      share.get(r.competition, DEFAULT_FIRST_HALF_SHARE))):
            gano = goal_market_outcome(mercado, r.hg, r.ag, r.hg1, r.ag1, fav_local, first)
            if gano is None and p_dev == 0:
                continue  # sin dato (sin descanso o sin eventos)
            filas.append({
                "competition": r.competition, "season": r.season, "utc_date": r.utc_date,
                "home_team": r.home_team, "away_team": r.away_team,
                "favorito": r.home_team if fav_local else r.away_team,
                "no_favorito": r.away_team if fav_local else r.home_team,
                "game_id": r.game_id, "familia": familia, "mercado": mercado,
                "prob": round(prob, 4), "p_devuelve": round(p_dev, 4), "gano": gano,
                "marcador": f"{r.hg}-{r.ag}",
            })
    return pd.DataFrame(filas)


# --- Cuotas BWIN de cada mercado (365scores) ----------------------------------

def _fmt_line(v: float) -> str:
    return f"{v:g}"


def bwin_price(mercado: str, fav_local: bool, lineas: pd.DataFrame) -> tuple[float, float] | None:
    """(cuota, hándicap del lado apostado) de BWIN para ese mercado, o None si
    BWIN no lo ofrece. `lineas`: special_odds de UN partido (o las líneas en vivo
    con las mismas columnas). El hándicap solo importa para el asiático."""
    fav, und = ("Home", "Away") if fav_local else ("Away", "Home")
    fav_1x2 = "1" if fav_local else "2"

    def op(tipo, opcion, linea=None):
        d = lineas[(lineas["line_type_id"] == tipo) & (lineas["option"] == opcion)]
        if linea is not None:
            d = d[pd.to_numeric(d["line"], errors="coerce").round(2) == round(linea, 2)]
        return float(d["odds"].iloc[0]) if not d.empty and pd.notna(d["odds"].iloc[0]) else None

    def ah(lado_local: bool, linea_lado: float):
        # 365 publica el hándicap del LOCAL; el visitante tiene el opuesto.
        linea_local = linea_lado if lado_local else -linea_lado
        c = op(11, "Home" if lado_local else "Away", linea_local)
        return (c, linea_lado) if c else None

    simples = {
        "Favorito empate no válido (HA 0)": (15, fav, None),
        "Favorito -0.5 (gana)": (1, fav_1x2, None),
        "Favorito deja valla invicta": (144 if fav_local else 145, "Yes", None),
        "No favorito NO deja valla invicta": (145 if fav_local else 144, "No", None),
        "Más de 1.5 goles": (3, "Over", 1.5),
        "Menos de 3.5 goles": (3, "Under", 3.5),
        "1T más de 0.5": (9, "Over", 0.5),
        "1T menos de 1.5": (9, "Under", 1.5),
        "1T menos de 2.5": (9, "Under", 2.5),
        "Ambos anotan 2T: No": (13, "No", None),
        "Favorito marca primero": (7, fav, None),
    }
    if mercado in simples:
        tipo, opcion, linea = simples[mercado]
        c = op(tipo, opcion, linea)
        return (c, 0.0) if c else None
    handicaps = {"No favorito +1.5": (False, 1.5), "No favorito +1": (False, 1.0), "No favorito +2.5": (False, 2.5),
                 "Favorito -1": (True, -1.0)}
    if mercado in handicaps:
        es_fav, linea = handicaps[mercado]
        return ah(fav_local if es_fav else not fav_local, linea)
    return None


def settle_return(mercado: str, gano: float | None, odds: float, margin_fav: int | None = None,
                  linea: float = 0.0) -> float:
    """Ganancia neta por unidad (devolución = 0)."""
    if gano is None or pd.isna(gano):
        return 0.0
    return odds - 1 if gano else -1.0


def with_bwin_odds(bt: pd.DataFrame) -> pd.DataFrame:
    """Agrega la cuota BWIN guardada (cuando hay) y la ganancia de apostar 1."""
    from . import events365

    odds = events365.load_odds()
    if odds.empty or bt.empty:
        return bt.assign(cuota=np.nan, ganancia=np.nan)
    por_juego = {int(k): g for k, g in odds.groupby("game_id")}
    cuotas, ganancias = [], []
    for r in bt.itertuples(index=False):
        lineas = por_juego.get(int(r.game_id)) if pd.notna(r.game_id) else None
        precio = None
        if lineas is not None:
            precio = bwin_price(r.mercado, r.favorito == r.home_team, lineas)
            if precio is None:
                otra = event_or_stat_price(r.mercado, lineas)
                precio = (otra, 0.0) if otra else None
        if not precio:
            cuotas.append(np.nan); ganancias.append(np.nan)
            continue
        cuota, _ = precio
        cuotas.append(cuota)
        ganancias.append(settle_return(r.mercado, r.gano, cuota))
    return bt.assign(cuota=cuotas, ganancia=ganancias)


def summarize(bt: pd.DataFrame, min_prob: float = MIN_PROB, by: list[str] | None = None) -> pd.DataFrame:
    """Por mercado (y `by`): frecuencia base, picks del modelo ≥ min_prob, su
    acierto real vs esperado, cuota justa y — cuando hay cuotas BWIN — la cuota
    media y la ganancia por apuesta."""
    claves = (by or []) + ["familia", "mercado"]
    filas = []
    for k, g in bt.groupby(claves):
        k = k if isinstance(k, tuple) else (k,)
        base = g["gano"].mean()
        p = g[g["prob"] >= min_prob]
        r = p.dropna(subset=["gano"])
        n, aciertos = len(r), r["gano"].sum()
        # probabilidad esperada SIN devoluciones (prob ya está condicionada a que no se devuelva)
        con_cuota = p.dropna(subset=["cuota"]) if "cuota" in p.columns else p.iloc[0:0]
        filas.append(dict(zip(claves, k)) | {
            "partidos": len(g), "base_pct": 100 * base if len(g) else np.nan,
            "picks": len(p), "acierto_pct": 100 * aciertos / n if n else np.nan,
            "esperado_pct": 100 * r["prob"].mean() if n else np.nan,
            "devueltos": int(p["gano"].isna().sum()),
            "piso_pct": 100 * wilson_lower(aciertos, n) if n else np.nan,
            "cuota_justa": n / aciertos if aciertos else np.nan,
            "con_cuota": len(con_cuota),
            "cuota_bwin": con_cuota["cuota"].mean() if len(con_cuota) else np.nan,
            "roi_pct": 100 * con_cuota["ganancia"].mean() if len(con_cuota) else np.nan,
        })
    return pd.DataFrame(filas)


def wilson_lower(k: float, n: int, z: float = 1.64) -> float:
    """Piso del acierto con ~90% de confianza (intervalo de Wilson)."""
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d


def verdict(row) -> str:
    """Clasificación de un mercado para la vista."""
    if not row["picks"] or pd.isna(row["acierto_pct"]):
        return "Sin picks"
    lift = row["acierto_pct"] - row["base_pct"]
    if row["piso_pct"] >= MIN_PROB * 100 and lift >= 5:
        return "Candidato"
    if row["acierto_pct"] >= MIN_PROB * 100 and lift < 3:
        return "Alto pero obvio"
    if row["acierto_pct"] >= MIN_PROB * 100:
        return "Prometedor"
    return "Descartado"


# --- Mercados de evento (roja, penal) y de estadísticas (tiros al arco, fueras de juego) ---

EVENT_MARKETS = {
    # mercado: (función que dice si pasó, tipo de línea BWIN, opción BWIN si pasó / si no)
    "Tarjeta roja en el partido": "roja",
    "Sin tarjeta roja": "roja",
    "Penal en el partido": "penal",
    "Sin penal": "penal",
}
STAT_LINES = {"sot": ("Tiros al arco", 139, [7.5, 8.5, 9.5, 10.5]),
              "off": ("Fueras de juego", 140, [2.5, 3.5, 4.5])}
MIN_LEAGUE_PRIOR = 30  # partidos previos de la liga para estimar su frecuencia


def event_backtest(games_index: pd.DataFrame) -> pd.DataFrame:
    """Roja / penal: la probabilidad es la frecuencia de la liga en los
    partidos ANTERIORES (no hay un modelo por equipo que sirva con tan pocos
    casos). `games_index`: partidos con game_id (de build_backtest)."""
    from . import events365

    ev = events365.load_events()
    if ev.empty or games_index.empty:
        return pd.DataFrame()
    rojas = set(ev.loc[ev["type"] == "Red Card", "game_id"].astype(int))
    penales = set(ev.loc[(ev["subtype"] == "Penalty") | (ev["type"] == "Penalty Miss"), "game_id"].astype(int))
    g = games_index.dropna(subset=["game_id"]).drop_duplicates("game_id").copy()
    g["game_id"] = g["game_id"].astype(int)
    g = g[g["game_id"].isin(set(ev["game_id"].astype(int)))]  # con eventos descargados
    g["_t"] = pd.to_datetime(g["utc_date"], utc=True, errors="coerce")
    g["roja"] = g["game_id"].isin(rojas).astype(float)
    g["penal"] = g["game_id"].isin(penales).astype(float)
    filas = []
    for comp, liga in g.sort_values("_t").groupby("competition"):
        for col in ("roja", "penal"):
            previa = liga[col].shift().expanding().mean()
            cuenta = liga[col].shift().expanding().count()
            for r, p, n in zip(liga.itertuples(index=False), previa, cuenta):
                if n < MIN_LEAGUE_PRIOR or pd.isna(p):
                    continue
                si, no = ("Tarjeta roja en el partido", "Sin tarjeta roja") if col == "roja" else \
                    ("Penal en el partido", "Sin penal")
                paso = getattr(r, col)
                for mercado, prob, gano in ((si, p, paso), (no, 1 - p, 1 - paso)):
                    filas.append({
                        "competition": comp, "season": r.season, "utc_date": r.utc_date,
                        "home_team": r.home_team, "away_team": r.away_team, "favorito": r.favorito,
                        "no_favorito": r.no_favorito, "game_id": r.game_id, "familia": "Eventos",
                        "mercado": mercado, "prob": round(float(prob), 4), "p_devuelve": 0.0, "gano": float(gano),
                        "marcador": r.marcador,
                    })
    return pd.DataFrame(filas)


def stats_backtest(games_index: pd.DataFrame) -> pd.DataFrame:
    """Tiros al arco y fueras de juego: mismo modelo que corners/tarjetas
    (totals_model: factores por equipo + binomial negativa), walk-forward."""
    from . import match_stats_log
    from . import totals_model as tm

    log = match_stats_log.load_log()
    if log.empty:
        return pd.DataFrame()
    log = log.assign(sot=pd.to_numeric(log["remates_a_puerta"], errors="coerce"),
                     off=pd.to_numeric(log["fueras_de_juego"], errors="coerce"))
    es_local = log["is_home"].astype(str).str.lower().isin(["true", "1"])
    k = ["competition", "utc_date", "home_team", "away_team"]
    df = (log[es_local].drop_duplicates(k).set_index(k)[["sot", "off"]]
          .join(log[~es_local].drop_duplicates(k).set_index(k)[["sot", "off"]], lsuffix="_home", rsuffix="_away",
                how="inner").reset_index())
    df["inicio"] = pd.to_datetime(df["utc_date"], utc=True, errors="coerce")
    idx = games_index.drop_duplicates(["competition", "home_team", "away_team", "utc_date"])
    df = df.merge(idx[["competition", "utc_date", "home_team", "away_team", "season", "favorito", "no_favorito",
                       "game_id", "marcador"]], on=k, how="inner")
    filas = []
    for comp, liga in df.sort_values("inicio").groupby("competition"):
        liga = liga.reset_index(drop=True)
        for i in range(len(liga)):
            m = liga.iloc[i]
            previos = liga.iloc[:i]
            for col, (nombre, _, lineas) in STAT_LINES.items():
                if pd.isna(m[f"{col}_home"]) or pd.isna(m[f"{col}_away"]):
                    continue
                s = tm.fit(previos, col)
                if s is None:
                    continue
                mu = tm.expected_total(s, m["home_team"], m["away_team"])
                real = m[f"{col}_home"] + m[f"{col}_away"]
                for linea in lineas:
                    p_mas = 0.5 + tm.CALIBRATION_SLOPE * (tm.prob_over(s, mu, linea) - 0.5)
                    for lado, prob, gano in (("más", p_mas, real > linea), ("menos", 1 - p_mas, real < linea)):
                        filas.append({
                            "competition": comp, "season": m["season"], "utc_date": m["utc_date"],
                            "home_team": m["home_team"], "away_team": m["away_team"], "favorito": m["favorito"],
                            "no_favorito": m["no_favorito"], "game_id": m["game_id"], "familia": "Estadísticas",
                            "mercado": f"{nombre}: {lado} de {linea:g}", "prob": round(float(prob), 4),
                            "p_devuelve": 0.0, "gano": float(gano), "marcador": m["marcador"],
                        })
    return pd.DataFrame(filas)


def event_or_stat_price(mercado: str, lineas: pd.DataFrame) -> float | None:
    def op(tipo, opcion, linea=None):
        d = lineas[(lineas["line_type_id"] == tipo) & (lineas["option"] == opcion)]
        if linea is not None:
            d = d[pd.to_numeric(d["line"], errors="coerce").round(2) == round(linea, 2)]
        return float(d["odds"].iloc[0]) if not d.empty and pd.notna(d["odds"].iloc[0]) else None

    fijos = {"Tarjeta roja en el partido": (142, "Yes"), "Sin tarjeta roja": (142, "No"),
             "Penal en el partido": (143, "Yes"), "Sin penal": (143, "No")}
    if mercado in fijos:
        return op(*fijos[mercado])
    for nombre, tipo, _ in STAT_LINES.values():
        if mercado.startswith(nombre + ":"):
            lado, linea = mercado.split(": ")[1].split(" de ")
            return op(tipo, "Over" if lado == "más" else "Under", float(linea))
    return None


# --- Patrones por equipo --------------------------------------------------------

def team_patterns(bt: pd.DataFrame, min_prob: float = MIN_PROB, min_picks: int = 8) -> pd.DataFrame:
    """Por equipo (como favorito o no favorito del pick) y mercado: acierto en la
    primera y en la segunda mitad de sus picks. Un patrón real aguanta en las
    dos; uno de azar suele caerse en la segunda (la segunda mitad es la prueba)."""
    p = bt[(bt["prob"] >= min_prob)].dropna(subset=["gano"]).copy()
    if p.empty:
        return pd.DataFrame()
    p["_t"] = pd.to_datetime(p["utc_date"], utc=True, errors="coerce")
    largo = pd.concat([p.assign(equipo=p["favorito"], rol="favorito"),
                       p.assign(equipo=p["no_favorito"], rol="no favorito")], ignore_index=True)
    filas = []
    for (comp, equipo, fam, mercado), g in largo.groupby(["competition", "equipo", "familia", "mercado"]):
        if len(g) < min_picks:
            continue
        g = g.sort_values("_t")
        mitad = len(g) // 2
        a, b = g.iloc[:mitad], g.iloc[mitad:]
        filas.append({"competition": comp, "equipo": equipo, "familia": fam, "mercado": mercado, "picks": len(g),
                      "acierto_pct": 100 * g["gano"].mean(), "esperado_pct": 100 * g["prob"].mean(),
                      "primera_mitad_pct": 100 * a["gano"].mean(), "segunda_mitad_pct": 100 * b["gano"].mean(),
                      "piso_pct": 100 * wilson_lower(g["gano"].sum(), len(g))})
    return pd.DataFrame(filas)


# --- Próximos partidos -----------------------------------------------------------

def upcoming_picks(preds: pd.DataFrame, mercados: list[str], share: dict[str, float] | None = None,
                   min_prob: float = MIN_PROB) -> pd.DataFrame:
    """Picks de los mercados `mercados` para partidos por jugar. `preds`:
    predicciones del día (columnas competition, home_team, away_team, home_xg,
    away_xg, utc_date)."""
    share = share or {}
    filas = []
    for r in preds.itertuples(index=False):
        fav_local = r.home_xg >= r.away_xg
        for familia, mercado, prob, p_dev in goal_market_probs(
                Match(float(r.home_xg), float(r.away_xg), share.get(r.competition, DEFAULT_FIRST_HALF_SHARE))):
            if mercado in mercados and prob >= min_prob:
                filas.append({"competition": r.competition, "utc_date": r.utc_date, "home_team": r.home_team,
                              "away_team": r.away_team, "favorito": r.home_team if fav_local else r.away_team,
                              "fav_local": fav_local, "familia": familia, "mercado": mercado,
                              "prob": prob, "p_devuelve": p_dev})
    return pd.DataFrame(filas)


def load_backtest() -> pd.DataFrame:
    if not BACKTEST_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(BACKTEST_PATH)


def main() -> int:
    t0 = pd.Timestamp.now()
    wf = walk_forward_xg()
    goles = build_backtest(wf)
    indice = goles.drop_duplicates(["competition", "home_team", "away_team", "utc_date"])
    partes = [goles, event_backtest(indice), stats_backtest(indice)]
    bt = with_bwin_odds(pd.concat([p for p in partes if not p.empty], ignore_index=True))
    BACKTEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(BACKTEST_PATH, index=False, compression={"method": "gzip", "mtime": 0})
    share = first_half_share(wf)
    pd.Series(share, name="first_half_share").to_csv(SHARE_PATH)
    print(f"{len(bt)} renglones ({bt['mercado'].nunique()} mercados) en "
          f"{(pd.Timestamp.now() - t0).total_seconds():.0f}s -> {BACKTEST_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def bwin_lines_for(code: str, home: str, away: str, utc_date: str) -> pd.DataFrame | None:
    """Líneas actuales de BWIN (365scores) de un partido por jugar, con las
    columnas de special_odds; None si no se encuentra el partido."""
    from . import events365, scores365
    from .odds_api import find_event

    inicio = pd.Timestamp(utc_date)
    desde = (inicio - pd.Timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
    hasta = (inicio + pd.Timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
    juegos = scores365.games(code, desde, hasta)
    eventos = [{"game_id": g["id"], "home_team": (g.get("homeCompetitor") or {}).get("name"),
                "away_team": (g.get("awayCompetitor") or {}).get("name"),
                "commence_time": scores365._utc(g.get("startTime"))} for g in juegos]
    ev = find_event(eventos, home, away, utc_date) if eventos else None
    if not ev:
        return None
    filas = events365.line_rows(int(ev["game_id"]), scores365.game_lines(int(ev["game_id"])))
    return pd.DataFrame(filas) if filas else None


# --- Leyenda de cada mercado (vista Avanzada y Tips) ---------------------------
# "Favorito" = el equipo con más goles esperados por el modelo en ESE partido
# (no siempre el local); "no favorito" = el otro.
MARKET_HELP = {
    "Favorito empate no válido (HA 0)":
        "Se apuesta a que gana el favorito; si empatan, se devuelve la plata (no se pierde). "
        "En BWIN: 'Empate no válido' (Draw No Bet).",
    "Favorito -0.5 (gana)": "El favorito gana el partido (igual que el 1 o el 2 del 1X2).",
    "Favorito -1":
        "Hándicap asiático -1 al favorito: gana si gana por 2 o más; si gana por 1 exacto se devuelve; "
        "empate o derrota pierde.",
    "No favorito +1":
        "Hándicap asiático +1 al no favorito: gana si el no favorito gana o empata; si pierde por 1 se "
        "devuelve; si pierde por 2 o más, pierde.",
    "No favorito +1.5":
        "Hándicap +1.5 al no favorito: gana si NO pierde por 2 o más (gana, empata o pierde por 1). "
        "Es lo mismo que el hándicap de 3 vías 'no favorito +1'.",
    "No favorito +2.5": "Gana si el no favorito no pierde por 3 o más goles. Sale casi siempre: paga muy poco.",
    "Favorito -1 (gana por 2+)": "Hándicap de 3 vías: el favorito gana por 2 goles o más.",
    "Empate -1 (favorito gana por 1)": "Hándicap de 3 vías, opción 'empate': el favorito gana por exactamente 1 gol.",
    "Favorito anota": "El favorito marca al menos un gol.",
    "Favorito más de 1.5": "El favorito marca 2 goles o más.",
    "No favorito menos de 1.5": "El no favorito marca 0 o 1 gol.",
    "Favorito deja valla invicta": "El favorito no recibe goles.",
    "No favorito NO deja valla invicta":
        "El no favorito recibe al menos un gol (el favorito anota). En BWIN: 'Valla invicta' del no favorito → No.",
    "Más de 1.5 goles": "Hay 2 goles o más en el partido.",
    "Menos de 3.5 goles": "Hay 3 goles o menos en el partido.",
    "1T más de 0.5": "Hay al menos un gol en el primer tiempo.",
    "1T menos de 1.5": "En el primer tiempo hay 0 o 1 gol.",
    "1T menos de 2.5": "En el primer tiempo hay 2 goles o menos.",
    "2T más de 0.5": "Hay al menos un gol en el segundo tiempo.",
    "Gol en ambos tiempos": "Hay al menos un gol en el primer tiempo Y al menos uno en el segundo.",
    "Ambos anotan 1T: No": "En el primer tiempo NO marcan los dos equipos (puede marcar uno solo o ninguno).",
    "Ambos anotan 2T: No": "En el segundo tiempo NO marcan los dos equipos.",
    "No favorito no gana el 1T": "Al descanso el no favorito no va ganando (empate o gana el favorito).",
    "Favorito gana algún tiempo":
        "El favorito marca más que el rival en el primer tiempo, en el segundo, o en ambos "
        "(cada tiempo se cuenta por separado).",
    "Favorito marca primero": "El primer gol del partido es del favorito (sin goles = pierde). En BWIN: 'Primer gol'.",
    "No favorito no marca primero": "El primer gol NO es del no favorito: lo marca el favorito o no hay goles.",
    "Tarjeta roja en el partido": "Al menos una tarjeta roja (incluye doble amarilla) en el partido.",
    "Sin tarjeta roja": "Ninguna tarjeta roja en el partido.",
    "Penal en el partido": "Se cobra al menos un penal (se haya convertido o no).",
    "Sin penal": "No se cobra ningún penal en el partido.",
}
STAT_HELP = {
    "Tiros al arco": "Tiros al arco (a puerta) de los dos equipos sumados, más o menos que la línea.",
    "Fueras de juego": "Fueras de juego de los dos equipos sumados, más o menos que la línea.",
}


def market_help(mercado: str) -> str:
    if mercado in MARKET_HELP:
        return MARKET_HELP[mercado]
    for nombre, texto in STAT_HELP.items():
        if mercado.startswith(nombre + ":"):
            return texto
    return ""
