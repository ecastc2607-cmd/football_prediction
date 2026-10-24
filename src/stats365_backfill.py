"""
Carga única de estadísticas finales de 365scores (tiros al arco, fueras de
juego, remates, posesión y lo básico) de los partidos europeos ya jugados, a
data/tracking/match_stats_log.csv.

Para qué (oct-2026): la vista Jornada muestra tiros al arco y fueras de juego
esperados, y en Europa el log casi no tenía esos campos (los cargaba Goal API
solo con corners/faltas/tarjetas). Los partidos salen de los ya vinculados a
365scores en el backtest de mercados especiales (special_markets), que trae el
game_id con los nombres de equipo del proyecto.

- Partido que ya está en el log sin esos campos: se completan (no se pisa lo que haya).
- Partido que no está: se agrega con la jornada del CSV local.
Después lo mantiene al día la captura automática (capture_final_stats).

Uso:
    python -m src.stats365_backfill
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import config, scores365, special_markets
from .match_stats_log import CORE_FIELDS, EXTRA_FIELDS, LOG_PATH, load_log

CAMPOS = CORE_FIELDS + EXTRA_FIELDS


def _stats(game_id: int) -> dict | None:
    """{"home": {...}, "away": {...}} con UNA consulta (trae los ids de los equipos)."""
    data = scores365._get("/game/stats/", {"games": game_id}) or {}
    juegos = data.get("games") or []
    if not juegos:
        return None
    home_id = (juegos[0].get("homeCompetitor") or {}).get("id")
    away_id = (juegos[0].get("awayCompetitor") or {}).get("id")
    salida: dict[str, dict] = {"home": {}, "away": {}}
    for st in data.get("statistics") or []:
        clave = scores365._LIVE_STATS.get(st.get("name"))
        lado = "home" if st.get("competitorId") == home_id else "away" if st.get("competitorId") == away_id else None
        if not clave or not lado:
            continue
        try:
            salida[lado][clave] = float(str(st.get("value", "")).rstrip("%"))
        except ValueError:
            continue
    if not (salida["home"] or salida["away"]):
        return None
    for lado in salida.values():
        for clave in scores365._ZERO_WHEN_MISSING:
            lado.setdefault(clave, 0.0)
    return salida


def _matchdays() -> dict:
    filas = []
    for p in config.PROCESSED_DIR.glob("matches_*_*.csv"):
        d = pd.read_csv(p, usecols=lambda c: c in {"competition", "home_team", "away_team", "utc_date", "matchday"})
        filas.append(d)
    todo = pd.concat(filas, ignore_index=True).dropna(subset=["matchday"])
    return {(r.competition, r.home_team, r.away_team, str(r.utc_date)[:16]): int(r.matchday)
            for r in todo.itertuples(index=False)}


def main() -> None:
    bt = special_markets.load_backtest()
    partidos = (bt.dropna(subset=["game_id"]).drop_duplicates(["competition", "home_team", "away_team", "utc_date"])
                [["competition", "season", "utc_date", "home_team", "away_team", "game_id"]])
    partidos = partidos[partidos["competition"] != "COL"]  # Colombia ya está completa (col_stats_backfill)
    log = load_log()
    clave = lambda c, u, h: (c, str(u)[:16], h)  # noqa: E731
    con_sot = set()
    if not log.empty:
        sot = pd.to_numeric(log["remates_a_puerta"], errors="coerce").notna()
        con_sot = {clave(c, u, h) for c, u, h in zip(log.loc[sot, "competition"], log.loc[sot, "utc_date"],
                                                      log.loc[sot, "home_team"])}
    pendientes = [r for r in partidos.itertuples(index=False) if clave(r.competition, r.utc_date, r.home_team) not in con_sot]
    print(f"{len(pendientes)} partidos sin tiros al arco / fueras de juego", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        resultados = list(pool.map(lambda r: (r, _stats(int(r.game_id))), pendientes))

    jornadas = _matchdays()
    indice = {}
    if not log.empty:
        for i, (c, u, h, t) in enumerate(zip(log["competition"], log["utc_date"], log["home_team"], log["team"])):
            indice[(clave(c, u, h), t)] = log.index[i]
    nuevas, completadas = [], 0
    for r, stats in resultados:
        if not stats:
            continue
        for lado, equipo in (("home", r.home_team), ("away", r.away_team)):
            datos = stats[lado]
            pos = indice.get((clave(r.competition, r.utc_date, r.home_team), equipo))
            if pos is not None:
                for campo in CAMPOS:
                    if pd.isna(log.at[pos, campo]) and campo in datos:
                        log.at[pos, campo] = datos[campo]
                completadas += 1
            else:
                fila = {"competition": r.competition, "season": int(r.season),
                        "matchday": jornadas.get((r.competition, r.home_team, r.away_team, str(r.utc_date)[:16]), 0),
                        "utc_date": r.utc_date, "home_team": r.home_team, "away_team": r.away_team,
                        "team": equipo, "is_home": lado == "home"}
                fila.update({c: datos.get(c) for c in CAMPOS})
                nuevas.append(fila)
    salida = pd.concat([log, pd.DataFrame(nuevas)], ignore_index=True) if nuevas else log
    salida[log.columns if not log.empty else salida.columns].to_csv(LOG_PATH, index=False)
    print(f"Listo: {completadas} filas completadas, {len(nuevas)} filas nuevas "
          f"({len(nuevas) // 2} partidos).")


if __name__ == "__main__":
    main()
