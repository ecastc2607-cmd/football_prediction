"""
Rellena el histórico de corners, faltas y tarjetas desde Goal API, respetando la
cuota diaria del plan gratis (1.000 peticiones/día).

Recorre los partidos YA TERMINADOS que tenemos en data/processed/*.csv, los busca
en Goal API y guarda las estadísticas en data/tracking/match_stats_log.csv — el
mismo archivo que alimenta los promedios por equipo del modelo.

Es reanudable: cada corrida se salta lo que ya está guardado, así que se puede
correr un día tras otro hasta completar el histórico sin llevar la cuenta a mano.

Costo real: 1 petición por fecha/liga (resuelve la jornada entera) + 1 por partido.
Una temporada completa de las 5 grandes ronda los 1.800 partidos, o sea unos dos
días de cuota gratis.

Uso:
    python scripts/backfill_match_stats.py                    # todo, hasta agotar el tope
    python scripts/backfill_match_stats.py --max-requests 300 # solo 300 peticiones
    python scripts/backfill_match_stats.py --competitions PL PD
    python scripts/backfill_match_stats.py --dry-run          # no gasta cuota: solo cuenta
"""
from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.goal_api_client import GoalApiClient, GoalApiRateLimited  # noqa: E402
from src.match_stats_log import append_rows, build_rows, existing_keys, match_key  # noqa: E402

# Tope por defecto por debajo de las 1.000 diarias: deja aire para que el
# dashboard siga funcionando el mismo día en que corras el backfill.
DEFAULT_MAX_REQUESTS = 850
# Cada cuántos partidos se vuelca lo acumulado al CSV. Si el proceso se corta
# (Ctrl+C, se agota la cuota), no se pierde más que el último lote.
FLUSH_EVERY = 25


def partidos_terminados(competitions: list[str]) -> pd.DataFrame:
    """Todos los partidos FINISHED de los CSV procesados, de todas las temporadas
    que tengamos descargadas."""
    frames = []
    for path in sorted(glob.glob(str(config.PROCESSED_DIR / "matches_*.csv"))):
        df = pd.read_csv(path)
        if df.empty or "status" not in df.columns:
            continue
        df = df[df["status"] == "FINISHED"]
        if competitions:
            df = df[df["competition"].isin(competitions)]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    todos = pd.concat(frames, ignore_index=True)
    # Los más recientes primero: si la cuota se agota a medio camino, lo que
    # queda guardado es lo más relevante para predecir la jornada próxima.
    return todos.sort_values("utc_date", ascending=False).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS,
                        help=f"tope de peticiones a Goal API (por defecto {DEFAULT_MAX_REQUESTS})")
    parser.add_argument("--competitions", nargs="*", default=[],
                        help="códigos a procesar (PL PD SA BL1 FL1 CL). Por defecto, todos.")
    parser.add_argument("--dry-run", action="store_true",
                        help="no llama al API: solo dice cuánto falta y cuánto costaría")
    args = parser.parse_args()

    pendientes = partidos_terminados(args.competitions)
    if pendientes.empty:
        print("No hay partidos terminados en data/processed/. Corre primero el pipeline.")
        return 1

    ya_guardados = existing_keys()
    pendientes = pendientes[[
        match_key(r.competition, r.season_start_year, r.home_team, r.away_team) not in ya_guardados
        for r in pendientes.itertuples()
    ]]

    fechas_liga = {(r.competition, str(r.utc_date)[:10]) for r in pendientes.itertuples()}
    costo_estimado = len(pendientes) + len(fechas_liga)

    print(f"Ya guardados : {len(ya_guardados)} partidos")
    print(f"Pendientes   : {len(pendientes)} partidos en {len(fechas_liga)} fecha(s)/liga")
    print(f"Costo estimado: ~{costo_estimado} peticiones "
          f"({len(fechas_liga)} de calendario + {len(pendientes)} de estadísticas)")
    print(f"Tope de esta corrida: {args.max_requests}")
    if costo_estimado > args.max_requests:
        dias = -(-costo_estimado // args.max_requests)  # división hacia arriba
        print(f"  -> no alcanza en una corrida: hacen falta ~{dias} días de cuota.")
    if args.dry_run:
        print("\n--dry-run: no se llamó al API.")
        return 0
    if pendientes.empty:
        print("\nNada que hacer: el histórico ya está completo.")
        return 0

    client = GoalApiClient()
    peticiones = 0
    nuevos: list[dict] = []
    guardados = sin_datos = sin_match = 0
    cache_fixtures: dict[tuple, list] = {}
    # Agrupar por (liga, fecha) hace que el calendario de cada día se pida una
    # sola vez para todos los partidos de esa jornada.
    por_dia = defaultdict(list)
    for r in pendientes.itertuples():
        por_dia[(r.competition, str(r.utc_date)[:10])].append(r)

    def volcar():
        nonlocal nuevos
        if nuevos:
            append_rows(nuevos)
            nuevos = []

    try:
        for (code, fecha), partidos in por_dia.items():
            if peticiones >= args.max_requests:
                break
            clave = (code, fecha)
            if clave not in cache_fixtures:
                cache_fixtures[clave] = client.fixtures_by_date(fecha, code)
                peticiones += 1
            fixtures = cache_fixtures[clave]

            for r in partidos:
                if peticiones >= args.max_requests:
                    break
                fixture_id = client.find_fixture_id(
                    r.home_team, r.away_team, str(r.utc_date), code, fixtures=fixtures
                )
                if fixture_id is None:
                    sin_match += 1
                    continue

                stats = client.match_statistics(fixture_id, r.home_team, r.away_team)
                peticiones += 1
                if not stats:
                    sin_datos += 1
                    continue

                nuevos.extend(build_rows(
                    code, r.season_start_year, r.matchday, str(r.utc_date),
                    r.home_team, r.away_team, stats,
                ))
                guardados += 1
                if guardados % FLUSH_EVERY == 0:
                    volcar()
                    print(f"  ... {guardados} partidos guardados, {peticiones} peticiones usadas")
    except GoalApiRateLimited as e:
        print(f"\nCuota agotada: {e}")
    except KeyboardInterrupt:
        print("\nInterrumpido a mano.")
    finally:
        volcar()

    print()
    print(f"Guardados en esta corrida : {guardados}")
    print(f"Sin estadísticas todavía  : {sin_datos}")
    print(f"No encontrados en Goal API: {sin_match}")
    print(f"Peticiones usadas         : {peticiones}")
    restantes = len(pendientes) - guardados - sin_datos - sin_match
    if restantes > 0:
        print(f"\nQuedan ~{restantes} partidos. Vuelve a correr el script mañana "
              "(la cuota se reinicia) y sigue donde quedó.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
