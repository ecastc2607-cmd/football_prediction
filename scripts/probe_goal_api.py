"""
Prueba de fuego a Goal API (https://goal-api.com) antes de migrar nada.

Responde las 3 preguntas que NO se pueden contestar leyendo la web del proveedor:
  1. ¿Cubre de verdad Premier/LaLiga/Serie A/Bundesliga/Ligue 1/Champions?
  2. ¿El endpoint de estadísticas trae córners, faltas y tarjetas, y con qué
     nombres exactos de campo? (esto es lo que rompió con Highlightly)
  3. ¿Cuántas peticiones gasta un partido? (para saber si 1.000/día alcanzan)

Uso:
    set GOAL_API_KEY=tu_key
    python scripts/probe_goal_api.py

No escribe nada en el proyecto: solo imprime lo que encuentra.
"""
from __future__ import annotations

import json
import os
import sys

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8")  # la consola de Windows es cp1252
except Exception:
    pass

BASE = "https://api.goal-api.com/v1"
KEY = os.getenv("GOAL_API_KEY", "")
PAGE_LIMIT = 100  # tope duro del API: "Limit must be between 1 and 100"

LIGAS_BUSCADAS = ["Premier League", "La Liga", "LaLiga", "Serie A",
                  "Bundesliga", "Ligue 1", "Champions League"]

llamadas = 0


def get(path: str, **params):
    global llamadas
    llamadas += 1
    r = requests.get(f"{BASE}{path}", params=params or None,
                     headers={"Authorization": f"Bearer {KEY}"}, timeout=20)
    if r.status_code != 200:
        print(f"   [HTTP {r.status_code}] {path} -> {r.text[:250]}")
        return None
    return r.json().get("data")


def todas_las_ligas() -> list[dict]:
    """Pagina /leagues hasta agotar (el API corta en 100 por página)."""
    ligas, offset = [], 0
    while True:
        page = get("/leagues", limit=PAGE_LIMIT, offset=offset)
        if not page:
            break
        ligas.extend(page)
        if len(page) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
        if offset > 1500:  # cortafuegos: la cobertura anunciada es ~1.019
            break
    return ligas


def main() -> int:
    if not KEY:
        print("Falta GOAL_API_KEY. Regístrate gratis (sin tarjeta) en "
              "https://goal-api.com/signup, copia la key del dashboard y:")
        print("    set GOAL_API_KEY=tu_key")
        return 1

    print("1) COBERTURA DE LIGAS")
    ligas = todas_las_ligas()
    print(f"   {len(ligas)} ligas recuperadas en total\n")

    ids_top = {}
    for buscada in LIGAS_BUSCADAS:
        hits = [lg for lg in ligas if buscada.lower() in str(lg.get("name", "")).lower()]
        print(f"   {'OK ' if hits else 'NO '} {buscada}: {len(hits)} coincidencia(s)")
        for lg in hits[:6]:
            pais = lg.get("country") or lg.get("countryName") or "?"
            print(f"        id={lg.get('id')}  {lg.get('name')}  ({pais})")
            ids_top.setdefault(buscada, lg.get("id"))

    print("\n2) ESTRUCTURA REAL DE /fixtures/:id/statistics")
    # Un partido de una liga top ya terminado — no uno cualquiera del mundo,
    # porque la cobertura de stats puede variar entre ligas grandes y menores.
    liga_id = ids_top.get("Premier League") or ids_top.get("La Liga")
    partido = None
    if liga_id:
        resultados = get(f"/results/league/{liga_id}", limit=5) or []
        if isinstance(resultados, list) and resultados:
            partido = resultados[0]
    if partido is None:
        resultados = get("/results", limit=5) or []
        partido = resultados[0] if isinstance(resultados, list) and resultados else None

    if not partido:
        print("   No se pudo obtener un partido terminado.")
    else:
        fid = partido.get("id") or partido.get("fixtureId")
        def nombre(x):
            return x.get("name") if isinstance(x, dict) else x
        print(f"   Partido: {nombre(partido.get('homeTeam'))} vs "
              f"{nombre(partido.get('awayTeam'))}  (id={fid})")

        stats = get(f"/fixtures/{fid}/statistics")
        if stats is None:
            print("   Sin datos de estadísticas para ese partido.")
        else:
            print("\n   --- Volcado completo (para leer los nombres exactos) ---")
            print(json.dumps(stats, indent=1, ensure_ascii=False)[:4000])

            # El sondeo anterior mostró 'Corners' y 'Ball Possession' DOS veces
            # con valores distintos dentro del mismo bloque. Hay que saber si eso
            # es sistemático antes de confiar en el dato.
            print("\n   --- Chequeo de duplicados por bloque ---")
            if isinstance(stats, dict):
                for bloque, filas in stats.items():
                    if not isinstance(filas, list):
                        continue
                    vistos = {}
                    for f in filas:
                        if isinstance(f, dict) and "type" in f:
                            vistos.setdefault(f["type"], []).append(
                                (f.get("home"), f.get("away")))
                    dups = {k: v for k, v in vistos.items() if len(v) > 1}
                    print(f"   bloque '{bloque}': {len(filas)} filas, "
                          f"{len(dups)} tipo(s) duplicado(s)")
                    for k, v in dups.items():
                        print(f"      {k}: {v}")

    print(f"\n3) COSTO EN PETICIONES: esta prueba gastó {llamadas} de tus 1.000/día.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
