"""
Regenera los CSV procesados (data/processed/matches_{codigo}_{temporada}.csv) a partir
del JSON crudo que ya está en data/raw/, sin volver a llamar a la API.

Útil cuando cambia la lógica de aplanado, o para recuperar una temporada que quedó
pisada por una corrida posterior (como pasó al fijar el nombre de archivo por temporada).

Uso:
    python -m src.reprocess_raw
"""
from __future__ import annotations

import json
import re

from . import config
from .fetch_football_data import flatten_matches, save_processed

RAW_FILENAME_RE = re.compile(r"^([A-Z0-9]+)_matches_(\d{4})\.json$")


def main():
    raw_files = sorted(config.RAW_DIR.glob("*_matches_*.json"))
    if not raw_files:
        print("No hay JSON crudos en data/raw/ todavía. Corre primero fetch_football_data.py.")
        return

    for path in raw_files:
        m = RAW_FILENAME_RE.match(path.name)
        if not m:
            continue
        code, season = m.group(1), int(m.group(2))
        raw = json.loads(path.read_text(encoding="utf-8"))
        df = flatten_matches(raw)
        csv_path = save_processed(df, code, season)
        finished = (df["status"] == "FINISHED").sum()
        print(f"{code} {season}: {len(df)} partidos -> {csv_path.relative_to(config.ROOT_DIR)} "
              f"({finished} finalizados)")


if __name__ == "__main__":
    main()
