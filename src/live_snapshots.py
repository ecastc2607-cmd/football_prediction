"""
Fotos de partidos en vivo, versionadas en git (pedido explícito del usuario),
para poder calibrar algún día el ajuste en vivo con datos propios.

El ajuste (live_matches.adjusted_live_probabilities) usa pesos sacados de la
literatura — xG por remate, efecto marcador — que nunca se validaron con
partidos del proyecto, porque no había histórico en vivo. Cada vez que la
pestaña "En vivo" muestra un partido, se guarda como mucho UNA foto por cada
BUCKET_MINUTES de juego: marcador, remates, base pre-partido y el 1X2 de cada
método. Cuando el partido termina, evaluate() lo cruza con el resultado final
(los CSV de partidos) y compara los métodos por log-loss.

Solo se guardan fotos mientras alguien mira la pestaña: no hay un proceso que
las tome solo. En Streamlit Cloud el archivo se pierde en cada reinicio; las
que valen son las de corridas locales que se suban a git.

Uso:
    python -m src.live_snapshots
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

SNAPSHOT_PATH = config.ROOT_DIR / "data" / "tracking" / "live_snapshots.csv"
BUCKET_MINUTES = 10
KEY = ["competition", "home_team", "away_team", "utc_date", "minute_bucket"]

# (nombre, prefijo de columnas) de los métodos que se comparan.
METHODS = [
    ("Solo pre-partido (anterior)", "pre"),
    ("Ajustado en vivo (actual)", "adj"),
]


def record(snapshot: dict) -> bool:
    """Guarda la foto si no hay otra de ese partido en el mismo tramo de
    BUCKET_MINUTES. Devuelve True si la guardó. Nunca lanza: guardar es
    secundario frente a mostrar el partido."""
    try:
        fila = dict(snapshot, minute_bucket=int(snapshot["minute"]) // BUCKET_MINUTES)
        if SNAPSHOT_PATH.exists():
            hist = pd.read_csv(SNAPSHOT_PATH)
            ya = set(map(tuple, hist[KEY].astype(str).itertuples(index=False)))
            if tuple(str(fila[k]) for k in KEY) in ya:
                return False
            hist = pd.concat([hist, pd.DataFrame([fila])], ignore_index=True)
        else:
            SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            hist = pd.DataFrame([fila])
        hist.to_csv(SNAPSHOT_PATH, index=False, encoding="utf-8")
        return True
    except Exception:
        return False


def _final_scores() -> dict:
    """{(competición, local, visitante, inicio_al_minuto): (goles_l, goles_v)}
    de todos los CSV de partidos terminados que haya en disco."""
    finales = {}
    for path in config.PROCESSED_DIR.glob("matches_*_*.csv"):
        try:
            df = pd.read_csv(path, usecols=["competition", "home_team", "away_team", "utc_date",
                                            "status", "home_goals", "away_goals"])
        except (ValueError, pd.errors.EmptyDataError):
            continue
        df = df[df["status"] == "FINISHED"].dropna(subset=["home_goals", "away_goals"])
        for r in df.itertuples():
            finales[(r.competition, r.home_team, r.away_team, str(r.utc_date)[:16])] = (
                int(r.home_goals), int(r.away_goals))
    return finales


def evaluate() -> pd.DataFrame:
    """Log-loss (más bajo = mejor) del 1X2 de cada método sobre las fotos de
    partidos ya terminados, en total y por tramo del partido."""
    if not SNAPSHOT_PATH.exists():
        return pd.DataFrame()
    fotos = pd.read_csv(SNAPSHOT_PATH)
    finales = _final_scores()
    resultado = []
    for r in fotos.itertuples():
        final = finales.get((r.competition, r.home_team, r.away_team, str(r.utc_date)[:16]))
        if final is None:
            continue
        real = "home" if final[0] > final[1] else "away" if final[0] < final[1] else "draw"
        tramo = "1º tiempo" if r.minute <= 45 else "2º tiempo"
        for nombre, pref in METHODS:
            p = getattr(r, f"{pref}_{real}", None)
            if p is None or p != p:
                continue
            resultado.append({"Método": nombre, "Tramo": tramo, "logloss": -np.log(max(float(p), 1e-6))})
    if not resultado:
        return pd.DataFrame()
    df = pd.DataFrame(resultado)
    total = df.groupby("Método")["logloss"].agg(["mean", "count"]).assign(Tramo="Todo")
    por_tramo = df.groupby(["Método", "Tramo"])["logloss"].agg(["mean", "count"]).reset_index("Tramo")
    tabla = pd.concat([total, por_tramo]).reset_index()
    return tabla.rename(columns={"mean": "Log-loss", "count": "Fotos"})[["Método", "Tramo", "Log-loss", "Fotos"]]


def main():
    tabla = evaluate()
    if tabla.empty:
        print("Todavía no hay fotos en vivo de partidos ya terminados para evaluar.")
        return
    print(tabla.to_string(index=False, formatters={"Log-loss": "{:.4f}".format}))


if __name__ == "__main__":
    main()
