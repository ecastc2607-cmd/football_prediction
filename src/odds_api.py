"""
Cuotas reales de casas de apuestas vía The Odds API (https://the-odds-api.com).

Por qué esta fuente y no otra (verificado, no supuesto):
  - Las casas legales en Colombia (BetPlay, Wplay, Rushbet, Betsson, Stake,
    Sportium, bwin...) no tienen API pública; leer sus páginas por debajo va
    contra sus términos y se rompe sin aviso.
  - Goal API sí trae cuotas, pero solo en su plan de pago ("canAccessOdds").
  - The Odds API tiene plan gratis (500 créditos/mes) pero NO región Colombia:
    se usa su región "eu", que incluye marcas que también operan en Colombia
    (Betsson, Codere, 1xBet, bwin...). Son las cuotas de sus sitios europeos:
    parecidas a las colombianas, no idénticas — por eso los Tips tienen un
    campo de cuota manual para corregir con la de tu casa antes de apostar.

Costo: cada consulta = mercados x regiones créditos. Acá siempre 2 mercados
(h2h = 1X2, totals = más/menos goles) x 1 región = 2 créditos por
competición, sin importar cuántos partidos traiga. /sports no gasta créditos.

Aislado igual que europa_league.py/national_teams.py: nunca lanza hacia el
dashboard; ante cualquier problema devuelve vacío y un mensaje.

Requiere ODDS_API_KEY en el .env (o en los secrets de Streamlit Cloud).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import requests

from . import config  # noqa: F401  -- importarlo corre load_dotenv() antes del os.getenv de abajo
from .goal_api_client import team_name_similarity

BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

REGION = "eu"
MARKETS = "h2h,totals"
TOTALS_LINE = 2.5

# Claves de The Odds API por competición del proyecto. No se confía a ciegas:
# available_sport_keys() las valida contra /sports (gratis) y las que no
# existan simplemente quedan sin cuota automática.
SPORT_KEYS = {
    "PL": "soccer_epl",
    "PD": "soccer_spain_la_liga",
    "SA": "soccer_italy_serie_a",
    "BL1": "soccer_germany_bundesliga",
    "FL1": "soccer_france_ligue_one",
    "CL": "soccer_uefa_champs_league",
    "EL": "soccer_uefa_europa_league",
    "NT": "soccer_uefa_nations_league",
}

# Marcas con operación legal en Colombia. Se comparan contra el título que
# devuelve el API (no contra su "key", que no está documentada por casa) para
# no depender de adivinar identificadores exactos.
COLOMBIA_BRANDS = (
    "betsson", "codere", "1xbet", "bwin", "betway", "stake",
    "sportium", "rushbet", "wplay", "betplay", "luckia", "yajuego", "zamba",
)


def operates_in_colombia(bookmaker_title: str) -> bool:
    titulo = (bookmaker_title or "").lower().replace(" ", "")
    return any(marca in titulo for marca in COLOMBIA_BRANDS)


@dataclass
class OddsResult:
    events: list[dict] = field(default_factory=list)
    remaining: int | None = None  # créditos que quedan este mes (header x-requests-remaining)
    error: str | None = None


def available_sport_keys(api_key: str = "") -> set[str]:
    key = api_key or ODDS_API_KEY
    if not key:
        return set()
    try:
        resp = requests.get(f"{BASE_URL}/sports", params={"apiKey": key}, timeout=15)
        resp.raise_for_status()
        return {s.get("key") for s in resp.json() if s.get("key")}
    except Exception:
        return set()


def fetch_odds(competition_code: str, date_from_utc: str, date_to_utc: str,
               api_key: str = "") -> OddsResult:
    """Eventos con cuotas de una competición entre dos instantes UTC
    ('YYYY-MM-DDTHH:MM:SSZ'). 2 créditos por llamada."""
    key = api_key or ODDS_API_KEY
    if not key:
        return OddsResult(error="Falta ODDS_API_KEY.")
    sport = SPORT_KEYS.get(competition_code)
    if not sport:
        return OddsResult(error=f"Sin clave de The Odds API para {competition_code}.")
    params = {
        "apiKey": key,
        "regions": REGION,
        "markets": MARKETS,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "commenceTimeFrom": date_from_utc,
        "commenceTimeTo": date_to_utc,
    }
    try:
        resp = requests.get(f"{BASE_URL}/sports/{sport}/odds", params=params, timeout=20)
    except Exception as e:
        return OddsResult(error=f"No se pudo consultar The Odds API: {e}")

    restante = resp.headers.get("x-requests-remaining")
    restante = int(float(restante)) if restante not in (None, "") else None
    if resp.status_code == 401:
        return OddsResult(remaining=restante, error="ODDS_API_KEY inválida.")
    if resp.status_code == 429:
        return OddsResult(remaining=restante, error="Se agotaron los créditos del mes de The Odds API.")
    if resp.status_code == 404:
        return OddsResult(remaining=restante, error=f"The Odds API no tiene '{sport}' activa ahora.")
    if not resp.ok:
        return OddsResult(remaining=restante, error=f"The Odds API respondió {resp.status_code}.")
    data = resp.json()
    return OddsResult(events=data if isinstance(data, list) else [], remaining=restante)


def _match_score(event: dict, home: str, away: str, home_short: str, away_short: str) -> float:
    ev_home, ev_away = event.get("home_team", ""), event.get("away_team", "")
    return (
        max(team_name_similarity(ev_home, home), team_name_similarity(ev_home, home_short))
        + max(team_name_similarity(ev_away, away), team_name_similarity(ev_away, away_short))
    )


def find_event(events: list[dict], home: str, away: str, utc_date: str,
               home_short: str = "", away_short: str = "") -> dict | None:
    """Mismo criterio que goal_api_client.find_fixture_id: ancla primero en la
    hora de inicio (ambas fuentes la dan en UTC), y el nombre solo desempata
    — los nombres entre fuentes pueden ser muy distintos ("FC Internazionale
    Milano" contra "Inter Milan")."""
    if not events:
        return None
    home_short, away_short = home_short or home, away_short or away
    misma_hora = [e for e in events if str(e.get("commence_time", ""))[:16] == str(utc_date)[:16]]
    if misma_hora:
        mejor = max(misma_hora, key=lambda e: _match_score(e, home, away, home_short, away_short))
        if _match_score(mejor, home, away, home_short, away_short) >= 0.9:
            return mejor
    mejor = max(events, key=lambda e: _match_score(e, home, away, home_short, away_short))
    return mejor if _match_score(mejor, home, away, home_short, away_short) >= 1.4 else None


def best_prices(event: dict, only_colombia: bool = True) -> dict[str, tuple[float, str]]:
    """{'home'|'draw'|'away'|'over'|'under': (mejor_cuota, casa)} entre las
    casas del evento. `only_colombia`: solo marcas que operan en Colombia."""
    home_name, away_name = event.get("home_team"), event.get("away_team")
    mejores: dict[str, tuple[float, str]] = {}

    def _anotar(clave: str, precio, casa: str):
        try:
            precio = float(precio)
        except (TypeError, ValueError):
            return
        if precio <= 1.0:
            return
        if clave not in mejores or precio > mejores[clave][0]:
            mejores[clave] = (precio, casa)

    for book in event.get("bookmakers", []):
        casa = book.get("title") or book.get("key", "")
        if only_colombia and not operates_in_colombia(casa):
            continue
        for market in book.get("markets", []):
            for o in market.get("outcomes", []):
                nombre = o.get("name")
                if market.get("key") == "h2h":
                    if nombre == home_name:
                        _anotar("home", o.get("price"), casa)
                    elif nombre == away_name:
                        _anotar("away", o.get("price"), casa)
                    elif str(nombre).lower() == "draw":
                        _anotar("draw", o.get("price"), casa)
                elif market.get("key") == "totals" and o.get("point") == TOTALS_LINE:
                    if str(nombre).lower() == "over":
                        _anotar("over", o.get("price"), casa)
                    elif str(nombre).lower() == "under":
                        _anotar("under", o.get("price"), casa)
        # Doble oportunidad DERIVADA del 1X2 de la misma casa: 1/(1/local +
        # 1/empate) es lo que esa casa pagaría por "1X" con el mismo margen.
        # Pedir el mercado real costaría ~2 créditos por partido (endpoint por
        # evento) y no entra en el plan gratis.
        h2h = {o.get("name"): o.get("price") for m in book.get("markets", []) if m.get("key") == "h2h"
               for o in m.get("outcomes", [])}
        local, visita = h2h.get(home_name), h2h.get(away_name)
        empate = next((v for k, v in h2h.items() if str(k).lower() == "draw"), None)
        try:
            if local and empate:
                _anotar("dc_1x", 1 / (1 / float(local) + 1 / float(empate)), casa)
            if visita and empate:
                _anotar("dc_x2", 1 / (1 / float(visita) + 1 / float(empate)), casa)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return mejores


# --- Historial de cuotas (versionado en git, pedido explícito del usuario) ---
# Una fila por evento y consulta, con la mejor cuota por pick (solo casas que
# operan en Colombia, y entre todas). Sirve para (1) no volver a gastar créditos
# en la misma ventana tras un reinicio de Streamlit Cloud y (2) medir después si
# los picks recomendados de verdad pagaban más de lo que valían.
SNAPSHOT_PATH = config.ROOT_DIR / "data" / "tracking" / "odds_snapshots.csv"
PRICE_KEYS = ["home", "draw", "away", "over", "under", "dc_1x", "dc_x2"]
# 6 h: las cuotas previas al partido se mueven poco y cada consulta nueva gasta
# 2 de los 500 créditos/mes del plan gratis.
SNAPSHOT_MAX_AGE_HOURS = 6


def _event_row(event: dict, meta: dict) -> dict:
    fila = dict(meta, event_id=event.get("id"), commence_time=str(event.get("commence_time", ""))[:19] + "Z",
                home_team=event.get("home_team"), away_team=event.get("away_team"))
    for filtro, solo_co in (("co", True), ("all", False)):
        precios = best_prices(event, only_colombia=solo_co)
        for k in PRICE_KEYS:
            precio, casa = precios.get(k, (None, ""))
            fila[f"{filtro}_{k}"] = round(precio, 3) if precio else None
            fila[f"{filtro}_{k}_book"] = casa
    return fila


def get_odds_table(competition_code: str, date_from_utc: str, date_to_utc: str,
                   api_key: str = "") -> tuple[list[dict], int | None, str | None]:
    """(eventos, créditos_restantes, error). Cada evento es un dict con
    home_team/away_team/commence_time (lo que usa find_event) y las mejores
    cuotas. Si la misma ventana se consultó hace menos de SNAPSHOT_MAX_AGE_HOURS,
    sale del historial sin gastar créditos."""
    import pandas as pd  # local: el resto del módulo no depende de pandas

    ahora = pd.Timestamp.now(tz="UTC")
    if SNAPSHOT_PATH.exists():
        hist = pd.read_csv(SNAPSHOT_PATH)
        ventana = hist[(hist["competition"] == competition_code) & (hist["window_from"] == date_from_utc)
                       & (hist["window_to"] == date_to_utc)]
        if not ventana.empty:
            ultima = ventana["fetched_at"].max()
            if ahora - pd.Timestamp(ultima) < pd.Timedelta(hours=SNAPSHOT_MAX_AGE_HOURS):
                filas = ventana[(ventana["fetched_at"] == ultima) & ventana["event_id"].notna()]
                return filas.where(filas.notna(), None).to_dict("records"), None, None

    resultado = fetch_odds(competition_code, date_from_utc, date_to_utc, api_key=api_key)
    if resultado.error:
        return [], resultado.remaining, resultado.error
    meta = {"fetched_at": ahora.strftime("%Y-%m-%dT%H:%M:%SZ"), "competition": competition_code,
            "window_from": date_from_utc, "window_to": date_to_utc}
    filas = [_event_row(e, meta) for e in resultado.events]
    # Una ventana sin partidos también se anota (fila sin event_id), para no
    # volver a pagar créditos por descubrir de nuevo que está vacía.
    a_guardar = filas or [dict(meta, event_id=None)]
    nuevo = pd.DataFrame(a_guardar)
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SNAPSHOT_PATH.exists():
        nuevo = pd.concat([pd.read_csv(SNAPSHOT_PATH), nuevo], ignore_index=True)
    nuevo.to_csv(SNAPSHOT_PATH, index=False, encoding="utf-8")
    return filas, resultado.remaining, None


def row_price(row: dict, key: str, only_colombia: bool = True) -> tuple[float, str] | None:
    """(cuota, casa) de una fila de get_odds_table, o None si no hay."""
    filtro = "co" if only_colombia else "all"
    precio = row.get(f"{filtro}_{key}")
    try:
        precio = float(precio)
    except (TypeError, ValueError):
        return None
    if precio != precio or precio <= 1:  # NaN o inválida
        return None
    return precio, row.get(f"{filtro}_{key}_book") or ""
