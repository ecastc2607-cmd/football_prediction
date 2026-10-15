"""
Copas UEFA de clubes vía Goal API (Europa League, Conference League) —
football-data.org no las incluye en el plan gratis. Antes esto vivía solo en
europa_league.py; se generalizó al sumar la Conference League (oct-2026), que
tiene exactamente el mismo formato en Goal API. europa_league.py y
conference_league.py son instancias de esta clase con el mismo contrato de
módulo (fetch_competition, resolve_current_season_and_matchday,
get_standings_map, get_live_matches, current_season).

Regla explícita del usuario: si esta fuente falla o cambia de forma, NUNCA debe
tumbar el flujo de las competiciones de football-data.org. Por eso cada función
falla "limpio" (devuelve vacío/None, o deja pasar una excepción clara) y quien
la llama en app.py la envuelve en su propio try/except.

Ahorro de cuota (pedido del usuario: guardar en el repo lo ya consultado):
/leagues/:id/fixtures ignora cualquier filtro de temporada/fecha (verificado
oct-2026) y trae TODO el historial — 12 páginas la Europa League, 22 la
Conference. Pero viene ordenado del partido más nuevo al más viejo, así que los
partidos ya terminados se guardan versionados en data/tracking/goal_api_cups/
y cada actualización pide páginas solo hasta llegar a partidos ya guardados
(normalmente 2 páginas en vez de 12-22).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from . import config
from .competition_status import CompetitionStatus
from .goal_api_client import GOAL_API_KEY, GoalApiClient, LEAGUE_IDS, live_fixtures, memoized
from .live_matches import REGULATION_MINUTES, live_win_probabilities
from .poisson_model import expected_goals
from .team_strength import team_strength_for_competition

# La fase de liga (formato "una tabla" desde la reforma 2024/25) es la que
# corresponde al concepto de "jornada" que usa el resto del proyecto.
# Goal API usa ambos nombres para la misma fase (2024/25 "League Stage";
# 2025/26 mezcla los dos — verificado oct-2026, junto con partidos de esa fase
# mal etiquetados como "Final", que no se pueden recuperar con seguridad).
LEAGUE_PHASE_STAGES = {"League Phase", "League Stage"}

# Estados que ya no cambian: se guardan en el historial versionado.
FINAL_STATUSES = {"FINISHED", "AFTER_ET", "AFTER_PEN", "CANCELLED", "AWARDED"}
HISTORY_DIR = config.ROOT_DIR / "data" / "tracking" / "goal_api_cups"
HISTORY_COLUMNS = [
    "id", "leagueYear", "stageName", "matchRound", "kickoffUtc", "matchStatus",
    "homeTeamName", "awayTeamName", "homeTeamFullName", "awayTeamFullName",
    "homeTeamScore", "awayTeamScore", "homeTeamHalftimeScore", "awayTeamHalftimeScore",
]
FIXTURES_MEMO_SECONDS = 15 * 60


def _get_with_retry(client: GoalApiClient, path: str, params: dict, attempts: int = 2):
    """Un reintento ante un timeout de red puntual (nos pasó en pruebas a
    mitad de una descarga larga). No reintenta GoalApiRateLimited ni otros
    errores del propio API: solo red/tiempo, donde repetir sí tiene sentido."""
    for intento in range(attempts):
        try:
            return client._get(path, params)
        except requests.exceptions.Timeout:
            if intento == attempts - 1:
                raise
            time.sleep(1.5)


def _normalize_utc(kickoff_iso: str) -> str:
    """'2026-09-16T16:45:00.000Z' -> '2026-09-16T16:45:00Z' — el resto del
    pipeline asume el formato exacto de football-data.org, sin milisegundos."""
    return (kickoff_iso or "")[:19] + "Z"


def _flatten(f: dict) -> dict:
    """Fixture de Goal API -> fila plana (lo mismo que se guarda en el historial)."""
    return {
        "id": f.get("id"),
        "leagueYear": f.get("leagueYear"),
        "stageName": f.get("stageName"),
        "matchRound": f.get("matchRound"),
        "kickoffUtc": f.get("kickoffUtc"),
        "matchStatus": f.get("matchStatus"),
        "homeTeamName": f.get("homeTeamName"),
        "awayTeamName": f.get("awayTeamName"),
        # Goal API da el nombre completo anidado y uno corto al nivel superior.
        "homeTeamFullName": (f.get("homeTeam") or {}).get("name") or f.get("homeTeamName"),
        "awayTeamFullName": (f.get("awayTeam") or {}).get("name") or f.get("awayTeamName"),
        "homeTeamScore": f.get("homeTeamScore"),
        "awayTeamScore": f.get("awayTeamScore"),
        "homeTeamHalftimeScore": f.get("homeTeamHalftimeScore"),
        "awayTeamHalftimeScore": f.get("awayTeamHalftimeScore"),
    }


class GoalApiCup:
    def __init__(self, code: str, strength_from_all_stages: bool = False,
                 extra_strength_from: tuple[str, ...] = ()):
        """`strength_from_all_stages`: además de la fase de liga, guardar en el
        CSV de la temporada la previa y las eliminatorias (con matchday vacío,
        así nunca se confunden con una jornada) para que cuenten en la fuerza
        de los equipos. Lo usa la Conference: 28 de sus 36 equipos no juegan
        ninguna liga que descarguemos, y casi todos jugaron la previa.

        `extra_strength_from`: otras copas cuyo historial versionado (solo
        partidos terminados, sin consultar la API) también suma a la fuerza —
        ej. la Conference toma la Europa League: varios de sus equipos cayeron
        desde la previa de la EL y no tienen ningún partido propio."""
        self.code = code
        self.extra_strength_from = extra_strength_from
        self.league_id = LEAGUE_IDS[code]
        self.strength_from_all_stages = strength_from_all_stages
        self.history_path = HISTORY_DIR / f"{code}_fixtures.csv"

    def _client(self) -> GoalApiClient:
        return GoalApiClient(api_key=GOAL_API_KEY)

    # --- Calendario completo, con historial versionado ---

    def _load_history(self) -> dict[str, dict]:
        if not self.history_path.exists():
            return {}
        try:
            df = pd.read_csv(self.history_path, dtype=str, keep_default_na=False)
        except Exception:
            return {}
        return {r["id"]: {k: (v if v != "" else None) for k, v in r.items()} for r in df.to_dict("records")}

    def _save_history(self, fixtures: list[dict]) -> None:
        finales = [f for f in fixtures if f.get("matchStatus") in FINAL_STATUSES and f.get("id")]
        if not finales:
            return
        df = pd.DataFrame(finales, columns=HISTORY_COLUMNS).sort_values("kickoffUtc", ascending=False)
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.history_path, index=False, encoding="utf-8")

    def _download_fixtures(self) -> list[dict]:
        client = self._client()
        guardados = self._load_history()
        frescos: dict[str, dict] = {}
        offset, page_size = 0, 100
        while offset <= 3000:  # cortafuegos: no debería hacer falta tantas páginas
            page = _get_with_retry(client, f"/leagues/{self.league_id}/fixtures",
                                   {"limit": page_size, "offset": offset}) or []
            for f in page:
                frescos[f.get("id")] = _flatten(f)
            # Página completa ya guardada como terminada: lo de atrás es más
            # viejo y también lo está (viene del más nuevo al más viejo).
            if guardados and page and all(
                (f.get("id") in guardados and f.get("matchStatus") in FINAL_STATUSES) for f in page
            ):
                break
            if len(page) < page_size:
                break
            offset += page_size
        todos = {**guardados, **frescos}
        lista = list(todos.values())
        try:
            self._save_history(lista)
        except OSError:
            pass  # sin disco escribible se sigue igual, solo no se ahorra la próxima vez
        return lista

    def all_fixtures(self) -> list[dict]:
        return memoized(f"{self.code}_all_fixtures", FIXTURES_MEMO_SECONDS, self._download_fixtures)

    def _season_fixtures(self, season: int) -> list[dict]:
        year_label = f"{season}/{season + 1}"
        return [f for f in self.all_fixtures() if f.get("leagueYear") == year_label]

    def _league_phase_fixtures(self, season: int) -> list[dict]:
        return [f for f in self._season_fixtures(season) if f.get("stageName") in LEAGUE_PHASE_STAGES]

    # --- Contrato de módulo ---

    def current_season(self) -> int:
        """Temporada que Goal API considera vigente ("2026/2027" -> 2026)."""
        def _pedir():
            data = self._client()._get(f"/leagues/{self.league_id}", {})
            season_label = (data or {}).get("season", "")
            if not season_label or "/" not in season_label:
                return datetime.now(timezone.utc).year
            return int(season_label.split("/")[0])
        # En memoria 6 h: una carga del dashboard la pedía varias veces.
        return memoized(f"{self.code}_current_season", 6 * 3600, _pedir)

    def resolve_current_season_and_matchday(self) -> CompetitionStatus:
        """La "jornada actual" es la primera de la fase de liga que no está
        100% FINISHED todavía."""
        season = self.current_season()
        fixtures = self._league_phase_fixtures(season)
        if not fixtures:
            return CompetitionStatus(
                season=season, matchday=1, season_start="", season_end="",
                reason="Aún no hay calendario de la fase de liga para esta temporada (Goal API).",
            )
        rounds = sorted({int(f["matchRound"]) for f in fixtures if f.get("matchRound")})
        for md in rounds:
            md_fixtures = [f for f in fixtures if int(f.get("matchRound") or -1) == md]
            if not all(f.get("matchStatus") == "FINISHED" for f in md_fixtures):
                return CompetitionStatus(
                    season=season, matchday=md, season_start="", season_end="",
                    reason=f"Jornada {md} en curso según Goal API (fase de liga).",
                )
        ultima = rounds[-1] if rounds else 1
        return CompetitionStatus(
            season=season, matchday=ultima, season_start="", season_end="",
            reason=f"La fase de liga llegó hasta la jornada {ultima} según los datos disponibles.",
        )

    def fetch_competition(self, season: int) -> pd.DataFrame:
        """Guarda la temporada en el MISMO formato y ruta que
        fetch_football_data.fetch_competition (data/processed/matches_{code}_{season}.csv).

        No atrapa excepciones: si Goal API falla, quien llama (app.py) decide
        cómo aislarlo, en vez de guardar un CSV a medias."""
        nombre = config.COMPETITIONS.get(self.code, self.code)
        print(f"Descargando {nombre} (temporada {season}) vía Goal API...")
        if self.strength_from_all_stages:
            fixtures = self._season_fixtures(season)
        else:
            fixtures = self._league_phase_fixtures(season)

        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = config.RAW_DIR / f"{self.code}_matches_{season}.json"
        raw_path.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2), encoding="utf-8")

        propios = {f.get("id") for f in fixtures}
        for otra in self.extra_strength_from:
            for f in GoalApiCup(otra)._load_history().values():
                if (f.get("leagueYear") == f"{season}/{season + 1}" and f.get("matchStatus") == "FINISHED"
                        and f.get("id") not in propios):
                    fixtures.append({**f, "stageName": f"{otra} · {f.get('stageName')}"})

        rows = []
        for f in fixtures:
            home_goals = pd.to_numeric(f.get("homeTeamScore"), errors="coerce")
            away_goals = pd.to_numeric(f.get("awayTeamScore"), errors="coerce")
            winner = None
            if f.get("matchStatus") == "FINISHED" and pd.notna(home_goals) and pd.notna(away_goals):
                winner = "HOME_TEAM" if home_goals > away_goals else "AWAY_TEAM" if away_goals > home_goals else "DRAW"
            fase_de_liga = f.get("stageName") in LEAGUE_PHASE_STAGES
            rows.append({
                "match_id": f.get("id"),
                "competition": self.code,
                "season_start_year": season,
                # Solo la fase de liga tiene "jornada": la previa y las
                # eliminatorias van sin matchday (aportan fuerza, no se predicen).
                "matchday": int(f["matchRound"]) if fase_de_liga and f.get("matchRound") else None,
                "stage": f.get("stageName"),
                "utc_date": _normalize_utc(f.get("kickoffUtc", "")),
                "status": f.get("matchStatus"),
                "home_team": f.get("homeTeamFullName") or f.get("homeTeamName"),
                "away_team": f.get("awayTeamFullName") or f.get("awayTeamName"),
                "home_team_short": f.get("homeTeamName"),
                "away_team_short": f.get("awayTeamName"),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_ht_goals": pd.to_numeric(f.get("homeTeamHalftimeScore"), errors="coerce"),
                "away_ht_goals": pd.to_numeric(f.get("awayTeamHalftimeScore"), errors="coerce"),
                "winner": winner,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["matchday"] = df["matchday"].astype("Int64")

        csv_path = config.PROCESSED_DIR / f"matches_{self.code}_{season}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")

        finished = (df["status"] == "FINISHED").sum() if not df.empty else 0
        print(f"  {len(df)} partidos guardados en {csv_path.relative_to(config.ROOT_DIR)} "
              f"({finished} finalizados, {len(df) - finished} pendientes/en curso).")
        return df

    def get_standings_map(self, season: int) -> dict[str, int]:
        """{nombre_equipo: posición} de la fase de liga. Devuelve {} ante
        cualquier problema, nunca lanza."""
        try:
            rows = self._client()._get(f"/standings/{self.league_id}", {"limit": 100})
        except Exception:
            return {}
        if not isinstance(rows, list):
            return {}
        posiciones = {}
        for r in rows:
            nombre = (r.get("team") or {}).get("name") or r.get("teamName")
            posicion = r.get("overallLeaguePosition")
            if nombre and posicion is not None:
                try:
                    posiciones[nombre] = int(posicion)
                except (TypeError, ValueError):
                    continue
        return posiciones

    def get_live_matches(self, ensure_data=None) -> pd.DataFrame:
        """Partidos en vivo, con el 1X2 recalculado — mismas columnas que
        live_matches.get_live_matches. Goal API sí entrega el minuto real
        (`matchElapsed`); se deja el nombre "minuto_estimado" para no
        bifurcar el resto del código. Nunca lanza."""
        try:
            todos_en_vivo = live_fixtures(GOAL_API_KEY)
        except Exception:
            return pd.DataFrame()

        propios = [f for f in (todos_en_vivo or []) if f.get("leagueId") == self.league_id]
        if not propios:
            return pd.DataFrame()

        season = self.current_season()
        try:
            if ensure_data:
                ensure_data(self.code, season)
            else:
                self.fetch_competition(season)
            strength = team_strength_for_competition(self.code, [season])
        except Exception:
            strength = None

        rows = []
        for f in propios:
            home = (f.get("homeTeam") or {}).get("name") or f.get("homeTeamName")
            away = (f.get("awayTeam") or {}).get("name") or f.get("awayTeamName")
            hg = pd.to_numeric(f.get("homeTeamScore"), errors="coerce")
            ag = pd.to_numeric(f.get("awayTeamScore"), errors="coerce")
            hg, ag = (0 if pd.isna(hg) else int(hg)), (0 if pd.isna(ag) else int(ag))
            minute = int(f.get("matchElapsed") or 0)
            remaining_frac = max(REGULATION_MINUTES - minute, 0) / REGULATION_MINUTES

            live_h = live_d = live_a = None
            home_xg = away_xg = None
            if strength is not None:
                try:
                    home_xg, away_xg = expected_goals(strength, home, away)
                    live_h, live_d, live_a = live_win_probabilities(
                        hg, ag, home_xg * remaining_frac, away_xg * remaining_frac
                    )
                except KeyError:
                    pass

            rows.append({
                "competition": self.code,
                "competition_name": config.COMPETITIONS.get(self.code, self.code),
                "home_team": home, "away_team": away,
                "utc_date": _normalize_utc(f.get("kickoffUtc", "")),
                "home_goals": hg, "away_goals": ag,
                "status": f.get("matchStatus"),
                "minuto_estimado": minute,
                "live_home_win": round(live_h * 100, 1) if live_h is not None else None,
                "live_draw": round(live_d * 100, 1) if live_d is not None else None,
                "live_away_win": round(live_a * 100, 1) if live_a is not None else None,
                "pre_home_xg": home_xg,
                "pre_away_xg": away_xg,
            })
        return pd.DataFrame(rows)
