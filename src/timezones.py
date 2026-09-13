"""
Formatea fechas UTC de football-data.org/Goal API a la hora de Colombia.

Colombia usa un único huso fijo (UTC-5, sin horario de verano), así que la
conversión es directa vía zoneinfo — pero el nombre de mes/día en español no
sale de strftime sin configurar el locale del sistema (y en Windows eso es
frágil), así que se arma a mano con las abreviaturas.
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

BOGOTA_TZ = ZoneInfo("America/Bogota")

_DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
_MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def to_bogota(utc_date: str) -> datetime | None:
    """Parsea un 'YYYY-MM-DDTHH:MM:SSZ' (o con milisegundos) y lo pasa a Bogotá."""
    if not utc_date:
        return None
    texto = str(utc_date).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(texto)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(BOGOTA_TZ)


def format_bogota(utc_date: str, with_weekday: bool = True) -> str:
    """'2026-09-12T19:00:00Z' -> 'sáb 12 sep · 14:00' (hora de Colombia)."""
    local = to_bogota(utc_date)
    if local is None:
        return "-"
    dia = f"{_DIAS[local.weekday()]} " if with_weekday else ""
    return f"{dia}{local.day} {_MESES[local.month - 1]} · {local.strftime('%H:%M')}"
