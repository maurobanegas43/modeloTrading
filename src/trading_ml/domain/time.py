"""Funciones de tiempo compartidas por las reglas del dominio."""

from datetime import datetime, timezone


def utc_iso(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat()
