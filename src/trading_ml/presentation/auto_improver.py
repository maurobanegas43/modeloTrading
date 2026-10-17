"""Punto de entrada para el ciclo autónomo de evaluación y promoción paper."""

from ..adapters.auto_improver import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
