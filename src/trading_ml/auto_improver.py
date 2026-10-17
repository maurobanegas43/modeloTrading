"""Compatibilidad del punto de entrada del optimizador paper."""

from .adapters.auto_improver import main, run_autonomous_improvement_cycle

__all__ = ["main", "run_autonomous_improvement_cycle"]


if __name__ == "__main__":
    main()
