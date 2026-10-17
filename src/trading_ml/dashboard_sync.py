"""Compatibilidad de importación; el adaptador vive en ``adapters``."""

from .adapters.dashboard_sync import build_dashboard_payload, main, write_live_js

__all__ = ["build_dashboard_payload", "main", "write_live_js"]


if __name__ == "__main__":
    main()
