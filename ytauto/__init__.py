"""Autonomous YouTube pipeline: research -> script -> render -> publish.

The package is importable with no side effects. `get_settings()` is what creates
directories, and nothing touches the network or the database until a command or
a pipeline stage asks it to.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__", "get_settings", "get_db"]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    # Lazy so `import ytauto` stays cheap and does not create directories as a
    # side effect of importing the version.
    if name == "get_settings":
        from .config import get_settings

        return get_settings
    if name == "get_db":
        from .db import get_db

        return get_db
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
