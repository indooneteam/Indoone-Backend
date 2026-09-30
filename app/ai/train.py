from __future__ import annotations

from .training import train as _implementation

# Backward-compatibility shim for code still importing the legacy app.ai.train path.
globals().update(
    {
        name: value
        for name, value in vars(_implementation).items()
        if not name.startswith("__")
    }
)
