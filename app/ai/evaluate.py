from __future__ import annotations

from .training import evaluate as _implementation

# Backward-compatibility shim for code still importing the legacy app.ai.evaluate path.
globals().update(
    {
        name: value
        for name, value in vars(_implementation).items()
        if not name.startswith("__")
    }
)
