from __future__ import annotations

from .training import training_data as _implementation

# Backward-compatibility shim for code still importing the legacy app.ai.training_data path.
globals().update(
    {
        name: value
        for name, value in vars(_implementation).items()
        if not name.startswith("__")
    }
)
