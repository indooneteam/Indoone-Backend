"""Temporary compatibility shim; implementation lives in app.web_research.research.

Keep private helpers exported until the server-specific integration switches imports.
New code should import directly from app.web_research.
"""

from app.web_research import research as _research

__all__ = [name for name in dir(_research) if not name.startswith("__")]
globals().update({name: getattr(_research, name) for name in __all__})
