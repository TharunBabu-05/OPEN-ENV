"""
server/app.py — Canonical entrypoint shim.

All API logic lives in the top-level app.py (hardened, session-based,
thread-safe). This module simply re-exports the FastAPI app and provides
the `main()` launcher so the pyproject.toml script entrypoint continues
to work unchanged.

Why a shim instead of duplicating logic:
- Single source of truth for session management, timeouts, and budget warnings.
- Any entrypoint (uvicorn direct, `server` script, Docker CMD) gets identical semantics.
- No global mutable env state — all state is per-session UUID.
"""

import sys
from pathlib import Path

# Ensure the project root is importable when launched as a package entrypoint.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Re-export the hardened app from top-level app.py.
from app import app  # noqa: F401  (re-export for uvicorn "server.app:app")


def main() -> None:
    """CLI entrypoint — launched via `server` script in pyproject.toml."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
