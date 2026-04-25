"""
FastAPI server for the ESG Compliance Environment.

Design decisions:
- Per-session environments (UUID-keyed) for thread-safe parallel RL rollouts.
- Per-step wall-clock timeout to prevent infinite loops during training.
- Budget abuse detection returned in step response info.
- Graceful session expiry after inactivity.
"""

import threading
import time
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from env import ESGEnvironment
from tasks import TASKS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STEP_TIMEOUT_SECONDS = 10.0    # Max wall-clock time per step
SESSION_TTL_SECONDS  = 3600    # Sessions expire after 1 hour of inactivity
MAX_SESSIONS         = 64      # Prevent memory exhaustion

# ---------------------------------------------------------------------------
# App + session store
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ESG Compliance Environment API",
    description="OpenEnv-compliant ESG environment with thread-safe session management.",
    version="1.1.0",
)

_sessions: Dict[str, dict] = {}   # session_id -> {env, last_active}
_sessions_lock = threading.Lock()


def _prune_expired_sessions() -> None:
    """Remove sessions that have been inactive beyond TTL."""
    now = time.monotonic()
    with _sessions_lock:
        expired = [
            sid for sid, s in _sessions.items()
            if now - s["last_active"] > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del _sessions[sid]


def _get_session(session_id: str) -> ESGEnvironment:
    with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found. Call /reset first.",
            )
        _sessions[session_id]["last_active"] = time.monotonic()
        return _sessions[session_id]["env"]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = Field(default="basic_compliance")
    seed: int = Field(default=42)


class StepRequest(BaseModel):
    session_id: str
    action: int = Field(..., ge=0, le=8)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    _prune_expired_sessions()
    return {
        "message": "ESG Environment Running",
        "version": "1.1.0",
        "available_tasks": list(TASKS.keys()),
        "active_sessions": len(_sessions),
        "step_timeout_seconds": STEP_TIMEOUT_SECONDS,
    }


@app.post("/reset")
def reset(req: ResetRequest):
    """Create a new isolated session and return initial observation."""
    _prune_expired_sessions()

    if req.task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'. Available: {list(TASKS.keys())}",
        )

    with _sessions_lock:
        if len(_sessions) >= MAX_SESSIONS:
            raise HTTPException(
                status_code=503,
                detail="Maximum concurrent sessions reached. Try again later.",
            )
        session_id = str(uuid.uuid4())
        env = ESGEnvironment(task_config=TASKS[req.task_id], seed=req.seed)
        obs = env.reset()
        _sessions[session_id] = {"env": env, "last_active": time.monotonic()}

    return {"session_id": session_id, "observation": obs.model_dump()}


@app.post("/step")
def step(payload: StepRequest):
    """Execute one action with wall-clock timeout protection."""
    env = _get_session(payload.session_id)

    result = {}
    error_holder = {}

    def _do_step():
        try:
            observation, reward, terminated, truncated, info = env.step(payload.action)

            # Budget abuse detection
            budget_warning = None
            if observation.available_budget < -50_000:
                budget_warning = "BUDGET_CRITICAL: available_budget below -$50,000"
            elif observation.available_budget < 0:
                budget_warning = "BUDGET_NEGATIVE: spending beyond budget"

            result.update({
                "observation": observation.model_dump(),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": {**info, "budget_warning": budget_warning},
            })
        except Exception as exc:
            error_holder["error"] = str(exc)

    t = threading.Thread(target=_do_step, daemon=True)
    t.start()
    t.join(timeout=STEP_TIMEOUT_SECONDS)

    if t.is_alive():
        raise HTTPException(
            status_code=504,
            detail=f"Step timed out after {STEP_TIMEOUT_SECONDS}s. Possible infinite loop.",
        )
    if error_holder:
        raise HTTPException(status_code=400, detail=error_holder["error"])

    return result


@app.get("/state/{session_id}")
def state(session_id: str):
    """Get current observation without stepping."""
    env = _get_session(session_id)
    try:
        return env.state().model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/session/{session_id}")
def close_session(session_id: str):
    """Explicitly close and clean up a session."""
    with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Session not found.")
        del _sessions[session_id]
    return {"message": f"Session '{session_id}' closed."}


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(_sessions)}