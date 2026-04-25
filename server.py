"""
DeepSentinel – FastAPI Server
Exposes the OpenEnv oversight interface over HTTP + WebSocket.
Compatible with Hugging Face Spaces (port 7860).

Endpoints:
  GET  /health          – liveness probe
  GET  /tasks           – list tasks with descriptions
  POST /reset           – start a new oversight episode, get FleetObservation
  POST /oversee         – submit OverseerAction, get OverseerStepResult
  GET  /state           – query episode state
  WS   /ws              – WebSocket for persistent sessions
"""

from __future__ import annotations
import json
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.models import OverseerAction, FleetObservation, OverseerStepResult
from env.environment import DeepSentinelEnvironment
from env.tasks import list_tasks, get_task

app = FastAPI(
    title="DeepSentinel OpenEnv",
    description=(
        "Multi-agent AI oversight environment. "
        "An overseer agent monitors a fleet of deepfake detectors, "
        "catches adversarial agents, and produces validated verdicts. "
        "OpenEnv 3-method interface: reset / oversee / state."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: Dict[str, DeepSentinelEnvironment] = {}


# ── Request/Response models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: str = "medium"
    seed: Optional[int] = None
    session_id: Optional[str] = None
    adversarial_flip_prob: float = 0.35


class ResetResponse(BaseModel):
    session_id: str
    fleet_observation: FleetObservation


class OverseeRequest(BaseModel):
    session_id: str
    action: OverseerAction


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "DeepSentinel",
        "sessions_active": len(_sessions),
    }


@app.get("/tasks")
def tasks():
    return list_tasks()


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest):
    sid = req.session_id or str(uuid.uuid4())
    env = _sessions.setdefault(sid, DeepSentinelEnvironment(session_id=sid))

    # Validate difficulty
    valid = {"easy", "medium", "hard"}
    if req.difficulty not in valid:
        raise HTTPException(status_code=422, detail=f"difficulty must be one of {valid}")

    fleet_obs = env.reset(
        difficulty=req.difficulty,
        seed=req.seed,
        adversarial_flip_prob=req.adversarial_flip_prob,
    )
    return ResetResponse(session_id=sid, fleet_observation=fleet_obs)


@app.post("/oversee", response_model=OverseerStepResult)
def oversee(req: OverseeRequest):
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /reset first.",
        )
    try:
        result = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state")
def state(session_id: str):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    s = env.state
    if s is None:
        return {"session_id": session_id, "state": None}
    return {"session_id": session_id, "state": s.model_dump()}


@app.get("/fleet")
def fleet_state(session_id: str):
    """Return the current FleetObservation for a session."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    obs = env.fleet_obs
    if obs is None:
        return {"session_id": session_id, "fleet_observation": None}
    return {"session_id": session_id, "fleet_observation": obs.model_dump()}


# ── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    env = DeepSentinelEnvironment(session_id=session_id)
    _sessions[session_id] = env
    await websocket.send_text(json.dumps({
        "type": "connected",
        "session_id": session_id,
        "service": "DeepSentinel",
    }))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON",
                }))
                continue

            msg_type = msg.get("type")

            if msg_type == "reset":
                try:
                    obs = env.reset(
                        difficulty=msg.get("difficulty", "medium"),
                        seed=msg.get("seed"),
                        adversarial_flip_prob=msg.get("adversarial_flip_prob", 0.35),
                    )
                    await websocket.send_text(json.dumps({
                        "type": "fleet_observation",
                        "data": obs.model_dump(),
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e),
                    }))

            elif msg_type == "oversee":
                try:
                    action = OverseerAction(**msg.get("action", {}))
                    result = env.step(action)
                    await websocket.send_text(json.dumps({
                        "type": "overseer_result",
                        "data": result.model_dump(),
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": str(e),
                    }))

            elif msg_type == "state":
                s = env.state
                await websocket.send_text(json.dumps({
                    "type": "state",
                    "data": s.model_dump() if s else None,
                }))

            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type!r}. "
                               f"Use: reset, oversee, state",
                }))

    except WebSocketDisconnect:
        _sessions.pop(session_id, None)
