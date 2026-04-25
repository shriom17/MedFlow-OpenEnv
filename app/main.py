"""FastAPI app for the MedFlow-OpenEnv project."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI

try:
    from openenv.core.env_server.http_server import create_app
except Exception:
    create_app = None

from app.env import HospitalQueueEnvironment
from app.models import HospitalAction, HospitalObservation
from app.tasks import ALL_TASKS
from app.templates import DASHBOARD_HTML


if create_app is not None:
    app = create_app(
        env=HospitalQueueEnvironment,
        action_cls=HospitalAction,
        observation_cls=HospitalObservation,
        env_name="hospital_queue_env",
        max_concurrent_envs=4,
    )
else:
    app = FastAPI(title="MedFlow OpenEnv", version="1.0.0")

LOGS_DIR = Path.cwd() / "outputs" / "logs"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ok(data: dict) -> dict:
    return {"success": True, "timestamp": _utc_now_iso(), "data": data}


def _task_summary() -> list[dict]:
    items = []
    for task in ALL_TASKS.values():
        items.append(
            {
                "task_id": task.task_id,
                "difficulty": task.difficulty,
                "description": task.description,
                "max_steps": task.max_steps,
                "patients": len(task.initial_patients) + len(task.dynamic_arrivals),
                "doctors": len(task.doctors),
                "beds": task.total_beds,
            }
        )
    return items


def _heuristic_action_from_obs(obs: HospitalObservation) -> dict:
    waiting = list(obs.waiting_patients)
    doctors = list(obs.doctors)
    free_docs = [d for d in doctors if not d.get("busy", False)]

    if waiting and free_docs:
        priority_rank = {"emergency": 0, "urgent": 1, "normal": 2}
        patient = sorted(
            waiting,
            key=lambda p: (
                priority_rank.get(p.get("priority", "normal"), 3),
                -int(p.get("severity_score", 0)),
                -int(p.get("wait_minutes", 0)),
                int(p.get("id", 0)),
            ),
        )[0]
        required = patient.get("required_specialization")
        matching = [d for d in free_docs if d.get("specialization") == required]
        general = [d for d in free_docs if d.get("specialization") == "General"]
        doctor = (matching or general or free_docs)[0]
        return {
            "action_type": "assign",
            "patient_id": int(patient["id"]),
            "doctor_id": int(doctor["id"]),
        }

    if waiting:
        patient = sorted(
            waiting,
            key=lambda p: (
                0 if p.get("priority") == "emergency" else 1 if p.get("priority") == "urgent" else 2,
                -int(p.get("wait_minutes", 0)),
                int(p.get("id", 0)),
            ),
        )[0]
        return {"action_type": "prioritize", "patient_id": int(patient["id"]), "doctor_id": None}

    return {"action_type": "wait", "patient_id": None, "doctor_id": None}


def _snapshot(task_id: str = "easy_small_clinic") -> dict:
    env = HospitalQueueEnvironment()
    obs = env.reset(task_id=task_id)
    return {
        "task_id": task_id,
        "observation": asdict(obs),
        "state": asdict(env.state),
        "recommended_action": _heuristic_action_from_obs(obs),
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "timestamp": _utc_now_iso(),
            "error": {"code": "internal_error", "message": str(exc)},
        },
    )


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Serve the interactive dashboard as the default landing page."""
    possible_paths = [
        Path(__file__).parent.parent / "templates" / "ui.html",
        Path.cwd() / "templates" / "ui.html",
        Path("/app/templates/ui.html"),
    ]
    for template_path in possible_paths:
        if template_path.exists():
            try:
                return template_path.read_text(encoding="utf-8")
            except Exception:
                continue
    return DASHBOARD_HTML


@app.get("/api", response_class=JSONResponse)
def api_info() -> dict[str, str]:
    return _ok(
        {
            "message": "MedFlow OpenEnv API is running",
            "docs": "/docs",
            "health": "/health",
            "openapi": "/openapi.json",
            "tasks": "/api/tasks",
            "snapshot": "/api/tools/snapshot?task_id=easy_small_clinic",
            "benchmark": "/api/benchmark/quick",
        }
    )


@app.get("/health")
def health() -> dict:
    return _ok({"status": "healthy", "service": "medflow-openenv"})


@app.get("/api/tasks")
def api_tasks() -> dict:
    return _ok({"tasks": _task_summary()})


@app.get("/api/tools/snapshot")
def api_snapshot(task_id: str = "easy_small_clinic") -> dict:
    if task_id not in ALL_TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    return _ok(_snapshot(task_id))


@app.get("/api/tools/recommendation")
def api_recommendation(task_id: str = "easy_small_clinic") -> dict:
    if task_id not in ALL_TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    data = _snapshot(task_id)
    return _ok(
        {
            "task_id": task_id,
            "recommended_action": data["recommended_action"],
            "observation": data["observation"],
        }
    )


@app.get("/api/benchmark/quick")
def api_benchmark_quick() -> dict:
    results = []
    for task_id in ALL_TASKS.keys():
        snapshot = _snapshot(task_id)
        results.append(
            {
                "task_id": task_id,
                "recommended_action": snapshot["recommended_action"],
                "initial_queue_length": snapshot["observation"]["queue_length"],
                "beds_available": snapshot["observation"]["beds_available"],
            }
        )
    return _ok({"results": results, "note": "This endpoint returns a zero-step benchmark snapshot."})


@app.get("/logs")
def list_logs() -> dict:
    """List available episode log files with simple summaries."""
    if not LOGS_DIR.exists():
        return _ok({"files": []})

    files = sorted(LOGS_DIR.glob("episode_*.json"))
    out = []
    for p in files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            total = d.get("total_reward") or d.get("reward") or None
            if total is None:
                steps = d.get("steps") or d.get("episode") or d.get("trajectory") or []
                if steps:
                    total = sum(s.get("reward", 0) for s in steps)
            out.append({"name": p.name, "total_reward": total})
        except Exception:
            out.append({"name": p.name, "total_reward": None, "error": "read_error"})

    return _ok({"files": out})


@app.get("/logs/{name}")
def get_log(name: str) -> dict:
    """Return the full JSON contents of a named log file."""
    path = LOGS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Log not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _ok({"log": data})
