"""FastAPI app for the MedFlow-OpenEnv project."""

from datetime import datetime, timezone
from pathlib import Path
import os

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from openenv.core.env_server.http_server import create_app
import json
from fastapi import HTTPException

from app.models import HospitalAction, HospitalObservation
from app.env import HospitalQueueEnvironment
from app.templates import DASHBOARD_HTML


app = create_app(
    env=HospitalQueueEnvironment,
    action_cls=HospitalAction,
    observation_cls=HospitalObservation,
    env_name="hospital_queue_env",
    max_concurrent_envs=4,
)
LOGS_DIR = Path.cwd() / "outputs" / "logs"
  

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ok(data: dict) -> dict:
    return {
        "success": True,
        "timestamp": _utc_now_iso(),
        "data": data,
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    # Keep error payload consistent for unhandled server errors.
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "timestamp": _utc_now_iso(),
            "error": {
                "code": "internal_error",
                "message": str(exc),
            },
        },
    )


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Serve the interactive dashboard as the default landing page."""
    # Try multiple possible paths first.
    possible_paths = [
        Path(__file__).parent.parent / "templates" / "ui.html",  # Production path
        Path.cwd() / "templates" / "ui.html",  # Current working directory
        Path("/app/templates/ui.html"),  # HF Space absolute path
    ]
    
    for template_path in possible_paths:
        if template_path.exists():
            try:
                return template_path.read_text(encoding="utf-8")
            except Exception:
                continue

    # Deployment-safe fallback: serve embedded template when file paths are unavailable.
    return DASHBOARD_HTML


@app.get("/api", response_class=JSONResponse)
def api_info() -> dict[str, str]:
    """REST API information endpoint."""
    try:
        return _ok(
            {
                "message": "MedFlow OpenEnv API is running",
                "docs": "/docs",
                "health": "/health",
                "openapi": "/openapi.json",
            }
        )
    except Exception as exc:
        return {
            "success": False,
            "timestamp": _utc_now_iso(),
            "error": {
                "code": "api_handler_error",
                "message": str(exc),
            },
        }


@app.get("/health")
def health() -> dict:
    try:
        return _ok(
            {
                "status": "healthy",
                "service": "medflow-openenv",
            }
        )
    except Exception as exc:
        return {
            "success": False,
            "timestamp": _utc_now_iso(),
            "error": {
                "code": "health_handler_error",
                "message": str(exc),
            },
        }


        @app.get("/logs")
        def list_logs() -> dict:
            """List available episode log files with simple summaries."""
            try:
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
            except Exception as exc:
                return {
                    "success": False,
                    "timestamp": _utc_now_iso(),
                    "error": {"code": "list_logs_error", "message": str(exc)},
                }


        @app.get("/logs/{name}")
        def get_log(name: str) -> dict:
            """Return the full JSON contents of a named log file."""
            try:
                path = LOGS_DIR / name
                if not path.exists():
                    raise HTTPException(status_code=404, detail="Log not found")
                data = json.loads(path.read_text(encoding="utf-8"))
                return _ok({"log": data})
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc))
