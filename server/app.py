from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = Path(os.getenv("DH_SERVER_STATE_DIR", ROOT / ".server-state"))
BUILD_DIR = Path(os.getenv("DH_BUILD_DIR", ROOT / ".docker-build"))
APP_BINARY = Path(os.getenv("DH_APP_BINARY", BUILD_DIR / "bin" / "digital_human_app"))
MODELS_DIR = Path(os.getenv("DH_MODELS_DIR", ROOT / "models"))
TASK_TIMEOUT_S = int(os.getenv("DH_TASK_TIMEOUT_S", "300"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskRequest(BaseModel):
    sample_id: str = Field(default="default", min_length=1)


class TaskStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.tasks: dict[str, dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.tasks = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                self.tasks = {}
        for task in self.tasks.values():
            if task.get("status") in {"queued", "running"}:
                task["status"] = "failed"
                task["error"] = "service restarted before task completion"
                task["finished_at"] = now()
        self._save()

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.tasks, ensure_ascii=False, indent=2))
        tmp.replace(self.path)

    def create(self, sample_id: str) -> dict[str, Any]:
        with self.lock:
            task_id = uuid.uuid4().hex
            task = {"task_id": task_id, "sample_id": sample_id,
                    "status": "queued", "created_at": now()}
            self.tasks[task_id] = task
            self._save()
            return dict(task)

    def update(self, task_id: str, **values: Any) -> dict[str, Any]:
        with self.lock:
            self.tasks[task_id].update(values)
            self._save()
            return dict(self.tasks[task_id])

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self.lock:
            task = self.tasks.get(task_id)
            return dict(task) if task else None


store = TaskStore(STATE_DIR / "tasks.json")
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="digital-human-task")
app = FastAPI(title="Digital Human SDK Task Service", version="0.1.0")


def samples() -> dict[str, dict[str, str]]:
    return {
        "default": {
            "face": str(ROOT / "face.jpg"),
            "audio": str(ROOT / "test_audio_30s_sync.wav"),
            "models": str(MODELS_DIR),
        }
    }


def run_task(task_id: str, sample_id: str) -> None:
    sample = samples().get(sample_id)
    if sample is None:
        store.update(task_id, status="failed", error="unknown sample", finished_at=now())
        return
    output_dir = STATE_DIR / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "result.mp4"
    log_path = output_dir / "worker.log"
    store.update(task_id, status="running", started_at=now(), output=str(output), log=str(log_path))
    command = [str(APP_BINARY), sample["face"], sample["audio"], str(output), sample["models"]]
    try:
        env = os.environ.copy()
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(
            value for value in (
                str(BUILD_DIR / "lib"),
                "/usr/local/lib",
                "/usr/lib/x86_64-linux-gnu",
                existing_ld,
            ) if value
        )
        with subprocess.Popen(command, cwd=BUILD_DIR, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, start_new_session=True) as process:
            try:
                stdout, stderr = process.communicate(timeout=TASK_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
                log_path.write_text(stdout + "\n" + stderr)
                store.update(task_id, status="failed", finished_at=now(), error="task timeout")
                return
        log_path.write_text(stdout + "\n" + stderr)
        metrics = extract_metrics(stdout + "\n" + stderr)
        if process.returncode == 0 and output.exists() and output.stat().st_size > 0:
            store.update(task_id, status="completed", finished_at=now(), metrics=metrics)
        else:
            store.update(task_id, status="failed", finished_at=now(), return_code=process.returncode,
                         error="renderer exited unsuccessfully", metrics=metrics)
    except OSError as exc:
        store.update(task_id, status="failed", finished_at=now(), error=str(exc))


def extract_metrics(log: str) -> dict[str, Any]:
    patterns = {
        "first_frame_ms": r"First frame latency:\s*([0-9.]+) ms",
        "pipeline_ms": r"全链路推理与渲染.*?耗时:\s*([0-9.]+) ms",
        "ffmpeg_merge_ms": r"FFmpeg 音视频合并.*?耗时:\s*([0-9.]+) ms",
    }
    result: dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = re.findall(pattern, log)
        if match:
            result[key] = float(match[-1])
    stage = re.findall(r"\[PerformanceMetrics\] stages=(\{.*\})", log)
    if stage:
        try:
            result["stages"] = json.loads(stage[-1])
        except json.JSONDecodeError:
            pass
    return result


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "renderer": str(APP_BINARY), "renderer_present": APP_BINARY.exists()}


@app.get("/api/samples")
def list_samples() -> list[dict[str, Any]]:
    result = []
    for sample_id, value in samples().items():
        result.append({"sample_id": sample_id,
                       "available": all(Path(value[key]).exists() for key in ("face", "audio", "models")),
                       "face": value["face"], "audio": value["audio"]})
    return result


@app.post("/api/tasks", status_code=202)
def submit_task(request: TaskRequest) -> dict[str, Any]:
    if request.sample_id not in samples():
        raise HTTPException(status_code=400, detail="unknown sample_id")
    task = store.create(request.sample_id)
    executor.submit(run_task, task["task_id"], request.sample_id)
    return task


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str) -> dict[str, Any]:
    task = store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@app.get("/api/tasks/{task_id}/result")
def task_result(task_id: str) -> FileResponse:
    task = store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    if task.get("status") != "completed":
        raise HTTPException(status_code=409, detail=f"task status is {task.get('status')}")
    output = Path(task["output"])
    if not output.exists():
        raise HTTPException(status_code=410, detail="result file is missing")
    return FileResponse(output, media_type="video/mp4", filename=f"{task_id}.mp4")


@app.get("/api/tasks/{task_id}/metrics")
def task_metrics(task_id: str) -> dict[str, Any]:
    task = store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return {"task_id": task_id, "status": task.get("status"), "metrics": task.get("metrics", {})}
