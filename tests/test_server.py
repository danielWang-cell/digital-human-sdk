import os
import tempfile
import unittest
from pathlib import Path

_state = tempfile.TemporaryDirectory()
os.environ["DH_SERVER_STATE_DIR"] = _state.name
from server import app as service
from fastapi.testclient import TestClient


class TaskServiceTest(unittest.TestCase):
    def testTimeoutPersistsFailureAndPartialOutput(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            renderer = root / "slow-renderer"
            renderer.write_text("#!/usr/bin/env python3\nimport time\nprint('partial output', flush=True)\ntime.sleep(2)\n")
            renderer.chmod(0o700)
            original = service.store, service.APP_BINARY, service.BUILD_DIR, service.STATE_DIR, service.TASK_TIMEOUT_S
            try:
                service.store = service.TaskStore(root / "tasks.json")
                service.APP_BINARY = renderer
                service.BUILD_DIR = root
                service.STATE_DIR = root
                service.TASK_TIMEOUT_S = 0.2
                task = service.store.create("default")
                service.run_task(task["task_id"], "default")
                saved = service.store.get(task["task_id"])
                self.assertEqual(saved["status"], "failed")
                self.assertEqual(saved["error"], "task timeout")
                self.assertIn("partial output", Path(saved["log"]).read_text())
                self.assertIn("finished_at", saved)
            finally:
                service.store, service.APP_BINARY, service.BUILD_DIR, service.STATE_DIR, service.TASK_TIMEOUT_S = original

    def testTimeoutStopsRendererChildren(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "child-finished"
            child_code = "import time,pathlib; time.sleep(0.7); pathlib.Path(" + repr(str(marker)) + ").write_text('late')"
            renderer = root / "renderer-with-child"
            renderer.write_text("#!/usr/bin/env python3\nimport subprocess,sys,time\nsubprocess.Popen([sys.executable, '-c', " + repr(child_code) + "])\nprint('started', flush=True)\ntime.sleep(2)\n")
            renderer.chmod(0o700)
            original = service.store, service.APP_BINARY, service.BUILD_DIR, service.STATE_DIR, service.TASK_TIMEOUT_S
            try:
                service.store = service.TaskStore(root / "tasks.json")
                service.APP_BINARY = renderer
                service.BUILD_DIR = root
                service.STATE_DIR = root
                service.TASK_TIMEOUT_S = 0.2
                task = service.store.create("default")
                service.run_task(task["task_id"], "default")
                import time
                time.sleep(0.8)
                self.assertFalse(marker.exists(), "renderer child survived task timeout")
                self.assertEqual(service.store.get(task["task_id"])["status"], "failed")
            finally:
                service.store, service.APP_BINARY, service.BUILD_DIR, service.STATE_DIR, service.TASK_TIMEOUT_S = original

    def testRestartMarksInterruptedTasksFailed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.json"
            store = service.TaskStore(path)
            task = store.create("default")
            store.update(task["task_id"], status="running")
            restarted = service.TaskStore(path)
            saved = restarted.get(task["task_id"])
            self.assertEqual(saved["status"], "failed")
            self.assertIn("service restarted", saved["error"])

    def testUnknownSampleAndTaskReturnErrors(self):
        client = TestClient(service.app)
        response = client.post("/api/tasks", json={"sample_id": "unknown"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "unknown sample_id")
        for suffix in ("", "/metrics", "/result"):
            self.assertEqual(client.get("/api/tasks/unknown" + suffix).status_code, 404)


if __name__ == "__main__":
    unittest.main()
