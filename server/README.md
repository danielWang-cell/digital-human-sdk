# FastAPI task service

Run from the repository root after building the C++ application:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

The first version accepts only the preconfigured `default` sample. Set
`DH_BUILD_DIR`, `DH_MODELS_DIR`, `DH_SERVER_STATE_DIR`, or
`DH_TASK_TIMEOUT_S` to adapt paths and limits. Tasks are persisted in
`tasks.json`; queued/running tasks are marked failed on service restart.
