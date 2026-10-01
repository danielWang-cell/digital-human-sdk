# Docker Compose deployment

Build the C++ renderer in the CPU environment from the repository root:

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace \
  weijhang/digital-human-env:ubuntu22.04-cpu-v1.0 \
  bash -lc 'cmake -S /workspace -B /workspace/.docker-build -DCMAKE_BUILD_TYPE=Release && cmake --build /workspace/.docker-build -j2'
```

Provide these local resources before submitting the `default` sample:
`face.jpg`, `test_audio_30s_sync.wav`, and `models/` containing
`wav2lip.ncnn.param`, `wav2lip.ncnn.bin`, and
`shape_predictor_68_face_landmarks.dat`. Do not commit resources or results.

Build and start the API and proxy:

```bash
docker compose up --build -d
curl http://localhost:8080/api/health
curl http://localhost:8080/api/samples
```

Health reports `renderer_present`; verify it is true. `status=ok` alone only
means the API is responding. A successful task and output quality check are
still required to validate rendering.

The stack uses `nginx -> FastAPI -> digital_human_app`. The repository is
mounted into the API container; task state and results use the `task-state`
named volume. The API only accepts the preconfigured sample and limits renderer
runtime through `DH_TASK_TIMEOUT_S` (default 300 seconds).

HTTP binds to `127.0.0.1:8080` for validation. This API has no application
authentication; a public entry point requires separate authentication setup.

Stop with `docker compose down`. Do not add `-v` if history and results must be
retained. No remote Worker claim, heartbeat or result upload endpoints exist.
