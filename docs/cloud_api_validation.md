# Cloud API validation — 2026-10-01

Workspace: `/srv/digital-human-sdk-service`, branch `codex/cloud-api-integration`,
base `b7bb4090aee0c99780b610cad2f7d1b2ac971577`.
Runtime: Ubuntu 22.04 cloud host, CPU image
`weijhang/digital-human-env:ubuntu22.04-cpu-v1.0`, Release generic x86-64,
2 build jobs; 3.6 GiB host RAM, no swap. Model/input files are local only.

## Completed checks

- Full CMake CPU build: exit 0, all configured targets built.
- API Docker image build: exit 0, pinned FastAPI/Pydantic/Uvicorn stack.
- `docker compose config --quiet`: exit 0.
- `python3 -m unittest discover -s tests -p test_server.py -v`: 4 tests passed.
  Tests cover partial timeout logs, descendant termination, restart state and
  unknown sample/task errors. Both timeout regressions failed before fixes.
- `ctest -N`: zero registered tests; this is not a C++ test pass.
- Validation API binds only `127.0.0.1:18080`; health reported
  `renderer_present=true`, default sample `available=true`.
- Real default task `b98dcadcf5bf4a4186b1f3b14741fb5c`: completed;
  status, metrics and result download endpoints checked on the cloud host.
  Pipeline timer 40009 ms, first-frame metric 195.179 ms, FFmpeg merge 1522 ms.
  These are one run, not a controlled comparative benchmark.
- Downloaded result: H.264 + AAC, 512x512, 751 video frames, 30.04 seconds,
  353729 bytes. Full FFmpeg decode succeeded.

## Limits

- Output picture and mouth motion have not been visually accepted. Export of
  a face-containing validation frame was rejected by automatic approval review;
  no image was exported. Do not equate decoding with visual quality.
- Public/local-client connectivity and authenticated deployment are unverified;
  the Compose stack itself was not started during this temporary API test.
- This branch does not implement the remote MuseTalk Worker protocol.
- Cloud direct github.com:443 access times out. Version transfer uses a verified
  Git bundle; pushing can use the local machine without changing cloud DNS.

## Final-code rerun

After process-group timeout cleanup, task
`dab5443da8f046169339aac921dc3189` completed. Pipeline timer 36733 ms,
first-frame metric 180.702 ms and FFmpeg merge 1313 ms. Its result was retrieved
through the API and fully decoded without errors. The temporary validation
container was stopped after the checks; no public service is running.
