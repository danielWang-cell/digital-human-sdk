#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/.docker-build"
MODEL_DIR="${1:-${ROOT_DIR}/models}"
ITERATIONS="${2:-50}"

if [[ ! -x "${BUILD_DIR}/bin/inference_benchmark" ]]; then
  echo "Build first in the SDK Docker container: ${BUILD_DIR}/bin/inference_benchmark" >&2
  exit 2
fi

export LD_LIBRARY_PATH="${BUILD_DIR}/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
exec "${BUILD_DIR}/bin/inference_benchmark" "${MODEL_DIR}" "${ITERATIONS}"
