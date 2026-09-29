#!/usr/bin/env bash
set -euo pipefail

# Run the complete offline image + audio -> mp4 pipeline and collect metrics.
# This script intentionally wraps the existing digital_human_app so benchmark
# behavior stays identical to the normal application path.

usage() {
  cat <<'EOF'
Usage:
  run_e2e_benchmark.sh [face.jpg] [audio.wav] [models_dir] [output_dir]

Environment overrides:
  BUILD_DIR   Build directory containing bin/digital_human_app (default: .docker-build)
  TIMEOUT_S   Maximum run time (default: 240)
  SAMPLE_MS   RSS polling interval (default: 100)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/.docker-build}"
APP="${APP:-${BUILD_DIR}/bin/digital_human_app}"
FACE="${1:-${ROOT_DIR}/face.jpg}"
AUDIO="${2:-${ROOT_DIR}/test_audio_30s_sync.wav}"
MODELS="${3:-${ROOT_DIR}/models}"
OUT_DIR="${4:-${BUILD_DIR}/e2e-benchmark}"
TIMEOUT_S="${TIMEOUT_S:-240}"
SAMPLE_MS="${SAMPLE_MS:-100}"

for required in "$APP" "$FACE" "$AUDIO" "$MODELS"; do
  if [[ ! -e "$required" ]]; then
    echo "benchmark_error=missing_input path=$required" >&2
    exit 2
  fi
done
if [[ ! -x "$APP" ]]; then
  echo "benchmark_error=app_not_executable path=$APP" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${OUT_DIR}/run_${RUN_ID}.log"
RSS_LOG="${OUT_DIR}/rss_${RUN_ID}.log"
RESULT="${OUT_DIR}/result_${RUN_ID}.json"
OUTPUT="${OUT_DIR}/output_${RUN_ID}.mp4"

start_ns="$(date +%s%N)"
(
  cd "$BUILD_DIR"
  exec "$APP" "$FACE" "$AUDIO" "$OUTPUT" "$MODELS"
) >"$LOG" 2>&1 &
PID=$!

peak_rss_kb=0
peak_children_rss_kb=0
timed_out=0

# Poll the application and its direct children. FFmpeg is launched near the
# end of the run; including children keeps the sampled peak representative of
# the complete process tree without requiring extra tools in the container.
while kill -0 "$PID" 2>/dev/null; do
  # The process can exit between the two ps calls (especially on a model
  # load failure). Treat that race as a zero sample instead of aborting the
  # benchmark because this script uses `set -euo pipefail`.
  rss_kb="$(ps -o rss= -p "$PID" 2>/dev/null | awk '{print $1+0}' || true)"
  child_rss_kb="$(ps --ppid "$PID" -o rss= 2>/dev/null | awk '{s+=$1} END {print s+0}' || true)"
  rss_kb="${rss_kb:-0}"
  child_rss_kb="${child_rss_kb:-0}"
  total_rss_kb=$((rss_kb + child_rss_kb))
  if (( total_rss_kb > peak_children_rss_kb )); then
    peak_children_rss_kb=$total_rss_kb
  fi
  if (( rss_kb > peak_rss_kb )); then
    peak_rss_kb=$rss_kb
  fi
  printf '%s %s %s\n' "$(date +%s%N)" "$rss_kb" "$total_rss_kb" >>"$RSS_LOG"
  elapsed_s=$(( ( $(date +%s) - ${start_ns:0:10} ) ))
  if (( elapsed_s >= TIMEOUT_S )); then
    timed_out=1
    kill -TERM "$PID" 2>/dev/null || true
    sleep 2
    kill -KILL "$PID" 2>/dev/null || true
    break
  fi
  sleep "$(awk -v ms="$SAMPLE_MS" 'BEGIN { printf "%.3f", ms / 1000.0 }')"
done

set +e
wait "$PID"
exit_code=$?
set -e
end_ns="$(date +%s%N)"

elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
peak_tree_rss_kb="$peak_children_rss_kb"

first_frame_ms="$(sed -n 's/.*First frame latency: *\([0-9.][0-9.]*\) ms.*/\1/p' "$LOG" | head -1)"
pipeline_ms="$(sed -n 's/.*全链路推理与渲染.*: *\([0-9][0-9]*\) ms.*/\1/p' "$LOG" | tail -1)"
merge_ms="$(sed -n 's/.*FFmpeg 音视频合并.*: *\([0-9][0-9]*\) ms.*/\1/p' "$LOG" | tail -1)"

# Progress lines are emitted with carriage returns. Normalize them first,
# then extract the largest observed task/render queue depth.
max_task_queue="$(tr '\r' '\n' <"$LOG" | sed -n 's/.*推理中: *\([0-9][0-9]*\) *| 渲染中: *\([0-9][0-9]*\).*/\1/p' | awk 'BEGIN{m=0} {if ($1>m)m=$1} END{print m+0}')"
max_render_queue="$(tr '\r' '\n' <"$LOG" | sed -n 's/.*推理中: *\([0-9][0-9]*\) *| 渲染中: *\([0-9][0-9]*\).*/\2/p' | awk 'BEGIN{m=0} {if ($1>m)m=$1} END{print m+0}')"

[[ -n "$first_frame_ms" ]] || first_frame_ms="null"
[[ -n "$pipeline_ms" ]] || pipeline_ms="null"
[[ -n "$merge_ms" ]] || merge_ms="null"
[[ -n "$max_task_queue" ]] || max_task_queue=0
[[ -n "$max_render_queue" ]] || max_render_queue=0

cat >"$RESULT" <<EOF
{
  "timestamp_utc": "${RUN_ID}",
  "face": "${FACE}",
  "audio": "${AUDIO}",
  "models": "${MODELS}",
  "output": "${OUTPUT}",
  "exit_code": ${exit_code},
  "timed_out": ${timed_out},
  "wall_time_ms": ${elapsed_ms},
  "first_frame_ms": ${first_frame_ms},
  "pipeline_ms": ${pipeline_ms},
  "ffmpeg_merge_ms": ${merge_ms},
  "peak_app_rss_kb": ${peak_rss_kb},
  "peak_process_tree_rss_kb": ${peak_tree_rss_kb},
  "max_task_queue": ${max_task_queue},
  "max_render_queue": ${max_render_queue},
  "log": "${LOG}",
  "rss_log": "${RSS_LOG}"
}
EOF

cat "$RESULT"
if (( exit_code != 0 || timed_out != 0 )); then
  echo "benchmark_error=run_failed result=$RESULT" >&2
  exit 1
fi
