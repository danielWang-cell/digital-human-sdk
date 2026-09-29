#!/usr/bin/env bash
set -e

CONTAINER_NAME="digital-human-sdk-realtime"
IMAGE_NAME="digital-human-sdk-stable:latest"
PROJECT_DIR="$HOME/home/digital-human-sdk"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[INFO] Container ${CONTAINER_NAME} exists."
    echo "[INFO] Starting and attaching..."
    docker start -ai ${CONTAINER_NAME}
else
    echo "[INFO] Creating container ${CONTAINER_NAME}..."

    docker run -it \
      --name ${CONTAINER_NAME} \
      -e PULSE_SERVER=unix:/mnt/wslg/PulseServer \
      -e DISPLAY=${DISPLAY} \
      -e WAYLAND_DISPLAY=${WAYLAND_DISPLAY} \
      -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir \
      -v /mnt/wslg:/mnt/wslg \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -v ${PROJECT_DIR}:/workspace \
      -w /workspace \
      ${IMAGE_NAME} \
      /bin/bash
fi
