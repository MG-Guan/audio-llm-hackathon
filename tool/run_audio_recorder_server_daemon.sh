#!/usr/bin/env bash

# Simple watchdog script that keeps the audio recorder server alive.
# Usage:
#   ./run_audio_recorder_server_daemon.sh
# Optional environment variables:
#   PYTHON_BIN        Python executable to use (default: python3)
#   UVICORN_HOST      Host binding (default: 0.0.0.0)
#   UVICORN_PORT      Port to expose (default: 9000)
#   UVICORN_EXTRA     Additional uvicorn CLI args
#   RESTART_DELAY     Seconds to wait before relaunch (default: 1)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_DIR="${PROJECT_ROOT}/audio_recorder_server"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/audio_recorder_server.log"

PYTHON_BIN="${PYTHON_BIN:-python3}"
UVICORN_HOST="${UVICORN_HOST:-0.0.0.0}"
UVICORN_PORT="${UVICORN_PORT:-9000}"
UVICORN_EXTRA="${UVICORN_EXTRA:-}"
RESTART_DELAY="${RESTART_DELAY:-1}"


mkdir -p "${LOG_DIR}"

stop_requested=0
trap 'stop_requested=1' INT TERM

launch_server() {
    export BOSON_API_KEY=bai-QCxvolzHJBYeF8pGB8uOpYlkLOg_Xbt7j4hxNzW1aydeAtTq
    cd "${APP_DIR}"
    "${PYTHON_BIN}" main.py ${UVICORN_EXTRA}
}

while true; do
    if [[ "${stop_requested}" -eq 1 ]]; then
        echo "$(date -Is) :: Stop requested. Exiting watchdog loop." | tee -a "${LOG_FILE}"
        break
    fi

    echo "$(date -Is) :: Starting audio recorder server..." | tee -a "${LOG_FILE}"
    if launch_server >>"${LOG_FILE}" 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi

    echo "$(date -Is) :: Server exited with code ${exit_code}." | tee -a "${LOG_FILE}"

    if [[ "${stop_requested}" -eq 1 ]]; then
        echo "$(date -Is) :: Stop requested after exit. Leaving." | tee -a "${LOG_FILE}"
        break
    fi

    echo "$(date -Is) :: Relaunching in ${RESTART_DELAY}s..." | tee -a "${LOG_FILE}"
    sleep "${RESTART_DELAY}"
done

exit 0
