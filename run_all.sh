#!/usr/bin/env bash

set -Eeuo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$project_dir"

if [[ -x "$project_dir/venv/bin/python" ]]; then
    python="$project_dir/venv/bin/python"
elif [[ -x "$project_dir/venv/Scripts/python.exe" ]]; then
    python="$project_dir/venv/Scripts/python.exe"
else
    printf 'LayerDivider virtual environment not found. Run the installer first.\n' >&2
    exit 1
fi

api_host="${API_HOST:-127.0.0.1}"
api_port="${API_PORT:-8000}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-127.0.0.1}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

api_pid=''
web_pid=''

cleanup() {
    local exit_code=$?
    local pid

    trap - EXIT INT TERM
    for pid in "$api_pid" "$web_pid"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    for pid in "$api_pid" "$web_pid"; do
        if [[ -n "$pid" ]]; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Starting LayerDivider API:     http://%s:%s/docs\n' "$api_host" "$api_port"
printf 'Starting LayerDivider Gradio:  http://%s:%s\n' \
    "$GRADIO_SERVER_NAME" "$GRADIO_SERVER_PORT"
printf 'Press Ctrl+C to stop both servers.\n'

"$python" -u "$project_dir/scripts/api.py" \
    --host "$api_host" --port "$api_port" &
api_pid=$!

"$python" -u "$project_dir/demo.py" "$@" &
web_pid=$!

set +e
wait -n "$api_pid" "$web_pid"
exit_code=$?
set -e

if ! kill -0 "$api_pid" 2>/dev/null; then
    printf 'The API server exited; stopping Gradio.\n' >&2
else
    printf 'The Gradio server exited; stopping the API.\n' >&2
fi

exit "$exit_code"
