#!/usr/bin/env bash

set -Eeuo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$project_dir"

is_python_312() {
    "$@" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' \
        >/dev/null 2>&1
}

python_launcher=()
if [[ -n "${PYTHON_3_12:-}" ]]; then
    python_launcher=("$PYTHON_3_12")
    if ! is_python_312 "${python_launcher[@]}"; then
        printf 'PYTHON_3_12 does not point to a working Python 3.12 interpreter.\n' >&2
        exit 1
    fi
elif command -v python3.12 >/dev/null 2>&1 && is_python_312 python3.12; then
    python_launcher=(python3.12)
elif command -v py >/dev/null 2>&1 && is_python_312 py -3.12; then
    python_launcher=(py -3.12)
elif command -v python3 >/dev/null 2>&1 && is_python_312 python3; then
    python_launcher=(python3)
elif command -v python >/dev/null 2>&1 && is_python_312 python; then
    python_launcher=(python)
else
    printf 'Python 3.12 was not found. Install it and rerun this script.\n' >&2
    exit 1
fi

if [[ ! -d "$project_dir/venv" ]]; then
    printf 'Creating Python 3.12 venv...\n'
    "${python_launcher[@]}" -m venv "$project_dir/venv"
fi

if [[ -x "$project_dir/venv/bin/python" ]]; then
    venv_python="$project_dir/venv/bin/python"
    hf_command="$project_dir/venv/bin/hf"
elif [[ -x "$project_dir/venv/Scripts/python.exe" ]]; then
    venv_python="$project_dir/venv/Scripts/python.exe"
    hf_command="$project_dir/venv/Scripts/hf.exe"
else
    printf 'The existing venv is incomplete. Remove it and rerun this script.\n' >&2
    exit 1
fi

if ! is_python_312 "$venv_python"; then
    printf 'The existing venv does not use Python 3.12. Remove it and rerun this script.\n' >&2
    exit 1
fi

printf 'Installing PyTorch with CUDA 12.8 for SAM 3...\n'
"$venv_python" -m pip install torch==2.10.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

printf 'Installing dependencies, including the official SAM 3 package...\n'
"$venv_python" -m pip install --upgrade -r "$project_dir/requirements.txt"

printf 'Install completed.\n'
printf 'Before first segmentation, accept access at:\n'
printf '  https://huggingface.co/facebook/sam3\n'
printf 'Then authenticate with:\n'
printf '  "%s" auth login\n' "$hf_command"
