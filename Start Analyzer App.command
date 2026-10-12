#!/bin/bash
# Double-click this file in Finder to start Analyzer App in your browser.
# The first time, it installs what the app needs (a few minutes, needs internet).
# Keep the Terminal window it opens; close it to stop the app.

set -e
cd "$(dirname "$0")"

# Files downloaded from the internet are marked "quarantined" by macOS. Clear
# that once so later double-clicks open without a warning.
xattr -dr com.apple.quarantine . 2>/dev/null || true

export PATH="$HOME/.local/bin:$PATH"

if [ ! -x .venv/bin/analyzer ] || [ pyproject.toml -nt .venv/installed ]; then
    echo "Setting up Analyzer App. This takes a few minutes the first time."
    if ! command -v uv >/dev/null 2>&1; then
        echo "Installing uv (it installs Python for the app)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv venv --python 3.12 --allow-existing .venv
    uv pip install --python .venv/bin/python -e ".[feedback]"
    touch .venv/installed
    echo "Setup done."
fi

exec .venv/bin/analyzer app
