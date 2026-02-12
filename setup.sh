#!/bin/bash

echo "Setting up environment"

if [[ "$(uname)" == "Linux" ]]; then
    command -v ffmpeg &> /dev/null || { apt update && apt install -y ffmpeg; }
fi

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync
source .venv/bin/activate

# setup data
echo "Setting up data"
python -m src.download_dataset