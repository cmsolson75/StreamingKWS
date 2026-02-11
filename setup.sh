#!/bin/bash

echo "Setting up environment"
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate

# setup data
echo "Setting up data"
python -m src.download_dataset