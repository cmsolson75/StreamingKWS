#!/bin/bash
set -e

# setup stuff
uv sync
source .venv/bin/activate
python -m src.download_dataset


# training
python -m src.train