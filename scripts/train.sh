#!/bin/bash
set -e

# setup stuff
uv sync
source .venv/bin/activate
python -m src.download_dataset

CLOUD_SYNC=false
# training
python -m src.train cloud_sync="$CLOUD_SYNC"