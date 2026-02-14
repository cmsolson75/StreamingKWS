#!/bin/bash
set -e

# setup stuff
uv run --extra metrics python tools/plot_metrics.py "$1"