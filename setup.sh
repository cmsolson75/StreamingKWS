#!/bin/bash

echo "Setting up environment"

if [[ "$(uname)" == "Linux" ]]; then
    command -v ffmpeg &> /dev/null || { apt update && apt install -y ffmpeg; }
    command -v unzip &> /dev/null || { apt update && apt install -y unzip; }
    command -v tmux &> /dev/null || { apt update && apt install -y tmux; }
    command -v nvtop &> /dev/null || { apt update && apt install -y nvtop; }

    if ! command -v aws &> /dev/null; then
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
        unzip -q /tmp/awscliv2.zip -d /tmp
        if [[ $EUID -eq 0 ]]; then
            /tmp/aws/install
        else
            sudo /tmp/aws/install
        fi
        rm -rf /tmp/aws /tmp/awscliv2.zip
    fi
fi

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi


source "$HOME/.cargo/env" 2>/dev/null || source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"

uv sync
source .venv/bin/activate

# setup data
echo "Setting up data"
python -m src.download_dataset