# Streaming Noisy KWS

Keyword spotting + streaming inference for Speech Commands.

The training pipeline learns a classifier over:
- selected keyword classes (from `configs/config.yaml`)
- `<UNKNOWN>` (all non-selected Speech Commands labels)
- `<SILENCE>` (synthetic silence + background-noise clips)

The streaming app listens on microphone input, waits for a wakeword (`marvin`), collects spoken digits, and stops on `stop`.

## Features

- Config-driven training (`configs/config.yaml`) with command-line overrides
- Weighted sampling across keyword / OOV / silence / background
- Audio augmentation (noise mix-in, gain) + spectrogram masking
- EMA model tracking during training
- Checkpointing with `latest.json` / `best.json` pointers
- Optional S3 sync for run directories
- Real-time streaming inference state machine

## Repository Layout

```text
.
├── configs/
│   ├── config.yaml         # train/data/model config
│   └── infer.yaml          # streaming inference config
├── scripts/
│   ├── setup.sh            # install deps + download dataset
│   ├── train.sh            # example training entrypoint
│   └── sync_checkpoint.sh  # pull run artifacts from S3
├── src/
│   ├── train.py            # training launcher
│   ├── train_utils.py      # train/eval loop
│   ├── eval.py             # checkpoint evaluation CLI
│   ├── streaming_infer.py  # microphone streaming inference
│   ├── inference/          # inference package (runner, stream app, I/O)
│   ├── dataset.py          # Speech Commands + OOV/silence/background datasets
│   ├── dataloaders.py      # sampler + DataLoader assembly
│   ├── model.py            # cnn + tc_resnet8 models
│   ├── transforms.py       # waveform + mel preprocessing
│   └── ...
├── runs/                   # generated training runs
└── labels.json             # class labels written by dataset loader
```

## Quickstart

### 1. Install dependencies

```bash
uv sync
```

Or run the helper:

```bash
bash scripts/setup.sh
```

### 2. Download dataset

```bash
uv run python -m src.download_dataset
```

This downloads Speech Commands into:
`data/SpeechCommands/speech_commands_v0.02`

## Training

Run training with YAML config:

```bash
uv run python -m src.train --config configs/config.yaml
```

Override values at runtime (dot paths):

```bash
uv run python -m src.train --config configs/config.yaml train.max_steps=30000 train.device=cuda env.cloud_sync=false
```

Add run tags:

```bash
uv run python -m src.train --config configs/config.yaml --tags baseline tcresnet
```

## Run Artifacts

Each run is created under `runs/<timestamp>_<config_hash>[+tags]`.

Typical contents:

```text
runs/<run_id>/
├── config.resolved.json
├── provenance.json
├── overwrites.txt
├── metrics.jsonl
└── checkpoints/
    ├── latest.json
    ├── best.json
    └── step_000200/
        ├── model.safetensors
        ├── train_state.pt
        └── manifest.json
```

To resume, set `env.resume` in config (or override with `env.resume=<run_folder_name>`).

Example resume run:

```bash
uv run python -m src.train --config configs/config.yaml env.resume=20260307T021240Z_d065b378dc
```

## Evaluation

Evaluate any saved checkpoint pointer:

```bash
uv run python -m src.eval -m runs/<run_id>/checkpoints/best.json --split test --device cpu
```

Supported splits:
- `test`
- `val`
- `oov`
- `kw`
- `silence`
- `background`

## Streaming Inference

1. Set model pointer in `configs/infer.yaml` to a checkpoint pointer JSON, for example:
   `runs/<run_id>/checkpoints/best.json`
2. Confirm `label_file` points to `labels.json`.
3. Run:

```bash
uv run python -m src.streaming_infer
```

Useful flags:

```bash
uv run python -m src.streaming_infer --config configs/infer.yaml
uv run python -m src.streaming_infer --config configs/infer.yaml --out out.txt
```

Default behavior:
- Say `marvin` to enter listening mode
- Say digits (`one` ... `nine`) to collect numbers
- Say `stop` to print `Recorded Numbers: <digits>` and return to idle
- If `--out <path>` is set, the digit sequence is appended as one line per capture

## Config Notes

- `data.subset`: keywords to classify directly
- `train.device`: `auto`, `cpu`, `cuda`, or `mps`
- `model.name`: model architecture (`cnn` or `tc_resnet8`)
- `sampler.*_weight`: must sum to `1.0`
- `augment.use_augmentations`: enables waveform + spectrogram augmentation
- `env.cloud_sync`: enables background S3 sync via `env.remote_name`

## Model Options

Current supported model names (set in `configs/config.yaml` under `model.name`):

- `tc_resnet8` (default): temporal-convolution residual model over mel frames; usually the stronger baseline in this repo.
- `cnn`: simpler 2D conv baseline over spectrogram input; useful for quick comparisons and debugging.

How to switch:

```yaml
model:
  name: tc_resnet8
```

or

```yaml
model:
  name: cnn
```

You can also override at launch:

```bash
uv run python -m src.train --config configs/config.yaml model.name=cnn
```

## Dev Commands

```bash
uv run ruff check src
uv run mypy src
```
