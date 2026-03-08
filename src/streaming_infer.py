from .configs import InferConfig
from .inference import AudioStream, InferenceRunner
from .inference.io import load_cfg_model_state, load_labels, load_model
import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/infer.yaml")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    infer_cfg = InferConfig.from_yaml(args.config)
    device = infer_cfg.device
    cfg, state_dict = load_cfg_model_state(infer_cfg.model, device)
    model = load_model(cfg, state_dict)

    inference_runner = InferenceRunner(model, cfg).to(device)

    labels = load_labels(infer_cfg.label_file)
    audio_stream = AudioStream(
        config=infer_cfg,
        inference_runner=inference_runner,
        labels=labels,
        output=args.out,
    )
    audio_stream.process_audio()


if __name__ == "__main__":
    main()
