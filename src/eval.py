import torch
import torch
from .seed import seed_everything
from .transforms import DbMelSpec

from .train_utils import evaluate
from .dataloaders import load_dataloader
from .configs import Config
from .model import AudioClassifier
import argparse
from pathlib import Path
import json
from safetensors.torch import load_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_folder",
        "-m",
        default="configs/config.yaml",
        help="path to model folder or json pointer",
    )
    parser.add_argument("--device", "-d", type=str)
    args = parser.parse_args()

    path = Path(args.model_folder)
    if path.suffix == ".json":
        json_data = json.loads(path.read_text())["path"]
        path = path.parent / json_data
    base_path = path.parent.parent

    state_dict = load_file(path / "model.safetensors")
    json_config = base_path / "config.resolved.json"
    cfg = Config.from_json(str(json_config))
    db_mel_spec = DbMelSpec(cfg).to(cfg.device)

    model = AudioClassifier(len(cfg.subset)).to(cfg.device)
    model.load_state_dict(state_dict)

    seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    test_loader = load_dataloader(cfg, "test")
    print(evaluate(model, cfg, test_loader, db_mel_spec))
