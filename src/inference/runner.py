import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..configs import Config
from ..transforms import AudioTransform, DbMelSpec


class InferenceRunner(nn.Module):
    def __init__(self, model: nn.Module, cfg: Config):
        super().__init__()
        self.model = model
        model.eval()

        self.cfg = cfg
        self.transform = AudioTransform(cfg)
        self.db_mel_spec = DbMelSpec(cfg, augment=False)

    @torch.inference_mode()
    def forward(self, x: np.ndarray) -> torch.Tensor:
        device = next(self.model.parameters()).device
        x = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        x = self.transform(x, self.cfg.preprocess.sample_rate)
        x = self.db_mel_spec(x)
        logits = self.model(x)
        return F.softmax(logits, dim=1)
