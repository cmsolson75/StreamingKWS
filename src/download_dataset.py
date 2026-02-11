import torchaudio
from pathlib import Path


if __name__ == "__main__":
    data_path = Path.cwd() / "data"
    data_path.mkdir(exist_ok=True)
    torchaudio.datasets.SPEECHCOMMANDS(root=data_path, download=True)
