from pathlib import Path
import json


class JSONLMetricLogger:
    def __init__(self, path: Path):
        self.jsonl_path = path / "metrics.jsonl"

    def log(self, data: dict) -> None:
        with self.jsonl_path.open(mode="a", encoding="utf-8") as f:
            json_line = json.dumps(data)
            f.write(json_line + "\n")

