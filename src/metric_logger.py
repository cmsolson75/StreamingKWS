from pathlib import Path
import json


class JSONLMetricLogger:
    def __init__(self, path: Path):
        self.jsonl_path = path / "metrics.jsonl"

    def log(self, data: dict) -> None:
        with self.jsonl_path.open(mode="a", encoding="utf-8") as f:
            json_line = json.dumps(data)
            f.write(json_line + "\n")


# if __name__ == "__main__":
#     metrics_logger = JSONLMetricLogger(Path("."))

#     data = {"loss": 10.2, "step": 200, "acc": 20}
#     metrics_logger.log(data)


#     data = {"loss": 9.2, "step": 300, "acc": 20}
#     metrics_logger.log(data)
