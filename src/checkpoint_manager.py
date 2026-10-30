# 学习目标：把一次实验的配置、模型参数和训练进度保存下来，以后可以查看或继续。
# 阅读路线：RunManager 建实验目录 → CheckpointManager.save 写存档 → load 恢复。
# 模型参数说明“学到了什么”；优化器等状态说明“训练进行到什么位置、怎样继续”。
from pathlib import Path
from safetensors.torch import save_file, load_file
from torch import nn
import torch
from .configs import Config
from datetime import datetime, timezone
import threading
import json
import subprocess


# ======================================================================
# 1. 管理一次实验：确定目录，保存配置，可选后台同步
# ======================================================================
class RunManager:
    def __init__(
        self,
        base_path: str,
        cfg: Config,
        tags: list,
        resume: str | None = None,
    ):
        # 输入：根目录、配置、标签和可选旧实验名；结果：self.path 指向本次实验目录。
        if resume:
            # 继续旧实验时复用其目录；split("+")[0] 去掉目录名中的附加标签。
            self.path = Path(base_path) / resume
            self.run_id = resume.split("+")[0]
        else:
            # UTC 时间 + 配置摘要组成实验名；strftime 决定时间文字格式。
            self.run_id = (
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{cfg.hash()}"
            )
            slug = "".join(f"+{tag}" for tag in tags)  # join 将多个标签片段拼成字符串。
            self.path = Path(base_path) / f"{self.run_id}{slug}"
            self.path.mkdir(parents=True, exist_ok=True)  # 创建目录，包括缺失的上级；已存在也可继续。
            cfg.to_json(self.path / "config.resolved.json")
        self.s3_bucket = cfg.env.remote_name if cfg.env.cloud_sync else None

        if self.s3_bucket:
            # Event 像一个通知开关；Thread 创建后台线程，避免主训练循环等待云上传。
            # daemon=True 表示它不会单独阻止整个程序退出。
            self._sync_pending = threading.Event()
            self._sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
            self._sync_thread.start()

    def _sync_worker(self):
        # 等待通知 → 清除通知 → 用 AWS 命令行同步；没有启用 cloud_sync 时不会启动此线程。
        while True:
            self._sync_pending.wait()
            self._sync_pending.clear()
            try:
                # subprocess.run 运行外部命令；check=True 让命令失败时抛异常，下面会显示错误。
                subprocess.run(
                    [
                        "aws",
                        "s3",
                        "sync",
                        str(self.path),
                        f"s3://{self.s3_bucket}/{self.path.name}",
                    ],
                    check=True,
                    capture_output=True,
                )
                print(f"Synced to s3://{self.s3_bucket}/{self.path.name}")
            except Exception as e:
                print(f"S3 sync failed: {e}")

    def sync(self) -> None:
        # .set() 只发出通知，不等待上传完成；正在训练的代码可以继续运行。
        if not self.s3_bucket:
            return
        self._sync_pending.set()


# ======================================================================
# 2. 写训练存档：模型文件、训练状态、最新 / 最佳指针和清单
# ======================================================================
class CheckpointManager:
    def __init__(self, run_manager: RunManager):
        self.run_manager = run_manager
        self.ckpt_dir = self.run_manager.path / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: nn.Module, step: int, best: bool, **train_state):
        # **train_state 收集额外的具名参数，例如 optimizer=optim、best_acc=0.9。
        # step:06d 把编号补齐为六位，例如 12 → step_000012，便于按名字排序。
        step_dir = self.ckpt_dir / f"step_{step:06d}"
        step_dir.mkdir(exist_ok=True)

        model_path = step_dir / "model.safetensors"
        state_path = step_dir / "train_state.pt"

        # state_dict() 取出“名称 → 参数或缓冲数组”的字典；safetensors 保存这些数组。
        save_file(model.state_dict(), model_path)
        state = {}
        for k, v in train_state.items():
            if hasattr(v, "state_dict"):  # 有状态导出方法的对象先转为字典，普通数值则直接保留。
                state[k] = v.state_dict()
            else:
                state[k] = v
        torch.save(state, state_path)  # 保存优化器等训练状态，与模型权重分开放。

        # latest.json 只记录“最新存档在哪”，不是模型参数本身；每次保存都会更新。
        (self.ckpt_dir / "latest.json").write_text(
            json.dumps({"step": step, "path": step_dir.name})
        )
        if best:
            # best.json 只在 best=True 时更新；最佳与最新可能指向不同目录。
            (self.ckpt_dir / "best.json").write_text(
                json.dumps({"step": step, "path": step_dir.name})
            )

        # manifest 是存档清单：UTC 时间、步数、文件名称和大小，便于检查保存结果。
        manifest = {
            "run_id": self.run_manager.run_id,
            "step": step,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": [
                {"path": "model.safetensors", "bytes": model_path.stat().st_size},
                {"path": "train_state.pt", "bytes": state_path.stat().st_size},
            ],
        }
        (step_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        self.run_manager.sync()

    # ------------------------------------------------------------------
    # 3. 读回进度：先找到最新目录，再将状态填回调用方提供的对象
    # ------------------------------------------------------------------
    def load_latest(self) -> tuple[Path, int] | None:
        # exists() 判断文件是否存在；json.loads 将读到的文本解析为字典。
        # 返回 (存档目录, 步数)，没有存档则返回 None。
        latest_file = self.ckpt_dir / "latest.json"
        if not latest_file.exists():
            return None
        latest = json.loads(latest_file.read_text())
        return self.ckpt_dir / latest["path"], latest["step"]

    def load(self, model: nn.Module, **train_state):
        result = self.load_latest()
        if result is None:
            return 0, 0.0, 0, {}
        step_dir, step = result
        # load_file 只读出参数字典；load_state_dict 才把参数装进现有模型。
        state_dict = load_file(step_dir / "model.safetensors")
        model.load_state_dict(state_dict)

        # 这里读取本项目训练生成的完整状态；仅向它提供可信来源的训练存档。
        saved_state = torch.load(step_dir / "train_state.pt", weights_only=False)
        for name, obj in train_state.items():
            if name in saved_state:
                obj.load_state_dict(saved_state[name])

        sampler_step = saved_state.get("sampler_step", step)
        best_acc = saved_state.get("best_acc", 0.0)
        # 只有通过 **train_state 传进来的对象会在上面自动恢复；例如想恢复 scheduler 也要传它。
        return step, best_acc, sampler_step, state_dict


# ======================================================================
# 4. 预测 → 验证 → 修改 → 自己写
# ======================================================================
# 在单独临时实验目录中，用 RunManager 和 CheckpointManager 保存一个很小的 nn.Linear。
# 预测：连续保存两次，但只有第一次 best=True，latest.json 和 best.json 分别指向哪次？
# 验证：读取两个 JSON，再用 load_latest() 对照；练习配置保持 cloud_sync=False。
# 自己写：先只保存 / 读取模型的 state_dict，再加入训练步数，最后加入优化器状态。
# 下方是原有旧调试示例；RunManager 参数数量和 save 的必填 best 参数需要对齐后才能运行。
if __name__ == "__main__":
    cfg = Config.from_yaml("configs/config.yaml")
    run_manager = RunManager(".", cfg, [], None, "20260212T214145Z_a9bbf6a412")

    ckpt_manager = CheckpointManager(run_manager)
    print(ckpt_manager.load_latest())
    model = nn.Linear(10, 10)
    optim = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda step: min((step + 1) / 10, 1)
    )
    ckpt_manager.save(model, 1, optim=optim, scheduler=scheduler)
    # x = torch.randn(1, 10)
    # y = torch.randint(0, 10, (1, 10), dtype=torch.float32)
    # epochs = 10
    # for epoch in range(epochs):
    #     logits = model(x)
    #     loss = F.cross_entropy(logits, y)

    #     optim.zero_grad()
    #     loss.backward()
    #     optim.step()
    #     scheduler.step()
    #     save_model_checkpoint(model)
    #     save_state(optim=optim, scheduler=scheduler)

    # print(optim.state_dict(), scheduler.state_dict())
