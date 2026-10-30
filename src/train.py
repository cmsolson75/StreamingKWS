# 学习目标：理解一次训练需要哪些工具，以及应该按什么顺序把它们接起来。
# 第一次先读 launch 的 3.1、3.4、3.5、3.7、3.8，抓住“配置 → 模型和数据 → 更新参数”。
# 再读存档、日志和设备设置；最后去 train_utils.py 看真正的训练循环。
# 本文件主要负责准备工具，创建模型或优化器本身不会让模型学会识别。
# 读到“预测 / 验证”先自己判断；底部“自己写”用来检查能否从零组织训练流程。
# from .xxx 导入本项目的代码；torch、EMA 等来自安装的第三方库。
import torch
from .seed import seed_everything
from .train_utils import train, evaluate
from .dataloaders import (
    load_dataloader,
    load_oov_loader,
    load_silence_loader,
    load_speech_cmds,
    load_background_loader,
)
from .configs import Config
from .model import model_factory
from .transforms import DbMelSpec
from .metric_logger import JSONLMetricLogger
import time
import argparse
import json
from .checkpoint_manager import CheckpointManager, RunManager
from ema_pytorch import EMA

import subprocess
import platform
import getpass


# ======================================================================
# 1. 记录实验背景：以后能查到这次训练用了哪份代码、哪台机器
# ======================================================================
def get_provenance(cfg: Config) -> dict:
    # 子目标：留下能追溯本次实验的背景，方便以后比较不同训练结果。
    # 输入 cfg，输出包含代码版本、机器和设备信息的字典；-> dict 是返回类型提示。
    # subprocess.run([...]) 运行外部命令；capture_output=True 收集输出，text=True 用文本返回。
    # .stdout 取正常输出；.strip() 去掉首尾空白；git_dirty 表示是否有未提交的改动。
    prov = {
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "git_dirty": subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout
        != "",
        # platform 读取操作系统、机器架构和主机名；getpass.getuser() 读取当前用户名。
        "system": platform.system(),
        "machine": platform.machine(),
        "user": getpass.getuser(),
        "hostname": platform.node(),
        "seed": cfg.env.seed,
    }

    # is_available() 检查硬件后端是否可用；这里再结合配置，记录本次使用的设备。
    # cuda 通常指 NVIDIA GPU，mps 是 Apple GPU 后端，cpu 是中央处理器。
    prov["accelerator"] = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available() and cfg.train.device == "cuda"
        else "MPS"
        if torch.backends.mps.is_available() and cfg.train.device == "mps"
        else "CPU"
    )

    return prov


# ======================================================================
# 2. 准备学习率计划：先预热，再进入余弦阶段
# ======================================================================
# optimizer（优化器）负责改模型参数；学习率决定每次改动的步子有多大。
# scheduler（调度器）安排学习率；这里创建计划，训练中反复调用 .step() 来推进。
def get_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config
) -> torch.optim.lr_scheduler.LRScheduler:
    # 输入：已创建的 optimizer 和配置；输出：先预热、再按余弦曲线变化的计划。
    # 设计意图：先用小步子开始，再通过余弦阶段调整到 eta_min。
    # LambdaLR 用函数返回的倍率乘初始学习率；lambda step: ... 是一个简短函数。
    # 例如 warmup_steps=1000，倍率从 1/1000 逐渐升到 1；min(..., 1.0) 限制最高倍率。
    # 预测：初始 lr=0.003 时，step=0 / 499 / 999 对应的预热学习率各是多少？
    # 验证：单独计算 0.003 * min((step + 1) / 1000, 1.0)，不需要启动训练。
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: min((step + 1) / cfg.train.warmup_steps, 1.0),
    )

    # CosineAnnealingLR 按余弦曲线调整学习率，起始值高于 eta_min 时会平滑减小。
    # T_max 是这一阶段的步数；eta_min=0.001 是最低学习率，不是倍率。
    # 当前 config.yaml 的初始学习率也是 0.001，因此该阶段保持不变，并不会继续衰减。
    cosine_decay_steps = cfg.train.max_steps - cfg.train.warmup_steps
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=cosine_decay_steps, eta_min=0.001
    )

    # SequentialLR 把两个计划接起来；到 milestones 指定的步数后切换到下一段。
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.train.warmup_steps],
    )
    return scheduler


# ======================================================================
# 3. 启动训练：按下面的顺序准备各个部分
# ======================================================================
def launch():
    # 下面按工具之间的依赖顺序阅读：先有模型参数，才能创建管理这些参数的优化器。
    # ------------------------------------------------------------------
    # 3.1 读取命令行：让启动命令决定配置文件和实验标签
    # ------------------------------------------------------------------
    # 子目标：把启动命令中的文字，变成程序可以访问的参数和 cfg。
    # API：ArgumentParser 创建读取器；add_argument 声明支持哪些参数。
    # 例如 python -m src.train -c configs/config.yaml --tags first_run
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", default="configs/config.yaml")
    # --config 和 -c 是同一个参数的两种写法；default 是没传时使用的值。
    parser.add_argument("--json_config", "-j", required=False)
    # required=False 表示可不传；nargs="+" 允许 --tags 后跟多个标签。
    # metavar / help 只决定帮助信息怎样显示，不改变训练过程。
    parser.add_argument(
        "--tags",
        "-t",
        nargs="+",
        metavar="TAG",
        help="List of tags (space-separated). E.g, --tags tag1 tag2 tag3",
        default="",
    )

    # API：parse_known_args() 返回 (已声明的参数, 剩余文字列表)。
    # 剩余文字交给本项目的 with_overrides，用来覆盖配置，例如 train.batch_size=32。
    # 预测：单独传 -c configs/config.yaml train.batch_size=32，哪部分进入 overwrites？
    # 验证：在下一行之后设断点，查看 args、overwrites，再查看 cfg.train.batch_size。
    args, overwrites = parser.parse_known_args()
    if args.json_config is not None:
        # from_json / from_yaml 读取配置；with_overrides 再应用命令行中的修改。
        cfg = Config.from_json(args.json_config).with_overrides(overwrites)
    else:
        cfg = Config.from_yaml(args.config).with_overrides(overwrites)

    # ------------------------------------------------------------------
    # 3.2 准备实验文件夹、存档工具和日志工具
    # ------------------------------------------------------------------
    # 为什么提前建目录：后面的配置、实验背景、指标和模型存档都需要保存位置。
    # 项目 API：RunManager 返回目录管理对象，resume 可指定要继续的旧实验。
    # CheckpointManager 读写训练存档（checkpoint）；JSONLMetricLogger 每行记录一次指标。
    run_manager = RunManager("runs", cfg, tags=args.tags, resume=cfg.env.resume)
    ckpt = CheckpointManager(run_manager)
    metrics_logger = JSONLMetricLogger(run_manager.path)

    # 路径对象用 / 拼接文件名；open(..., "w") 写文件，with 结束后自动关闭。
    # json.dump(内容, 文件, indent=2) 写成带缩进的 JSON，方便以后查看。
    with open(run_manager.path / "provenance.json", "w") as f:
        json.dump(get_provenance(cfg), f, indent=2)
    with open(run_manager.path / "overwrites.txt", "w") as f:
        for overwrite in overwrites:
            f.write(overwrite + "\n")
    print(f"Config: {cfg.model_dump_json(indent=4)}")  # 将最终生效的配置显示出来。

    # ------------------------------------------------------------------
    # 3.3 设置随机数和计算方式
    # ------------------------------------------------------------------
    # 固定随机数起点，让实验更容易复现；不保证不同设备上结果完全一致。
    seed_everything(cfg.env.seed)

    # 为什么先设随机数：后面模型的初始参数和随机抽样都受随机操作影响。
    # 以下是速度相关设置，第二遍阅读时再研究即可。
    # "high" 和 allow_tf32 允许受支持的 GPU 用较低的内部精度加速部分运算。
    # benchmark 会尝试选择较快的卷积算法，固定输入大小时通常更适用。
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------
    # 3.4 准备输入转换和模型：音频 → 频谱 → 各类别分数
    # ------------------------------------------------------------------
    # 子目标：让加载器提供的声音数组，能变成模型需要的频谱输入。
    # 数据流：声音 [B, 1, N] → DbMelSpec → 频谱 [B, 1, F, T] → model → 分数 [B, 类别数]。
    # N=声音采样点数，F=频率格数，T=时间格数；B=这一批样本数。
    # 项目 API：DbMelSpec(cfg, 是否增强) 创建“把声音转换成频谱”的工具。
    # .to(device) 将内部数组或模型参数放到指定 CPU / GPU；输入数据也要放到同一设备。
    db_mel_spec = DbMelSpec(
        cfg,  # cfg 提供采样率、频率格数等转换设置
        cfg.augment.use_augmentations  # 是否对频谱做随机遮挡增强
    ).to(cfg.train.device)
    # 为什么 +2：除关键词外，还要输出 UNKNOWN（非目标词）、SILENCE（静音和背景噪声）的分数。
    num_classes = len(cfg.data.subset) + 2
    # API：model_factory(名字, 类别数, Dropout 比例) 返回 model.py 中对应的模型对象。
    # 预测：增加一个关键词后，模型输出的行数和列数分别会怎样变化？
    model = model_factory(
        cfg.model.name,  # 选用 "cnn" 或 "tc_resnet8"
        num_classes,  # 每条音频输出多少个类别分数
        cfg.train.dropout  # 训练时随机置零部分特征，减少死记训练数据
    ).to(cfg.train.device)
    # parameters() 取出可管理的参数；numel() 数数组里有多少个数；sum(...) 求总和。
    print(f"Total Model Params: {sum([p.numel() for p in model.parameters()])}")

    # ------------------------------------------------------------------
    # 3.5 准备参数平均、数值保护、优化器和学习率计划
    # ------------------------------------------------------------------
    # 先分清用途：model 做预测，optim 改参数，scheduler 改学习率，scaler 帮助数值稳定。
    # EMA(model, ...) 额外维护一份平滑后的参数，减少单次更新的波动，用于训练期间的验证。
    # 为什么不是“自动找最佳模型”：最佳存档由训练循环根据验证准确率选择。
    ema_model = EMA(
        model,
        beta=0.999,  # 越接近 1，后期平均参数变化越慢
        power=3 / 4,  # 控制早期平均速度逐渐变化的曲线
        update_every=1,  # 每次调用 update() 都进行更新检查
        update_after_step=100,  # 前期先跟随原模型，过此阶段再进行平滑平均
        include_online_model=False,  # 不把原模型注册成 EMA 对象的子模块
    )
    # AMP 用不同精度计算以节省资源；GradScaler 缩放梯度，减少小数值丢失的问题。
    # 梯度表示“参数往哪个方向改能减小误差”；enabled=False 时这个缩放工具不生效。
    scaler = torch.amp.GradScaler(cfg.train.device, enabled=cfg.train.amp)

    # API：AdamW(model.parameters(), lr=...) 接收要学习的参数和更新步幅。
    # 结果是优化器对象；创建它还没有更新参数，训练中调用 .step() 才会更新。
    # 为什么传 model.parameters()：让优化器知道该改的是哪个模型中的数值。
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)

    scheduler = get_scheduler(optim, cfg)  # 创建前面定义的两阶段学习率计划。

    # ------------------------------------------------------------------
    # 3.6 尝试读存档：有旧进度就继续，没有就从头开始
    # ------------------------------------------------------------------
    # 项目 API：ckpt.load(...) 把存档中的模型、scaler 和优化器状态填回对应对象。
    # 返回：训练步数、历史最高准确率、已取样本数、模型参数字典。
    # 为什么两个 step：训练按“批”计数，采样器跳过数据按“单个样本”计数。
    # 预测：批大小为 32，若已有 10 批，恢复抽样时应该跳过多少个样本？
    start_step, best_acc, sampler_step, state_dict = ckpt.load(
        model, scaler=scaler, optimizer=optim
    )
    if start_step > 0:
        print(f"Resumed from step {start_step}")

    # ------------------------------------------------------------------
    # 3.7 准备数据加载器：每次给训练或评估提供一批音频和标签
    # ------------------------------------------------------------------
    # 子目标：分别准备“拿来学的数据”和“检查学得如何的数据”。
    # 项目 API：load_dataloader(cfg, split, ...) 返回每次提供 (一批声音, 一批标签) 的加载器。
    # 为什么还有单类加载器：总准确率之外，还要知道哪一类容易出错。
    train_loader = load_dataloader(cfg, "train", start_step=sampler_step)
    val_loader = load_dataloader(cfg, "val")
    oov_loader = load_oov_loader(cfg, "val")
    sc_loader = load_speech_cmds(cfg, "val")
    silence_loader = load_silence_loader(cfg)
    background_loader = load_background_loader(cfg)

    # 列表中每项是 (显示名称, 加载器)，后面可以用一个循环逐类评估。
    eval_loaders = [
        ("oov", oov_loader),
        ("kw", sc_loader),
        ("silence", silence_loader),
        ("background", background_loader),
    ]

    # ------------------------------------------------------------------
    # 3.8 开始训练：具体的取数据、预测、算误差和更新参数都在 train_utils.py
    # ------------------------------------------------------------------
    # perf_counter() 给出计时读数；结束读数减开始读数就是经过的秒数。
    start = time.perf_counter()
    # 项目 API：train(...) 接收工具和加载器，执行循环；eval_period / log_period 控制检查间隔。
    # 在 train_utils.training_step 中追踪一批数据：
    # 取声音和标签 → 转频谱 → 模型给分 → loss 衡量误差 → backward 计算梯度 → 更新参数。
    # 梯度提供每个参数该怎样调整的信息；模型正是通过反复更新这些参数来学习。
    # 预测 / 验证：去该函数找出“真正更新参数”和“推进学习率计划”的调用各是哪一行。
    train(
        model,
        ema_model,
        optim,
        scaler,
        scheduler,
        cfg,
        train_loader,
        val_loader,
        eval_loaders,
        cfg.train.eval_period,
        cfg.train.log_period,
        db_mel_spec,
        checkpoint_manager=ckpt,
        start_step=start_step,
        metric_logger=metrics_logger,
        best_acc=best_acc,
    )

    # ------------------------------------------------------------------
    # 3.9 显示速度，再用测试数据检查当前模型
    # ------------------------------------------------------------------
    stop = time.perf_counter()
    total_time = stop - start
    # 以下按配置的 max_steps 估算速度；恢复训练时不代表本次实际执行步数。
    avg_wall_seconds_per_step = total_time / cfg.train.max_steps
    steps_per_second = cfg.train.max_steps / total_time
    samples_per_second = steps_per_second * cfg.train.batch_size

    # f"{数值:.2f}" 把数值显示成两位小数；samples/s 是每秒处理的样本数。
    print(
        f"Total wall Time: {total_time:.2f} seconds | "
        f"Throughput: {samples_per_second:.2f} samples/s | "
        f"Latency: {avg_wall_seconds_per_step:.4f} s/step"
    )
    test_loader = load_dataloader(cfg, "test")
    # API：evaluate(model, cfg, loader, 转换工具) 返回 (平均损失, 准确率)。
    # 为什么还要测试：用没有参与训练的那部分文件检查表现；这里不会更新模型参数。
    # 这里传入当前 model，没有重新读取最佳存档，也没有改用 EMA 的平均版本。
    print(evaluate(model, cfg, test_loader, db_mel_spec))


# ======================================================================
# 4. 自己写：从最少步骤搭起训练入口，再逐项添加辅助功能
# ======================================================================
# 在单独练习文件中，先按“读配置 → 建模型和输入转换 → 建加载器 → 建优化器”复写。
# 再从 train_utils.py 追踪一批数据，解释为什么必须先算误差和梯度，才能更新参数。
# 能讲清这条流程后，再补 scheduler、日志、EMA 和存档，不必一开始记住所有参数。
# 下面是原有启动入口：作为程序运行时启动训练，被其他文件 import 时不会自动训练。
if __name__ == "__main__":
    launch()
