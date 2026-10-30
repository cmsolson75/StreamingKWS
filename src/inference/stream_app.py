# 学习目标：把“模型认出的词”变成有顺序的应用行为：等唤醒 → 收集数字 → 停止并输出。
# 阅读路线：先看两个 _handle 方法理解规则，再看 detect_keyword，最后看 process_audio。
# 数据流：麦克风小片段 → 环形缓冲区 → 最近一段声音 → 类别概率 → 标签 → 状态变化。
from enum import Enum, auto
import time

import numpy as np
import sounddevice as sd
import torch

from ..configs import InferConfig
from .buffer import CircularBuffer
from .runner import InferenceRunner


# ======================================================================
# 1. 定义状态：同一个词在不同状态下可以触发不同动作
# ======================================================================
# Enum 让状态有明确名字；auto() 自动生成不同的内部值，不需要手动记数字。
class AppState(Enum):
    IDLE = auto()
    LISTENING = auto()


# ======================================================================
# 2. 创建应用：记住配置、预测工具、标签和当前收集的数字
# ======================================================================
class AudioStream:
    def __init__(
        self,
        config: InferConfig,
        inference_runner: InferenceRunner,
        labels: list,
        output: str | None = None,
    ):
        self.cfg = config
        self.buffer = CircularBuffer(self.cfg.buffer_size)  # 默认按最新约一秒声音进行识别。
        self.inference_runner = inference_runner
        self.labels = labels

        # time.time() 给出当前时间读数；cooldown 存“最早何时允许下次触发”。
        self.cooldown = time.time()
        self.state = AppState.IDLE
        self.numbers = []
        self.output_path = output

    # ---------- 2.1 输出结果：没有输出路径或没有数字时不写文件 ----------
    def write_file(self) -> None:
        if self.output_path is None or not self.numbers:
            return
        # "a" 是追加；join 把 1、2、3 连成 "123"，每次停止后追加一行。
        with open(self.output_path, "a") as f:
            f.write("".join(str(num) for num in self.numbers) + "\n")

    # ---------- 2.2 冷却：避免相邻且重叠的音频窗口反复触发同一次发音 ----------
    def can_trigger(self):
        return time.time() >= self.cooldown

    def update_cooldown(self):
        self.cooldown = time.time() + self.cfg.cooldown_time

    # ---------- 2.3 接音频：回调只搬数据，较慢的模型预测放到主循环 ----------
    def callback(self, indata, frames, time, status):
        # sounddevice 会自动调用这个方法；indata 是 [本次采样点数, 声道数] 的 NumPy 数组。
        # frames 是本次帧数，time 是设备时间信息，status 表示录音状态；此处只用状态和数据。
        if status:
            print(status)
        self.buffer.put_many(indata[:, 0])  # [:, 0] 保留全部时刻，只取第一个声道。

    # ---------- 2.4 认词：最高概率超过阈值且已过冷却时间才接受 ----------
    def detect_keyword(self, data: np.ndarray) -> str | None:
        # 输入是一段声音 [N]；runner 返回 [1, 类别数]，squeeze(0) 去掉单样本维度。
        probs = self.inference_runner(data).squeeze(0)
        max_idx = torch.argmax(probs)  # argmax 取最大值的位置，不是概率本身。
        if probs[max_idx] > self.cfg.threshold and self.can_trigger():
            self.update_cooldown()
            # item() 将单个 Tensor 变普通整数，再用标签表把编号还原成词。
            return self.labels[max_idx.item()]
        return None

    # ---------- 2.5 空闲状态：只响应唤醒词，其他词忽略 ----------
    def _handle_idle(self, label: str):
        if label == self.cfg.wakeword:
            self.state = AppState.LISTENING
            print("Listening")

    # ---------- 2.6 聆听状态：记录数字，收到停止词就输出并清空 ----------
    def _handle_listening(self, label: str):
        if label == self.cfg.stopword:
            self.state = AppState.IDLE
            print("Stopping...")
            if self.numbers:
                str_out = "".join(str(num) for num in self.numbers)
                print(f"Recorded Numbers: {str_out}")
            self.write_file()
            self.numbers = []
            return

        num = self.cfg.number_map.get(label)  # 字典 get 查不到时返回 None；不在映射里的词不记入。
        if num is not None:
            self.numbers.append(num)
            print(f"Number: {num}")

    # ------------------------------------------------------------------
    # 3. 开输入流：回调持续收音，主循环每隔一小段时间检查最新窗口
    # ------------------------------------------------------------------
    def process_audio(self):
        # InputStream 创建麦克风输入流；with 进入时启动、退出时停止并释放资源。
        # samplerate 是每秒采样点数，channels 是声道数，callback 是接数据的方法。
        # dtype="float32" 是样本数值类型；blocksize=0 让音频后端决定每次回调的数据量。
        with sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.cfg.channels,
            callback=self.callback,
            dtype="float32",
            blocksize=0,
        ):
            try:
                while True:
                    time.sleep(0.2)  # 暂停约 0.2 秒，避免空转；这不是重新录制一段声音。
                    data = self.buffer.get()
                    if len(data) < self.cfg.buffer_size:  # 启动初期还没攒够一个窗口，先等待。
                        continue
                    label = self.detect_keyword(data)
                    if label is None:
                        continue

                    # 把已识别的词交给当前状态对应的处理方法；两种状态不会同时处理同一个词。
                    if self.state == AppState.IDLE:
                        self._handle_idle(label)
                    elif self.state == AppState.LISTENING:
                        self._handle_listening(label)
            except KeyboardInterrupt:  # Ctrl+C 触发，退出循环后由 with 关闭输入流。
                print("Stopped")


# 学习练习：先验证规则，再接真实麦克风，不必一开始同时调试录音与模型。
# 预测：依次收到“数字 → 唤醒词 → one → two → 停止词”，会收集哪几个数字、最后是什么状态？
# 验证：在单独脚本中创建 app（output=None），手动调用 _handle_idle / _handle_listening，
# 查看 app.state 和 app.numbers；手动调用是在测试规则，会绕开概率阈值和冷却检查。
# 修改：在配置副本中更换 wakeword，再检查旧唤醒词是否还会进入 LISTENING。
# 自己写：先只写两状态和数字列表，再加概率阈值、冷却，最后接回调与缓冲区。
