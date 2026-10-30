# 学习目标：把已训练模型接到实时音频应用，而不是只对一批文件做预测。
# 阅读路线：本文件组装工具 → inference/runner.py 处理单个窗口 → stream_app.py 管理录音和状态。
# 应用配置 InferConfig 决定用哪个存档、唤醒词和阈值；训练 Config 决定网络和频谱格式。
from .configs import InferConfig
from .inference import AudioStream, InferenceRunner
from .inference.io import load_cfg_model_state, load_labels, load_model
import argparse


# ======================================================================
# 1. 读应用参数 → 还原模型 → 创建推理工具 → 交给实时音频循环
# ======================================================================
def main() -> None:
    # ArgumentParser 读取命令行；--config 指应用配置文件，--out 可指定识别数字的保存位置。
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/infer.yaml")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    infer_cfg = InferConfig.from_yaml(args.config)
    device = infer_cfg.device
    # 项目 API：返回存档对应的训练配置和参数字典，再由 load_model 建网络并装入权重。
    cfg, state_dict = load_cfg_model_state(infer_cfg.model, device)
    model = load_model(cfg, state_dict)

    # InferenceRunner 把“统一声音 → 转频谱 → 预测 → 概率”包在一起；.to 同步内部数组的设备。
    inference_runner = InferenceRunner(model, cfg).to(device)

    labels = load_labels(infer_cfg.label_file)  # 标签列表把模型输出列号还原成词，顺序须与训练一致。
    audio_stream = AudioStream(
        config=infer_cfg,
        inference_runner=inference_runner,
        labels=labels,
        output=args.out,
    )
    audio_stream.process_audio()  # 这里才打开输入流并持续识别；Ctrl+C 结束循环。


# ======================================================================
# 2. 先理解交接关系，再练习接入实时音频
# ======================================================================
# 预测：改变 infer_cfg.threshold 会改网络权重吗？追踪它在哪个文件被读取来验证。
# 修改练习：先用随机 NumPy 波形单独调用 InferenceRunner，确认输出形状，不必开启麦克风。
# 自己写：先连好“配置 → 模型 → 单窗口预测”，再接 AudioStream 和输出文件选项。
if __name__ == "__main__":
    main()
