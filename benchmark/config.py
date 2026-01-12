# config.py
import os
import torch


class Config:
    # --- 路径配置 ---
    # 获取当前项目根目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    # 模型路径 (可以是本地路径 models/qwen3... 或 HuggingFace Hub ID)
    MODEL_PATH = os.path.join(BASE_DIR, "models", "qwen3-ui-1.7b")

    # 测试数据路径
    DATA_PATH = os.path.join(BASE_DIR, "data", "test_benchmark.json")

    # 结果保存路径
    RESULT_DIR = os.path.join(BASE_DIR, "results")

    # --- 硬件与模型配置 ---
    # 自动检测设备 (cuda, mps, cpu)
    DEVICE = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu"

    TORCH_DTYPE = torch.float16
    LOAD_IN_4BIT = True

    # --- 推理参数配置 ---
    BATCH_SIZE = 2  # 批处理大小 (显存大可以调大)
    MAX_NEW_TOKENS = 64  # 最大生成长度
    TEMPERATURE = 0.7  # 温度 (0.0-1.0)
    TOP_P = 0.9  # 核采样

    # --- 其它 ---
    WARMUP_ROUNDS = 1  # 预热轮数（不计入统计，用于消除加载延迟）
