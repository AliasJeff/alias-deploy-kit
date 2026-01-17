# config.py
import os
import torch


class Config:

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen3-1.7B")

    DATA_PATH = os.path.join(BASE_DIR, "data", "test_benchmark.json")

    RESULT_DIR = os.path.join(BASE_DIR, "results")

    DEVICE = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu"

    TORCH_DTYPE = torch.float16
    LOAD_IN_4BIT = False

    BATCH_SIZE = 100
    MAX_NEW_TOKENS = 64
    TEMPERATURE = 0.7
    TOP_P = 0.9

    WARMUP_ROUNDS = 1

    SYSTEM_INSTRUCTIONS = "只输出适配手机端的html代码，输出最小可行的html，限制200token，不要输出任何其他内容。 </no_think>"
