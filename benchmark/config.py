# config.py
import os
import torch


class Config:

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    MODEL_PATH = os.path.join(BASE_DIR, "models", "qwen3-ui-1.7b")

    DATA_PATH = os.path.join(BASE_DIR, "data", "test_benchmark.json")

    RESULT_DIR = os.path.join(BASE_DIR, "results")

    DEVICE = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu"

    TORCH_DTYPE = torch.float16
    LOAD_IN_4BIT = True

    BATCH_SIZE = 220
    MAX_NEW_TOKENS = 64
    TEMPERATURE = 0.7
    TOP_P = 0.9

    WARMUP_ROUNDS = 1
