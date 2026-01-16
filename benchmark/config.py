# config.py
import os
import torch


class Config:

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen3-1.7B-awq-gemm-4bit")

    DATA_PATH = os.path.join(BASE_DIR, "data", "test_benchmark.json")

    RESULT_DIR = os.path.join(BASE_DIR, "results")

    DEVICE = "cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu"

    TORCH_DTYPE = torch.float16
    LOAD_IN_4BIT = False

    BATCH_SIZE = 300
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.7
    TOP_P = 0.9

    WARMUP_ROUNDS = 1
