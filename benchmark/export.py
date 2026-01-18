# export.py
import os
import time
import logging
import random
from transformers import AutoTokenizer
from datasets import load_dataset
from config import Config
from awq import AutoAWQForCausalLM


class QuantizationExporter:

    def __init__(self):
        self.cfg = Config()
        model_name = os.path.basename(self.cfg.MODEL_PATH)
        self.output_dir = os.path.join(self.cfg.RESULT_DIR,
                                       f"{model_name}-awq-gemm-4bit")
        os.makedirs(self.cfg.RESULT_DIR, exist_ok=True)
        os.makedirs(self.cfg.LOG_DIR, exist_ok=True)
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger("LLM_Export")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(message)s',
                                      datefmt='%H:%M:%S')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def get_wikitext2_text(self, n_samples=128):
        self.logger.info("📥 加载校准数据...")
        try:
            data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            text_list = [x for x in data['text'] if len(x) > 50]
            if len(text_list) > n_samples:
                return random.sample(text_list, n_samples)
            return text_list
        except Exception as e:
            self.logger.error(f"❌ 数据加载失败: {e}")
            exit(1)

    def run(self):
        start_time = time.time()

        quant_config = {
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM"
        }

        self.logger.info(f"📥 加载原模型: {self.cfg.MODEL_PATH}")
        try:
            model = AutoAWQForCausalLM.from_pretrained(
                self.cfg.MODEL_PATH, **{
                    "low_cpu_mem_usage": True,
                    "use_cache": False
                })
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.MODEL_PATH,
                                                      trust_remote_code=True)
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            exit(1)

        calib_data = self.get_wikitext2_text(n_samples=128)

        self.logger.info("⚡ 开始执行 GEMM 量化...")
        model.quantize(tokenizer,
                       quant_config=quant_config,
                       calib_data=calib_data)

        self.logger.info(f"💾 保存模型到: {self.output_dir}")
        model.save_quantized(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        self.logger.info(f"🏁 完成! 耗时: {time.time() - start_time:.2f}s")
        self.logger.info(f"👉 请修改 run.py 的 MODEL_PATH 为: {self.output_dir}")


if __name__ == "__main__":
    exporter = QuantizationExporter()
    exporter.run()
