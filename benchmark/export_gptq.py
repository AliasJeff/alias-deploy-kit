import os
import time
import logging
from datasets import load_dataset
from config import Config
from gptqmodel import GPTQModel, QuantizeConfig


class QuantizationExporter:

    def __init__(self):
        self.cfg = Config()
        model_name = os.path.basename(self.cfg.MODEL_PATH)
        self.output_dir = os.path.join(self.cfg.RESULT_DIR,
                                       f"{model_name}-gptq-4bit")

        os.makedirs(self.cfg.RESULT_DIR, exist_ok=True)
        os.makedirs(self.cfg.LOG_DIR, exist_ok=True)
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger("LLM_Export_GPTQ")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(message)s',
                                      datefmt='%H:%M:%S')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def get_calibration_data(self, n_samples=1024):
        """
        加载 C4 数据集 (根据你的 GPTQ 代码片段适配)
        """
        self.logger.info(f"📥 加载校准数据 (C4, samples={n_samples})...")
        try:
            data = load_dataset(
                "allenai/c4",
                data_files="en/c4-train.00001-of-01024.json.gz",
                split="train").select(range(n_samples))["text"]
            return data
        except Exception as e:
            self.logger.error(f"❌ 数据加载失败: {e}")
            exit(1)

    def run(self):
        start_time = time.time()

        quant_config = QuantizeConfig(bits=self.cfg.QUANTIZATION_BITS,
                                      group_size=self.cfg.GROUP_SIZE)

        self.logger.info(f"📥 加载原模型: {self.cfg.MODEL_PATH}")
        try:
            model = GPTQModel.load(self.cfg.MODEL_PATH, quant_config)
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            exit(1)

        calib_data = self.get_calibration_data(n_samples=1024)

        self.logger.info("⚡ 开始执行 GPTQ 量化...")
        try:
            model.quantize(calib_data, batch_size=2)
        except Exception as e:
            self.logger.error(f"❌ 量化过程出错: {e}")
            exit(1)

        self.logger.info(f"💾 保存模型到: {self.output_dir}")
        model.save(self.output_dir)

        self.logger.info(f"🏁 完成! 耗时: {time.time() - start_time:.2f}s")
        self.logger.info(f"👉 请修改推理脚本的 model_path 为: {self.output_dir}")


if __name__ == "__main__":
    exporter = QuantizationExporter()
    exporter.run()
