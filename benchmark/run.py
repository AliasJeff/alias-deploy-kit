# run.py
import os
import json
import time
import torch
import psutil
import logging
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import Config


class BenchmarkRunner:

    def __init__(self):
        self.cfg = Config()
        self.results = []

        os.makedirs(self.cfg.RESULT_DIR, exist_ok=True)
        os.makedirs(self.cfg.LOG_DIR, exist_ok=True)

        self.logger = self.setup_logger()
        self.logger.info(f"🚀 初始化基准测试，设备: {self.cfg.DEVICE}")

        # 检查 4bit 兼容性警告
        if self.cfg.LOAD_IN_4BIT and self.cfg.DEVICE != "cuda":
            self.logger.warning(
                "⚠️ 检测到开启了 LOAD_IN_4BIT 但设备不是 CUDA。bitsandbytes 可能无法工作。")

    def setup_logger(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.cfg.LOG_DIR, f"{today_str}.log")

        logger = logging.getLogger("LLM_Benchmark")
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

        file_handler = logging.FileHandler(log_file,
                                           mode='a',
                                           encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def get_directory_size(self, start_path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size / (1024**3)

    def get_memory_usage(self):
        mem_info = {}
        process = psutil.Process(os.getpid())
        mem_info['ram_usage_mb'] = process.memory_info().rss / 1024 / 1024

        if self.cfg.DEVICE == "cuda":
            mem_info['gpu_vram_mb'] = torch.cuda.memory_allocated(
            ) / 1024 / 1024
            mem_info['gpu_vram_max_mb'] = torch.cuda.max_memory_allocated(
            ) / 1024 / 1024
        return mem_info

    def load_model(self):
        self.logger.info(f"📥 正在加载模型: {self.cfg.MODEL_PATH} ...")

        # 构造量化配置
        bnb_config = None
        if self.cfg.LOAD_IN_4BIT:
            self.logger.info("🔧 已启用 4-bit 量化加载 (NF4)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # 推荐使用 nf4 格式
                bnb_4bit_use_double_quant=True,  # 开启双重量化以节省更多显存
                bnb_4bit_compute_dtype=self.cfg.
                TORCH_DTYPE  # 计算时使用的精度 (fp16/bf16)
            )

        start_time = time.time()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.MODEL_PATH, trust_remote_code=True)

            # 注意: 使用 load_in_4bit 时，建议 device_map="auto" 或者由 accelerate 自动处理
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.MODEL_PATH,
                device_map="auto"
                if self.cfg.LOAD_IN_4BIT else self.cfg.DEVICE,
                quantization_config=bnb_config,
                torch_dtype=self.cfg.TORCH_DTYPE,
                trust_remote_code=True)
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            self.logger.error(f"请检查路径或 bitsandbytes 是否安装正确")
            exit(1)

        load_time = time.time() - start_time
        self.logger.info(f"✅ 模型加载完成，耗时: {load_time:.2f}s")

        self.model_info = {
            "model_name": os.path.basename(self.cfg.MODEL_PATH),
            "model_size_gb": self.get_directory_size(self.cfg.MODEL_PATH),
            "quantization": "4bit" if self.cfg.LOAD_IN_4BIT else "None",
            "param_count": sum(p.numel() for p in self.model.parameters())
        }

    def load_data(self):
        try:
            with open(self.cfg.DATA_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"❌ 未找到测试数据: {self.cfg.DATA_PATH}")
            exit(1)

    def run(self):
        self.load_model()
        data = self.load_data()

        # 预热
        if self.cfg.WARMUP_ROUNDS > 0:
            self.logger.info(f"🔥 开始预热 ({self.cfg.WARMUP_ROUNDS} 轮)...")
            try:
                # 构造简单的输入
                dummy_input = self.tokenizer("Hello", return_tensors="pt").to(
                    self.model.device)
                for _ in range(self.cfg.WARMUP_ROUNDS):
                    self.model.generate(**dummy_input, max_new_tokens=10)
            except Exception as e:
                self.logger.warning(f"⚠️ 预热过程中出现小问题 (可忽略): {e}")

        self.logger.info(f"⚡ 开始推理，共 {len(data)} 条测试数据...")

        total_start_time = time.time()
        total_output_tokens = 0
        latencies = []

        for idx, item in enumerate(data):
            prompt = item['prompt']

            try:
                # 1. 编码
                formatted_prompt = self.tokenizer.apply_chat_template(
                    [{
                        "role": "user",
                        "content": prompt
                    }],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # 注意：确保输入也在正确的设备上
                inputs = self.tokenizer(
                    [formatted_prompt],
                    return_tensors="pt",
                ).to(self.model.device)  # 使用 model.device 更安全

                input_token_len = inputs.input_ids.shape[1]

                # 2. 推理
                if self.cfg.DEVICE == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                t0 = time.perf_counter()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.cfg.MAX_NEW_TOKENS,
                        temperature=self.cfg.TEMPERATURE,
                        top_p=self.cfg.TOP_P,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                t1 = time.perf_counter()
                latency = t1 - t0
                latencies.append(latency)

                # 3. 解码
                output_text = self.tokenizer.decode(
                    outputs[0][input_token_len:], skip_special_tokens=True)
                output_token_len = len(outputs[0]) - input_token_len

                total_output_tokens += output_token_len

                # 速度计算
                tps = output_token_len / latency

                result_entry = {
                    "id": item['id'],
                    "prompt": prompt,
                    "output": output_text,
                    "metrics": {
                        "input_tokens": input_token_len,
                        "output_tokens": output_token_len,
                        "latency": round(latency, 4),
                        "tps": round(tps, 2),
                        "memory_stats": self.get_memory_usage()  # 实时记录内存
                    }
                }
                self.results.append(result_entry)

                self.logger.info(
                    f"[{idx+1}/{len(data)}] 用时: {latency:.2f}s | TPS: {tps:.2f} | Prompt: {prompt[:10]}..."
                )

            except Exception as e:
                self.logger.error(f"❌ 处理 ID {item['id']} 时出错: {e}")

        total_duration = time.time() - total_start_time
        self.save_report(total_duration, total_output_tokens, latencies)

    def save_report(self, total_duration, total_output_tokens, latencies):
        if not latencies:
            self.logger.error("❌ 没有成功的推理记录，无法生成报告")
            return

        avg_latency = np.mean(latencies)
        rps = len(self.results) / total_duration
        global_tps = total_output_tokens / total_duration

        report = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model_info,
                "config": {
                    k: v
                    for k, v in vars(self.cfg).items()
                    if not k.startswith("__")
                }
            },
            "summary": {
                "total_requests": len(self.results),
                "total_duration": round(total_duration, 2),
                "avg_latency": round(avg_latency, 4),
                "rps": round(rps, 2),
                "global_tps": round(global_tps, 2),
                "final_memory": self.results[-1]['metrics'].get('memory_stats')
            },
            "details": self.results
        }

        output_file = os.path.join(self.cfg.RESULT_DIR,
                                   f"benchmark_{int(time.time())}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.logger.info("=" * 30)
        self.logger.info("📊 测试报告摘要")
        self.logger.info("=" * 30)
        self.logger.info(
            f"模型量化: {self.model_info.get('quantization', 'None')}")
        self.logger.info(f"平均延迟: {avg_latency:.4f} s/req")
        self.logger.info(f"推理吞吐: {global_tps:.2f} tokens/s")
        self.logger.info(f"详细结果: {output_file}")
        self.logger.info(
            f"日志文件: {os.path.join(self.cfg.LOG_DIR, datetime.now().strftime('%Y-%m-%d') + '.log')}"
        )


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()
