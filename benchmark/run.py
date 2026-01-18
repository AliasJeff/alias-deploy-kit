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

            self.tokenizer.padding_side = 'left'
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.info("🔧 Tokenizer 缺少 pad_token，已自动设置为 eos_token")

            is_awq = "awq" in self.cfg.MODEL_PATH.lower(
            ) or "marlin" in self.cfg.MODEL_PATH.lower()

            if is_awq:
                self.logger.info(
                    "🔧 检测到 AWQ/Marlin 模型，使用 AutoAWQForCausalLM 加载...")
                from awq import AutoAWQForCausalLM

                self.model = AutoAWQForCausalLM.from_pretrained(
                    self.cfg.MODEL_PATH,
                    low_cpu_mem_usage=True,
                    device_map="cuda",  # 强制使用 GPU
                    torch_dtype=self.cfg.TORCH_DTYPE,
                    trust_remote_code=True)
                self.device = self.model.model.device
            else:
                if self.cfg.LOAD_IN_4BIT:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.cfg.MODEL_PATH,
                        device_map="auto",
                        quantization_config=bnb_config,
                        torch_dtype=self.cfg.TORCH_DTYPE,
                        trust_remote_code=True)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.cfg.MODEL_PATH,
                        device_map=self.cfg.DEVICE,
                        torch_dtype=self.cfg.TORCH_DTYPE,
                        trust_remote_code=True)
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
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
        device = self.device if getattr(
            self, "device", None) is not None else self.model.device

        if device.type == "cuda":
            total_vram = torch.cuda.get_device_properties(device).total_memory
            try:
                torch.cuda.set_per_process_memory_fraction(0.95, device)
            except Exception as e:
                self.logger.warning(f"⚠️ 无法设置显存硬限制: {e}")

        batch_size = self.cfg.BATCH_SIZE

        if self.cfg.WARMUP_ROUNDS > 0:
            self.logger.info(
                f"🔥 开始预热 ({self.cfg.WARMUP_ROUNDS} 轮) 并测算极限 Batch Size...")
            try:
                dummy_text = "设计一个企业级SaaS后台管理系统仪表盘"
                dummy_input = self.tokenizer(dummy_text,
                                             return_tensors="pt",
                                             truncation=True,
                                             max_length=2048).to(device)

                static_memory = 0
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    static_memory = torch.cuda.memory_allocated()

                for _ in range(self.cfg.WARMUP_ROUNDS):
                    self.model.generate(**dummy_input,
                                        max_new_tokens=self.cfg.MAX_NEW_TOKENS)

                if self.cfg.AUTO_BATCH_SIZE and device.type == "cuda":
                    peak_memory = torch.cuda.max_memory_allocated()
                    total_vram = torch.cuda.get_device_properties(
                        device).total_memory

                    self.logger.info(
                        f"🔍 显存使用情况: 总显存={total_vram/1024**3:.2f}GB | 峰值显存={peak_memory/1024**3:.2f}GB | 静态显存={static_memory/1024**3:.2f}GB"
                    )
                    mem_per_sample = peak_memory - static_memory

                    available_mem = (total_vram * 0.9) - static_memory

                    if mem_per_sample > 0:
                        calculated_bs = int(available_mem / mem_per_sample)
                        optimal_batch_size = max(1, calculated_bs)

                        self.logger.info(
                            f"💾 显存测算: 单条极限占用={mem_per_sample/1024**3:.2f}GB | "
                            f"可用显存={available_mem/1024**3:.2f}GB")
                        self.logger.info(
                            f"🚀 自动调整 BATCH_SIZE: {batch_size} -> {optimal_batch_size}"
                        )
                        batch_size = optimal_batch_size
                    else:
                        self.logger.warning("⚠️ 显存占用过小无法准确测算，保持默认值")

            except Exception as e:
                self.logger.warning(f"⚠️ 预热或显存计算出错 (已忽略): {e}")
                import traceback
                traceback.print_exc()

        if len(data) > batch_size:
            self.logger.info(f"✂️ 截取前 {batch_size} 条数据进行单轮极限测试...")
            data = data[:batch_size]

        actual_run_size = len(data)

        self.logger.info(f"⚡ 开始极限压力测试，本轮并发数量: {actual_run_size}...")

        total_start_time = time.time()
        total_output_tokens = 0
        latencies = []

        # 这里的 range 步长是 batch_size，且 len(data) <= batch_size，所以循环只会跑一次
        for i in range(0, len(data), batch_size):
            batch_items = data[i:i + batch_size]
            current_batch_count = len(batch_items)
            batch_prompts = [item['prompt'] for item in batch_items]

            try:
                formatted_prompts = []
                for p in batch_prompts:
                    messages = []
                    if hasattr(self.cfg, "SYSTEM_INSTRUCTIONS"):
                        messages.append({
                            "role": "system",
                            "content": self.cfg.SYSTEM_INSTRUCTIONS
                        })
                    messages.append({"role": "user", "content": p})

                    formatted = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    formatted_prompts.append(formatted)

                inputs = self.tokenizer(formatted_prompts,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True,
                                        max_length=2048).to(device)

                input_token_len = inputs.input_ids.shape[1]

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
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                t1 = time.perf_counter()

                batch_latency = t1 - t0
                avg_item_latency = batch_latency / len(batch_items)

                for _ in batch_items:
                    latencies.append(avg_item_latency)

                generated_tokens = outputs[:, input_token_len:]
                decoded_outputs = self.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True)

                for idx, (item, out_text, out_tokens) in enumerate(
                        zip(batch_items, decoded_outputs, generated_tokens)):
                    valid_out_tokens = len([
                        t for t in out_tokens
                        if t != self.tokenizer.pad_token_id
                    ])
                    total_output_tokens += valid_out_tokens

                    item_tps = valid_out_tokens / batch_latency

                    result_entry = {
                        "id": item['id'],
                        "prompt": {
                            "user":
                            item['prompt'],
                            "system":
                            self.cfg.SYSTEM_INSTRUCTIONS if hasattr(
                                self.cfg, "SYSTEM_INSTRUCTIONS") else None,
                        },
                        "output": out_text,
                        "metrics": {
                            "input_tokens": input_token_len,
                            "output_tokens": valid_out_tokens,
                            "latency": round(avg_item_latency, 4),
                            "batch_latency": round(batch_latency, 4),
                            "tps": round(item_tps, 2),
                            "memory_stats": self.get_memory_usage(),
                            "batch_size": current_batch_count
                        }
                    }
                    self.results.append(result_entry)

                self.logger.info(
                    f"[Batch Limit Test] size={len(batch_items)} | "
                    f"Batch耗时: {batch_latency:.2f}s | "
                    f"显存占用: {self.get_memory_usage()}")

            except Exception as e:
                self.logger.error(f"❌ 处理 Batch {i} 出错: {e}")
                import traceback
                traceback.print_exc()

        total_duration = time.time() - total_start_time

        self.logger.info(f"🏁 极限测试完成，总耗时: {total_duration:.2f}s")

        self.save_report(total_duration, total_output_tokens, latencies,
                         actual_run_size)

    def save_report(self, total_duration, total_output_tokens, latencies,
                    run_batch_size):
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
                "total_requests": run_batch_size,
                "config": {
                    "torch_dtype": str(self.cfg.TORCH_DTYPE),
                    "load_in_4bit": self.cfg.LOAD_IN_4BIT,
                    "max_new_tokens": self.cfg.MAX_NEW_TOKENS,
                    "temperature": self.cfg.TEMPERATURE,
                    "top_p": self.cfg.TOP_P,
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
        self.logger.info(f"最大并发 (Batch Size): {run_batch_size}")
        self.logger.info(f"平均延迟: {avg_latency:.4f} s/req")
        self.logger.info(f"推理吞吐: {global_tps:.2f} tokens/s")
        self.logger.info(f"详细结果: {output_file}")
        self.logger.info("=" * 30)
        self.logger.info(
            f"日志文件: {os.path.join(self.cfg.LOG_DIR, datetime.now().strftime('%Y-%m-%d') + '.log')}"
        )


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()
