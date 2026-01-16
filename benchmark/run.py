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

        # 预热
        if self.cfg.WARMUP_ROUNDS > 0:
            self.logger.info(f"🔥 开始预热 ({self.cfg.WARMUP_ROUNDS} 轮)...")
            try:
                # 构造简单的输入
                dummy_input = self.tokenizer("Hello", return_tensors="pt").to(
                    self.device)
                for _ in range(self.cfg.WARMUP_ROUNDS):
                    self.model.generate(**dummy_input, max_new_tokens=10)
            except Exception as e:
                self.logger.warning(f"⚠️ 预热过程中出现小问题 (可忽略): {e}")

        self.logger.info(f"⚡ 开始推理，共 {len(data)} 条测试数据...")

        total_start_time = time.time()
        total_output_tokens = 0
        latencies = []

        batch_size = self.cfg.BATCH_SIZE

        for i in range(0, len(data), batch_size):
            # 获取当前批次的数据 (切片)
            batch_items = data[i:i + batch_size]
            batch_prompts = [item['prompt'] for item in batch_items]

            try:
                # 1. 批量编码
                # 注意：apply_chat_template 默认处理单条，我们需要手动对列表中的每条应用 template
                formatted_prompts = []
                for p in batch_prompts:
                    formatted = self.tokenizer.apply_chat_template(
                        [{
                            "role":
                            "user",
                            "content":
                            f"{p} 只输出适配手机端的html代码，输出最小可行的html，限制200token，不要输出任何其他内容。 </no_think>"
                        }],
                        tokenize=False,
                        add_generation_prompt=True)
                    formatted_prompts.append(formatted)

                # 使用 padding=True 确保 tensor 维度对齐
                inputs = self.tokenizer(formatted_prompts,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True,
                                        max_length=2048).to(self.device)

                input_token_len = inputs.input_ids.shape[1]

                # 2. 批量推理
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
                        pad_token_id=self.tokenizer.
                        pad_token_id,  # 显式指定 pad token
                    )
                t1 = time.perf_counter()

                batch_latency = t1 - t0
                # 记录该批次的每个样本的平均延迟（用于统计）
                # 注意：实际生产中更关注吞吐量，这里为了兼容 report 格式，我们记录平均值
                avg_item_latency = batch_latency / len(batch_items)

                for _ in batch_items:
                    latencies.append(avg_item_latency)

                # 3. 批量解码
                # 只解码新生成的 tokens (outputs 包含 input + new_tokens)
                generated_tokens = outputs[:, input_token_len:]
                decoded_outputs = self.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True)

                # 4. 结果回填
                for idx, (item, out_text, out_tokens) in enumerate(
                        zip(batch_items, decoded_outputs, generated_tokens)):
                    # 计算当前样本的 token 数量 (去除 padding)
                    # 因为 batch 生成时会有 padding，需要计算实际有效 token
                    valid_out_tokens = len([
                        t for t in out_tokens
                        if t != self.tokenizer.pad_token_id
                    ])
                    total_output_tokens += valid_out_tokens

                    # 估算 TPS (基于该样本有效 token 和 批次总时间)
                    # 注意：Batch 场景下 TPS 算法有多种，这里使用 (单个样本Token / 批次时间) 会偏小，
                    # 也可以用 (批次总Token / 批次时间)。这里为了兼容单条记录，仅记录单个 TPS。
                    item_tps = valid_out_tokens / batch_latency

                    result_entry = {
                        "id": item['id'],
                        "prompt": item['prompt'],
                        "output": out_text,
                        "metrics": {
                            "input_tokens": input_token_len,  # 批次内取最大长度
                            "output_tokens": valid_out_tokens,
                            "latency": round(avg_item_latency, 4),  # 记录平均延迟
                            "batch_latency": round(batch_latency,
                                                   4),  # [新增] 记录该批次实际物理耗时
                            "tps": round(item_tps, 2),
                            "memory_stats": self.get_memory_usage()
                        }
                    }
                    self.results.append(result_entry)

                self.logger.info(
                    f"[Batch {i//batch_size + 1}] size={len(batch_items)} | "
                    f"Batch耗时: {batch_latency:.2f}s | "
                    f"Prompt预览: {batch_prompts[0][:10]}...")

            except Exception as e:
                self.logger.error(f"❌ 处理 Batch {i} 出错: {e}")
                import traceback
                traceback.print_exc()

        # 计算总耗时（覆盖所有 Batch）
        total_duration = time.time() - total_start_time

        self.logger.info(f"🏁 所有测试完成，总耗时: {total_duration:.2f}s")
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
