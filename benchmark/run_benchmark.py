import os
import json
import time
import torch
import logging
import gc
import math
import psutil
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS_TO_TEST = [
    {
        "name": "Qwen2-1.5B",
        "path": "./models/Qwen2-1.5B",
        "load_in_4bit": False
    },
    {
        "name": "Qwen3-1.7B",
        "path": "./models/Qwen3-1.7B",
        "load_in_4bit": False
    },
    {
        "name": "Qwen3-1.7B-AWQ-GEMM-4bit",
        "path": "./models/Qwen3-1.7B-awq-gemm-4bit",
        "load_in_4bit": False
    },
    {
        "name": "Qwen3-4B",
        "path": "./models/Qwen3-4B",
        "load_in_4bit": False
    },
    {
        "name": "Qwen3-4B-AWQ-GEMM-4bit",
        "path": "./models/Qwen3-4B-awq-gemm-4bit",
        "load_in_4bit": False
    },
]

CONFIG = {
    "result_dir": "./benchmark_results",
    "log_dir": "./benchmark_logs",
    "device": "cuda",
    "torch_dtype": torch.bfloat16,

    # 单请求 Latency 测试参数
    "latency_test": {
        "prompt_lens": [64, 256, 1024],
        "gen_len": 128,  # 用于测试 Decode 速度的固定生成长度
    },

    # 满显存 Throughput 测试参数
    "throughput_test": {
        "prompt_lens": [64, 256, 1024],
        "new_tokens": [16, 64, 512],
        "max_bs_cap": 2048  # 搜索最大 Batch Size 时的安全上限
    }
}


class BenchmarkRunner:

    def __init__(self):
        self.results = []
        os.makedirs(CONFIG["result_dir"], exist_ok=True)
        os.makedirs(CONFIG["log_dir"], exist_ok=True)
        self.logger = self.setup_logger()
        self.device = torch.device(CONFIG["device"])

    def setup_logger(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(CONFIG["log_dir"], f"bench_{today_str}.log")

        logger = logging.getLogger("Benchmark")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

        # 文件输出
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # 控制台输出
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def get_memory_usage(self):
        """获取当前显存占用信息 (GB)"""
        mem_info = {}
        if torch.cuda.is_available():
            mem_info['allocated_gb'] = round(
                torch.cuda.memory_allocated() / (1024**3), 2)
            mem_info['reserved_gb'] = round(
                torch.cuda.memory_reserved() / (1024**3), 2)
            mem_info['max_allocated_gb'] = round(
                torch.cuda.max_memory_allocated() / (1024**3), 2)
        return mem_info

    def clear_cache(self):
        """强制清理显存和垃圾回收"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def load_model(self, model_conf):
        self.logger.info(
            f"📥 正在加载模型: {model_conf['name']} ({model_conf['path']})...")
        self.clear_cache()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_conf['path'], trust_remote_code=True)
            # Decoder-only 模型 Batch 推理通常使用左填充
            self.tokenizer.padding_side = 'left'
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            bnb_config = None
            if model_conf.get("load_in_4bit", False):
                self.logger.info("🔧 启用 4-bit 量化加载")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=CONFIG["torch_dtype"])

            self.model = AutoModelForCausalLM.from_pretrained(
                model_conf['path'],
                device_map="auto"
                if model_conf.get("load_in_4bit") else CONFIG["device"],
                quantization_config=bnb_config,
                torch_dtype=CONFIG["torch_dtype"],
                trust_remote_code=True)
            self.model.eval()
            self.logger.info(f"✅ 模型加载成功 | 显存: {self.get_memory_usage()}")
            return True
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            return False

    def generate_dummy_input(self, batch_size, prompt_len):
        """生成指定长度的随机 input_ids (避开 Tokenizer 开销，确保长度精确)"""
        vocab_size = self.model.config.vocab_size
        # 生成随机 token，避开特殊 token (0-100)
        input_ids = torch.randint(100,
                                  vocab_size - 100, (batch_size, prompt_len),
                                  device=self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    # ================= 单请求性能测试 =================
    def run_latency_test(self, model_name):
        self.logger.info(f"⚡ [{model_name}] 开始单请求性能测试 (Latency)...")
        results = []
        gen_len = CONFIG["latency_test"]["gen_len"]

        for p_len in CONFIG["latency_test"]["prompt_lens"]:
            self.clear_cache()

            # 1. 预热
            dummy = self.generate_dummy_input(1, p_len)
            with torch.no_grad():
                self.model.generate(**dummy, max_new_tokens=1)

            # 2. 测试 Prefill (Forward Pass)
            # 使用纯 forward pass 测量 prefill 时间，比 generate(max_new_tokens=1) 更纯粹
            inputs = self.generate_dummy_input(1, p_len)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = self.model(input_ids=inputs['input_ids'],
                               attention_mask=inputs['attention_mask'])
            torch.cuda.synchronize()
            prefill_time = time.perf_counter() - t0

            prefill_speed = p_len / prefill_time

            # 3. 测试 Decode
            # 运行 generate 生成 N 个 token，总时间 T_total
            # Decode 时间 ≈ T_total - Prefill 时间
            torch.cuda.synchronize()
            t_gen_start = time.perf_counter()
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_len,
                    min_new_tokens=gen_len,  # 强制生成固定长度
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id)
            torch.cuda.synchronize()
            total_time = time.perf_counter() - t_gen_start

            # 计算纯 Decode 时间
            decode_time = max(1e-6, total_time - prefill_time)
            decode_speed = gen_len / decode_time

            self.logger.info(
                f"   Input: {p_len} | Prefill: {prefill_speed:.1f} t/s | Decode: {decode_speed:.1f} t/s"
            )

            results.append({
                "prompt_len": p_len,
                "gen_len": gen_len,
                "prefill_tokens_per_s": round(prefill_speed, 2),
                "decode_tokens_per_s": round(decode_speed, 2),
                "latency_ms": round(total_time * 1000, 2)
            })

        return results

    # ================= 满显存吞吐测试 =================
    def find_max_batch_size(self, input_len, output_len):
        """
        核心逻辑：使用二分查找法寻找不 OOM 的最大 Batch Size。
        这比线性估算更安全，能真正测出“占满显存”的极限。
        """
        low = 1
        high = CONFIG["throughput_test"]["max_bs_cap"]
        max_bs = 1

        self.logger.info(
            f"   🔍 探测最大 Batch Size (In={input_len}, Out={output_len})...")

        while low <= high:
            mid = (low + high) // 2
            try:
                self.clear_cache()
                inputs = self.generate_dummy_input(mid, input_len)

                # 尝试完整生成
                with torch.no_grad():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=output_len,
                        min_new_tokens=output_len,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id)

                # 如果成功，尝试更大的 BS
                max_bs = mid
                low = mid + 1
            except RuntimeError as e:
                # 如果 OOM，尝试更小的 BS
                if "out of memory" in str(e).lower():
                    high = mid - 1
                else:
                    self.logger.warning(f"      BS={mid} 发生非显存错误: {e}")
                    high = mid - 1
            except Exception:
                high = mid - 1

        return max_bs

    def run_throughput_test(self, model_name):
        self.logger.info(f"🚀 [{model_name}] 开始极限吞吐测试 (Target: Max Memory)...")
        results = []

        prompt_lens = CONFIG["throughput_test"]["prompt_lens"]
        new_tokens_list = CONFIG["throughput_test"]["new_tokens"]

        for p_len in prompt_lens:
            for n_tokens in new_tokens_list:
                # 1. 自动探测最大 Batch Size
                bs = self.find_max_batch_size(p_len, n_tokens)

                if bs == 0:
                    self.logger.warning(
                        f"   In={p_len}, Out={n_tokens} | 无法运行 (OOM at BS=1)")
                    continue

                # 2. 使用最大 BS 进行正式测速
                self.clear_cache()
                inputs = self.generate_dummy_input(bs, p_len)

                torch.cuda.synchronize()
                t0 = time.perf_counter()

                with torch.no_grad():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=n_tokens,
                        min_new_tokens=n_tokens,
                        do_sample=True,  # 模拟真实业务（开启采样）
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id)

                torch.cuda.synchronize()
                latency = time.perf_counter() - t0

                # 计算指标
                rps = bs / latency
                total_gen_tokens = bs * n_tokens
                # 记录每秒处理请求数

                mem_info = self.get_memory_usage()

                self.logger.info(
                    f"   In={p_len}, Out={n_tokens} | MaxBS={bs} | "
                    f"RPS={rps:.2f} | Time={latency:.2f}s | Mem={mem_info.get('max_allocated_gb', 0)}GB"
                )

                results.append({
                    "input_len": p_len,
                    "output_len": n_tokens,
                    "max_batch_size": bs,
                    "rps": round(rps, 2),
                    "latency_sec": round(latency, 4),
                    "memory_stats": mem_info
                })

        return results

    def run(self):
        final_report = {
            "meta": {
                "timestamp":
                datetime.now().isoformat(),
                "config":
                CONFIG,
                "gpu":
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "CPU"
            },
            "results": []
        }

        for model_conf in MODELS_TO_TEST:
            model_name = model_conf["name"]

            self.logger.info("=" * 40)
            self.logger.info(f"🤖 开始评测模型: {model_name}")
            self.logger.info("=" * 40)

            if not self.load_model(model_conf):
                continue

            # 执行测试
            latency_res = self.run_latency_test(model_name)
            throughput_res = self.run_throughput_test(model_name)

            final_report["results"].append({
                "model_name":
                model_name,
                "model_path":
                model_conf["path"],
                "single_latency_metrics":
                latency_res,
                "max_throughput_metrics":
                throughput_res
            })

            # 清理显存，准备下一个模型
            del self.model
            del self.tokenizer
            self.clear_cache()

        # 保存报告
        timestamp = int(time.time())
        output_file = os.path.join(CONFIG["result_dir"],
                                   f"benchmark_report_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        self.logger.info("=" * 40)
        self.logger.info(f"🏁 所有测试完成，报告已保存至: {output_file}")
        self.logger.info("=" * 40)


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()
