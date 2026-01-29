import os
# 设置环境变量以减少显存碎片，必须在 import torch 之前设置才能生效
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        "name": "Qwen3-4B-AWQ-GEMM-4bit",
        "path": "./models/Qwen3-4B-awq-gemm-4bit",
        "load_in_4bit": False
    },
    {
        "name": "Qwen3-4B",
        "path": "./models/Qwen3-4B",
        "load_in_4bit": False
    },
    {
        "name": "Qwen3-1.7B-AWQ-GEMM-4bit",
        "path": "./models/Qwen3-1.7B-awq-gemm-4bit",
        "load_in_4bit": False
    },
    {
        "name": "Qwen3-1.7B",
        "path": "./models/Qwen3-1.7B",
        "load_in_4bit": False
    },
    {
        "name": "Qwen2-1.5B",
        "path": "./models/Qwen2-1.5B",
        "load_in_4bit": False
    },
]

CONFIG = {
    "result_dir": "./benchmark_results",
    "log_dir": "./benchmark_logs",
    "device": "cuda",
    "torch_dtype": torch.bfloat16,
    "max_vram_gb": 7.5,

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
        if logger.hasHandlers():
            logger.handlers.clear()

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

    def _apply_vram_limit_if_needed(self):
        """
        若 CONFIG["max_vram_gb"] 存在且为正数，则限制 PyTorch 进程可用显存比例。
        不设置则保持默认（尽量用满）。
        """
        if not torch.cuda.is_available():
            return

        max_gb = CONFIG.get("max_vram_gb", None)
        if max_gb is None:
            return

        try:
            max_gb = float(max_gb)
        except Exception:
            return

        if max_gb <= 0:
            return

        device_idx = torch.cuda.current_device()
        total_bytes = torch.cuda.get_device_properties(device_idx).total_memory
        total_gb = total_bytes / (1024**3)

        frac = max_gb / total_gb
        # clamp 到 (0, 1]
        frac = max(1e-6, min(1.0, frac))

        try:
            torch.cuda.set_per_process_memory_fraction(frac, device=device_idx)
            self.logger.info(
                f"🧩 已启用显存上限: {max_gb:.2f}GB / {total_gb:.2f}GB (fraction={frac:.4f})"
            )
        except Exception as e:
            self.logger.warning(f"⚠️ 设置显存上限失败（忽略继续）: {e}")

    def load_model(self, model_conf):
        self.logger.info(
            f"📥 正在加载模型: {model_conf['name']} ({model_conf['path']})...")
        self.clear_cache()

        # 应用显存上限
        self._apply_vram_limit_if_needed()

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
            torch.cuda.synchronize()
            t_gen_start = time.perf_counter()
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_len,
                    min_new_tokens=gen_len,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id)
            torch.cuda.synchronize()
            total_time = time.perf_counter() - t_gen_start

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
        注意：此处为了速度使用 do_sample=False (Greedy)，这会导致显存占用比实际测试略小。
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

                # 探测时使用 do_sample=False
                with torch.no_grad():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=output_len,
                        min_new_tokens=output_len,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id)

                max_bs = mid
                low = mid + 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    high = mid - 1
                else:
                    self.logger.warning(f"      BS={mid} 发生非显存错误: {e}")
                    high = mid - 1
            except Exception:
                high = mid - 1

        max_bs = int(max_bs * 0.9)
        self.logger.info(f"      ✅ 最大 Batch Size = {max_bs} (预留10%显存)")
        return max_bs

    def run_throughput_test(self, model_name):
        self.logger.info(f"🚀 [{model_name}] 开始极限吞吐测试 (Target: Max Memory)...")
        results = []

        prompt_lens = CONFIG["throughput_test"]["prompt_lens"]
        new_tokens_list = CONFIG["throughput_test"]["new_tokens"]

        for p_len in prompt_lens:
            for n_tokens in new_tokens_list:
                # 1. 初始探测
                bs = self.find_max_batch_size(p_len, n_tokens)

                if bs == 0:
                    self.logger.warning(
                        f"   In={p_len}, Out={n_tokens} | 无法运行 (OOM at BS=1)")
                    continue

                # OOM 自动降级重试循环
                actual_bs = bs
                success = False
                latency = 0
                mem_info = {}
                retry_count = 0

                # 只要 BS > 0 且未成功，就持续重试
                while actual_bs > 0 and not success:
                    try:
                        self.clear_cache()
                        # 注意：必须使用新的 Batch Size 重新生成 Input
                        inputs = self.generate_dummy_input(actual_bs, p_len)

                        torch.cuda.synchronize()
                        t0 = time.perf_counter()

                        with torch.no_grad():
                            self.model.generate(
                                **inputs,
                                max_new_tokens=n_tokens,
                                min_new_tokens=n_tokens,
                                do_sample=True,  # 采样模式比探测模式更吃显存
                                temperature=0.7,
                                pad_token_id=self.tokenizer.pad_token_id)

                        torch.cuda.synchronize()
                        latency = time.perf_counter() - t0
                        mem_info = self.get_memory_usage()
                        success = True  # 标记成功，跳出循环

                    except (RuntimeError, torch.OutOfMemoryError) as e:
                        # 捕获 OOM 异常
                        if "out of memory" in str(e).lower() or isinstance(
                                e, torch.OutOfMemoryError):
                            old_bs = actual_bs
                            # 策略：如果 OOM，打 8 折重试
                            actual_bs = int(actual_bs * 0.8)
                            # 避免 BS 过小时陷入死循环 (如 2->2)
                            if actual_bs == old_bs:
                                actual_bs -= 1

                            self.logger.warning(
                                f"      ⚠️ 实际运行 OOM (BS={old_bs}) -> 自动降级至 {actual_bs} 重试..."
                            )
                            retry_count += 1
                        else:
                            # 其他错误直接抛出
                            self.logger.error(f"      ❌ 运行时发生非 OOM 错误: {e}")
                            break
                # =======================================================

                if not success:
                    self.logger.error(f"   In={p_len}, Out={n_tokens} | 最终失败")
                    continue

                # 计算指标
                rps = actual_bs / latency

                self.logger.info(
                    f"   In={p_len}, Out={n_tokens} | MaxBS={actual_bs} (Init={bs}) | "
                    f"RPS={rps:.2f} | Time={latency:.2f}s | Mem={mem_info.get('max_allocated_gb', 0)}GB"
                )

                results.append({
                    "input_len": p_len,
                    "output_len": n_tokens,
                    "max_batch_size": actual_bs,
                    "init_batch_size": bs,
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

            del self.model
            del self.tokenizer
            self.clear_cache()

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
