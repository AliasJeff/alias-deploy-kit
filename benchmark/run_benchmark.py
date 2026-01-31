import os
# 设置环境变量以减少显存碎片，必须在 import torch 之前设置才能生效
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import torch
import logging
import gc
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

MODELS_TO_TEST = [
    {
        "name": "Qwen3-4B-GPTQ-4bit",
        "path": "./models/Qwen3-4B-gptq-4bit",
        "load_in_4bit": False
    },
    {
        "name": "Qwen3-4B",
        "path": "./models/Qwen3-4B",
        "load_in_4bit": False
    },
    {
        "name": "Qwen3-1.7B-GPTQ-4bit",
        "path": "./models/Qwen3-1.7B-gptq-4bit",
        "load_in_4bit": False
    },
    {
        "name": "Qwen3-1.7B",
        "path": "./models/Qwen3-1.7B",
        "load_in_4bit": False
    },
    {
        "name": "Qwen2-1.5B-GPTQ-4bit",
        "path": "./models/Qwen2-1.5B-gptq-4bit",
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
    "torch_dtype": torch.float16,
    "max_vram_gb": 7.3,

    # 单请求 Latency 测试参数
    "latency_test": {
        "prompt_lens": [16, 64, 256],
        "gen_len": 128,  # 用于测试 Decode 速度的固定生成长度
    },

    # 满显存 Throughput 测试参数
    "throughput_test": {
        "prompt_lens": [16, 64, 256],
        "new_tokens": [16, 64, 80, 128, 512],
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
                self.logger.info("🔧 Tokenizer 缺少 pad_token，已自动设置为 eos_token")

            is_awq = "awq" in model_conf['path'].lower(
            ) or "marlin" in model_conf['path'].lower()

            is_gptq = "gptq" in model_conf['path'].lower()

            if is_awq:
                self.logger.info(
                    "🔧 检测到 AWQ/Marlin 模型，使用 AutoAWQForCausalLM 加载...")
                from awq import AutoAWQForCausalLM

                self.model = AutoAWQForCausalLM.from_pretrained(
                    model_conf['path'],
                    low_cpu_mem_usage=True,
                    device_map="cuda",  # 强制使用 GPU
                    torch_dtype=CONFIG["torch_dtype"],
                    trust_remote_code=True)
                self.device = self.model.model.device
            elif is_gptq:
                self.logger.info("🔧 检测到 GPTQ 模型，使用 GPTQModel 加载...")
                gptq_config = GPTQConfig(
                    bits=4,
                    desc_act=True,
                    act_group_aware=False,
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_conf['path'],
                    quantization_config=gptq_config,
                    device_map="cuda",
                    torch_dtype=CONFIG["torch_dtype"],
                    trust_remote_code=True)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_conf['path'],
                    device_map="cuda",
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
        low = 1
        high = CONFIG["throughput_test"]["max_bs_cap"]
        max_bs = 1

        threshold = 40
        alignment = 8

        test_output_len = output_len

        self.logger.info(
            f" 🔍 探测最大 BS (精准模式: In={input_len}, Out={test_output_len})...")

        while (high - low) > threshold:
            mid = (low + high) // 2

            if mid > alignment:
                mid = (mid // alignment) * alignment
            if mid <= low:
                mid = low + alignment

            try:
                self.clear_cache()
                inputs = self.generate_dummy_input(mid, input_len)

                with torch.no_grad():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=test_output_len,
                        min_new_tokens=test_output_len,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id)

                max_bs = mid
                low = mid
                # self.logger.debug(f"    ✅ BS={mid} Pass")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    high = mid
                else:
                    self.logger.warning(f"    ⚠️ BS={mid} 非显存错误: {e}")
                    high = mid
            except Exception:
                high = mid

        safe_bs = int(max_bs * 0.95)
        safe_bs = (safe_bs // alignment) * alignment
        safe_bs = max(1, safe_bs)

        self.logger.info(
            f"    ✅ 探测结束，最大 Batch Size ≈ {safe_bs} (范围 {low}-{high})")
        return safe_bs

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

                actual_bs = bs
                success = False

                prefill_latency = 0.0
                decode_latency = 0.0
                total_latency = 0.0
                mem_info = {}
                retry_count = 0

                while actual_bs > 0 and not success:
                    try:
                        self.clear_cache()
                        inputs = self.generate_dummy_input(actual_bs, p_len)

                        # 1: 测量 Prefill 时间 (纯 Forward Pass)
                        torch.cuda.synchronize()
                        t_prefill_start = time.perf_counter()

                        with torch.no_grad():
                            # 只做一次前向传播
                            _ = self.model(
                                input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'])

                        torch.cuda.synchronize()
                        prefill_latency = time.perf_counter() - t_prefill_start

                        # 释放 Prefill 产生的临时显存
                        del _
                        self.clear_cache()

                        # 2: 测量 Total 时间 (Prefill + Decode)
                        torch.cuda.synchronize()
                        t_total_start = time.perf_counter()

                        with torch.no_grad():
                            self.model.generate(
                                **inputs,
                                max_new_tokens=n_tokens,
                                min_new_tokens=n_tokens,
                                do_sample=True,
                                temperature=0.7,
                                pad_token_id=self.tokenizer.pad_token_id)

                        torch.cuda.synchronize()
                        total_latency = time.perf_counter() - t_total_start

                        mem_info = self.get_memory_usage()
                        success = True

                    except (RuntimeError, torch.OutOfMemoryError) as e:
                        if "out of memory" in str(e).lower() or isinstance(
                                e, torch.OutOfMemoryError):
                            old_bs = actual_bs
                            actual_bs = int(actual_bs * 0.8)
                            if actual_bs == old_bs:
                                actual_bs -= 1
                            self.logger.warning(
                                f"      ⚠️ 实际运行 OOM (BS={old_bs}) -> 自动降级至 {actual_bs} 重试..."
                            )
                            retry_count += 1
                        else:
                            self.logger.error(f"      ❌ 运行时发生非 OOM 错误: {e}")
                            break

                if not success:
                    self.logger.error(f"   In={p_len}, Out={n_tokens} | 最终失败")
                    continue

                decode_latency = max(0, total_latency - prefill_latency)

                rps = actual_bs / total_latency

                decode_tps = (actual_bs * n_tokens
                              ) / decode_latency if decode_latency > 0 else 0

                prefill_tps = (
                    actual_bs *
                    p_len) / prefill_latency if prefill_latency > 0 else 0

                self.logger.info(
                    f"   In={p_len}, Out={n_tokens} | BS={actual_bs} | "
                    f"RPS={rps:.2f} | "
                    f"Prefill={(prefill_latency*1000):.1f}ms | "
                    f"Decode={(decode_latency*1000):.1f}ms | "
                    f"GenSpeed={decode_tps:.0f} tok/s")

                results.append({
                    "input_len": p_len,
                    "output_len": n_tokens,
                    "max_batch_size": actual_bs,
                    "init_batch_size": bs,
                    "rps": round(rps, 2),
                    "total_latency": round(total_latency, 4),
                    "prefill_latency": round(prefill_latency, 4),
                    "decode_latency": round(decode_latency, 4),
                    "prefill_tps": round(prefill_tps, 2),
                    "decode_tps": round(decode_tps, 2),
                    "memory_stats": mem_info
                })

        return results

    def run(self):
        serializable_config = CONFIG.copy()
        if "torch_dtype" in serializable_config:
            serializable_config["torch_dtype"] = str(
                serializable_config["torch_dtype"])

        final_report = {
            "meta": {
                "timestamp":
                datetime.now().isoformat(),
                "config":
                serializable_config,
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
