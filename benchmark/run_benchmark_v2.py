import os
# 设置环境变量以减少显存碎片，必须在 import torch 之前设置才能生效
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import torch
import logging
import gc
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS_TO_TEST = [
    {
        "name": "Qwen3-4B",
        "path": "./models/Qwen3-4B"
    },
    {
        "name": "Qwen3-4B-AWQ-GEMM-4bit",
        "path": "./models/Qwen3-4B-awq-gemm-4bit"
    },
    {
        "name": "Qwen3-1.7B-AWQ-GEMM-4bit",
        "path": "./models/Qwen3-1.7B-awq-gemm-4bit"
    },
    {
        "name": "Qwen3-1.7B",
        "path": "./models/Qwen3-1.7B"
    },
    {
        "name": "Qwen2-1.5B-AWQ-GEMM-4bit",
        "path": "./models/Qwen2-1.5B-awq-gemm-4bit"
    },
    {
        "name": "Qwen2-1.5B",
        "path": "./models/Qwen2-1.5B"
    },
]

CONFIG = {
    "data_path": "./data/test_benchmark.json",
    "result_dir": "./benchmark_results_v2",
    "log_dir": "./benchmark_logs",
    "device": "cuda",
    "torch_dtype": torch.float16,
    # "max_vram_gb": 7.5,
    "n_batch_size": [50, 100, 150, 200, 250, 300],
    "new_tokens": [16, 64, 80, 128, 256, 512, 1024],
    "system_prompt": "直接生成html代码，不要输出其他任何内容"
}


class BenchmarkRunner:

    def __init__(self):
        self.all_results = []
        os.makedirs(CONFIG["result_dir"], exist_ok=True)
        os.makedirs(CONFIG["log_dir"], exist_ok=True)
        self.logger = self.setup_logger()
        self.device = torch.device(CONFIG["device"])
        self.tokenizer = None
        self.model = None
        self.test_data = self.load_test_data()

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

    def load_test_data(self):
        """读取测试 Prompt 数据"""
        path = CONFIG["data_path"]
        if not os.path.exists(path):
            self.logger.error(f"❌ 数据文件未找到: {path}")
            return []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"📚 成功加载测试数据，共 {len(data)} 条")
            return data
        except Exception as e:
            self.logger.error(f"❌ 读取数据失败: {e}")
            return []

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
        """应用显存上限限制（若配置）"""
        if not torch.cuda.is_available():
            return
        max_gb = CONFIG.get("max_vram_gb", None)
        if max_gb is None:
            return

        try:
            max_gb = float(max_gb)
            if max_gb <= 0: return

            device_idx = torch.cuda.current_device()
            total_bytes = torch.cuda.get_device_properties(
                device_idx).total_memory
            frac = max(1e-6, min(1.0, (max_gb * 1024**3) / total_bytes))

            torch.cuda.set_per_process_memory_fraction(frac, device=device_idx)
            self.logger.info(f"🧩 已启用显存上限: {max_gb}GB (fraction={frac:.4f})")
        except Exception as e:
            self.logger.warning(f"⚠️ 设置显存上限失败: {e}")

    def load_model(self, model_conf):
        self.logger.info(
            f"📥 正在加载模型: {model_conf['name']} ({model_conf['path']})...")
        self.clear_cache()
        self._apply_vram_limit_if_needed()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_conf['path'], trust_remote_code=True)
            # Decoder-only Batch 推理必须使用左填充
            self.tokenizer.padding_side = 'left'
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.info("🔧 Tokenizer 缺少 pad_token，已自动设置为 eos_token")

            is_awq = "awq" in model_conf['path'].lower(
            ) or "marlin" in model_conf['path'].lower()

            if is_awq:
                self.logger.info(
                    "🔧 检测到 AWQ/Marlin 模型，使用 AutoAWQForCausalLM 加载...")
                try:
                    from awq import AutoAWQForCausalLM
                    self.model = AutoAWQForCausalLM.from_pretrained(
                        model_conf['path'],
                        low_cpu_mem_usage=True,
                        device_map="cuda",
                        trust_remote_code=True)
                except ImportError:
                    self.logger.error("❌ 未安装 autoawq，请执行 pip install autoawq")
                    return False
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

    def prepare_batch_inputs(self, batch_size):
        """
        准备Batch输入数据。
        如果数据量 < batch_size，会自动循环填充数据直到填满 batch_size，
        以确保压力测试的真实并行度。
        """
        raw_prompts = [item['prompt'] for item in self.test_data]
        if not raw_prompts:
            return None, []

        target_prompts = []
        while len(target_prompts) < batch_size:
            target_prompts.extend(raw_prompts)
        target_prompts = target_prompts[:batch_size]

        formatted_texts = []
        for p in target_prompts:
            messages = [
                {
                    "role": "system",
                    "content": CONFIG["system_prompt"]
                },
                {
                    "role": "user",
                    "content": p
                },
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            formatted_texts.append(text)

        # Tokenize
        try:
            inputs = self.tokenizer(formatted_texts,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=4096).to(self.device)
            return inputs, target_prompts
        except Exception as e:
            self.logger.error(f"❌ Tokenize 失败: {e}")
            return None, []

    def run_benchmark_step(self, model_name, batch_size, new_tokens):
        """执行单次配置的性能测试"""
        inputs, _ = self.prepare_batch_inputs(batch_size)
        if inputs is None: return None

        input_token_count = inputs.input_ids.numel()

        self.logger.info(f"🚀 开始测试: BS={batch_size}, NewTokens={new_tokens}")

        self.clear_cache()

        torch.cuda.synchronize()
        t0 = time.time()

        with torch.inference_mode():
            outputs = self.model(**inputs, use_cache=True, return_dict=True)

        torch.cuda.synchronize()
        t1 = time.time()
        prefill_latency = t1 - t0

        try:
            past_key_values = outputs.past_key_values
            input_ids = inputs.input_ids[:, -1:]

            torch.cuda.synchronize()
            t2 = time.time()

            with torch.inference_mode():
                for _ in range(new_tokens):
                    out = self.model(input_ids=input_ids,
                                     past_key_values=past_key_values,
                                     use_cache=True,
                                     return_dict=True)
                    past_key_values = out.past_key_values
                    input_ids = out.logits.argmax(-1)

            torch.cuda.synchronize()
            t3 = time.time()

            decode_latency = t3 - t2
            total_latency = t3 - t0

            # 计算总输出 Token (Batch输出总和 - Batch输入总和)
            total_output_tokens = outputs.numel() - input_token_count

            # Overall TPS 计算加入 Input Token
            tps_overall = 0
            if total_latency > 0:
                tps_overall = (input_token_count +
                               total_output_tokens) / total_latency

            metrics = {
                "model": model_name,
                "config": {
                    "batch_size": batch_size,
                    "new_tokens": new_tokens
                },
                "metrics": {
                    "prefill_time_s":
                    round(prefill_latency, 4),
                    "decode_time_s":
                    round(decode_latency, 4),
                    "total_time_s":
                    round(total_latency, 4),
                    "total_input_tokens":
                    input_token_count,
                    "total_output_tokens":
                    total_output_tokens,
                    "tokens_per_second_prefill":
                    round(input_token_count /
                          prefill_latency, 2) if prefill_latency > 0 else 0,
                    "tokens_per_second_decode":
                    round(total_output_tokens /
                          decode_latency, 2) if decode_latency > 0 else 0,
                    "tokens_per_second_overall":
                    round(tps_overall, 2),
                    "request_per_second":
                    round(batch_size /
                          total_latency, 2) if total_latency > 0 else 0,
                    "gpu_mem_gb":
                    self.get_memory_usage()
                },
                "timestamp": datetime.now().isoformat()
            }

            self.logger.info(
                f"📊 结果: Prefill={metrics['metrics']['prefill_time_s']}s | "
                f"Decode={metrics['metrics']['decode_time_s']}s | "
                f"Overall TPS={metrics['metrics']['tokens_per_second_overall']}"
            )
            return metrics

        except torch.cuda.OutOfMemoryError:
            self.logger.error(f"💥 OOM at BS={batch_size}, Tokens={new_tokens}")
            self.clear_cache()
            return {
                "model": model_name,
                "error": "OOM",
                "config": {
                    "batch_size": batch_size,
                    "new_tokens": new_tokens
                }
            }
        except Exception as e:
            self.logger.error(f"❌ 测试出错: {e}")
            return None

    def run_examples(self, batch_size, num_examples=2):
        """运行少量样例用于人工检查"""
        self.logger.info("🧪 生成样例中...")
        # 取前 N 个 prompt (不足则取全部)
        inputs, raw_prompts = self.prepare_batch_inputs(
            min(batch_size, num_examples))
        if inputs is None: return []

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,  # 样例固定长度
                    do_sample=True,  # 样例开启采样，看生成质量
                    temperature=0.7,
                    enable_thinking=False,
                    pad_token_id=self.tokenizer.pad_token_id)

            # 解码 (跳过 input 部分)
            input_len = inputs.input_ids.shape[1]
            generated_ids = outputs[:, input_len:]
            decoded_texts = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)

            examples = []
            for i, text in enumerate(decoded_texts):
                examples.append({
                    "prompt":
                    raw_prompts[i][:50] + "...",
                    "response":
                    text[:200] + "..." if len(text) > 200 else text
                })
            return examples
        except Exception as e:
            self.logger.error(f"❌ 样例生成失败: {e}")
            return []

    def save_final_results(self):
        filename = "benchmark_v2_results.json"
        filepath = os.path.join(CONFIG["result_dir"], filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.all_results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"💾 所有测试结果已保存至: {filepath}")
        except Exception as e:
            self.logger.error(f"保存失败: {e}")

    def run(self):
        if not self.test_data:
            self.logger.error("❌ 无测试数据，终止运行。")
            return

        for model_conf in MODELS_TO_TEST:
            model_name = model_conf['name']
            if not self.load_model(model_conf):
                continue

            for bs in CONFIG["n_batch_size"]:
                for nt in CONFIG["new_tokens"]:
                    self.clear_cache()

                    metrics = self.run_benchmark_step(model_name, bs, nt)

                    if metrics:
                        if "error" not in metrics:
                            examples = self.run_examples(bs)
                            metrics["examples"] = examples

                        self.all_results.append(metrics)

            # 卸载模型
            del self.model
            del self.tokenizer
            self.clear_cache()
            self.logger.info(f"🏁 模型 {model_name} 测试完成。\n")

        self.save_final_results()


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()
