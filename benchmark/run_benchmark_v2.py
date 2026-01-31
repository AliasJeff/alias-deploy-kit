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
        "name": "Qwen3-4B-GPTQ-4bit",
        "path": "./models/Qwen3-4B-gptq-4bit",
    },
    {
        "name": "Qwen3-4B",
        "path": "./models/Qwen3-4B"
    },
    {
        "name": "Qwen3-1.7B-GPTQ-4bit",
        "path": "./models/Qwen3-1.7B-gptq-4bit",
    },
    {
        "name": "Qwen3-1.7B",
        "path": "./models/Qwen3-1.7B"
    },
    {
        "name": "Qwen2-1.5B-GPTQ-4bit",
        "path": "./models/Qwen2-1.5B-gptq-4bit",
    },
    {
        "name": "Qwen2-1.5B",
        "path": "./models/Qwen2-1.5B",
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
    "system_prompt": "直接生成html代码，不要输出其他任何内容 </no_think>"
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
        inputs, _ = self.prepare_batch_inputs(batch_size)
        if inputs is None: return None

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        # 计算实际输入的 token 数量 (排除 padding)
        if 'attention_mask' in inputs:
            input_token_count = inputs.attention_mask.sum().item()
        else:
            input_token_count = inputs.input_ids.numel()

        input_seq_len = inputs.input_ids.shape[1]

        self.logger.info(
            f"🚀 开始测试 (Split Metrics): BS={batch_size}, Input={input_seq_len}, MaxTokens={new_tokens}"
        )

        self.clear_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # 测量 Prefill 时间
            torch.cuda.synchronize()
            t_prefill_start = time.perf_counter()

            with torch.inference_mode():
                # 纯 Forward Pass，模拟处理 Prompt 的过程
                _ = self.model(input_ids=inputs.input_ids,
                               attention_mask=inputs.attention_mask)

            torch.cuda.synchronize()
            prefill_latency = time.perf_counter() - t_prefill_start

            del _
            self.clear_cache()

            # 测量 Total (Prefill + Decode) 时间
            torch.cuda.synchronize()
            t_total_start = time.perf_counter()

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=new_tokens,
                    pad_token_id=pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=False)

            torch.cuda.synchronize()
            total_latency = time.perf_counter() - t_total_start
            mem_peak = torch.cuda.max_memory_allocated() / (1024**3)

            generated_ids = outputs[:, input_seq_len:]
            valid_generated_mask = (generated_ids != pad_token_id)
            total_output_tokens = valid_generated_mask.sum().item()

            # 计算 Decode 延迟：总时间 - Prefill 时间
            # NOTE: 这是一种近似计算，假设 generate 内部的 prefill 耗时与我们手动测的一致
            decode_latency = max(0.0001, total_latency - prefill_latency)

            prefill_tps = input_token_count / prefill_latency if prefill_latency > 0 else 0
            decode_tps = total_output_tokens / decode_latency if decode_latency > 0 else 0
            rps = batch_size / total_latency if total_latency > 0 else 0

            metrics = {
                "model": model_name,
                "config": {
                    "batch_size": batch_size,
                    "max_new_tokens": new_tokens
                },
                "metrics": {
                    "total_time_s": round(total_latency, 4),
                    "prefill_time_s": round(prefill_latency, 4),
                    "decode_time_s": round(decode_latency, 4),
                    "total_input_tokens": input_token_count,
                    "total_output_tokens": total_output_tokens,
                    "tokens_per_second_prefill": round(prefill_tps, 2),
                    "tokens_per_second_gen": round(decode_tps, 2),
                    "request_per_second": round(rps, 2),
                    "gpu_mem_peak_gb": round(mem_peak, 2)
                },
                "timestamp": datetime.now().isoformat()
            }

            self.logger.info(
                f"📊 结果: Total={metrics['metrics']['total_time_s']}s | "
                f"Prefill TPS={metrics['metrics']['tokens_per_second_prefill']} | "
                f"Decode TPS={metrics['metrics']['tokens_per_second_gen']} | "
                f"RPS={metrics['metrics']['request_per_second']} | "
                f"Mem={metrics['metrics']['gpu_mem_peak_gb']}GB")

            return metrics

        except torch.cuda.OutOfMemoryError:
            self.logger.error(f"💥 OOM at BS={batch_size}")
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
            self.logger.error(f"❌ 测试出错: {e}", exc_info=True)
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
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id)

            # 解码 (跳过 input 部分)
            input_len = inputs.input_ids.shape[1]
            generated_ids = outputs[:, input_len:]
            decoded_texts = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True)

            examples = []
            for i, text in enumerate(decoded_texts):
                examples.append({"prompt": raw_prompts[i], "response": text})
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

            self.logger.info("⚡ 开始预热")
            dummy_prompt = "Hello"
            inputs = self.tokenizer(dummy_prompt,
                                    return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                _ = self.model.generate(**inputs,
                                        max_new_tokens=1,
                                        do_sample=False,
                                        use_cache=True)

            for bs in CONFIG["n_batch_size"]:
                for nt in CONFIG["new_tokens"]:
                    self.clear_cache()

                    metrics = self.run_benchmark_step(model_name, bs, nt)

                    if metrics:
                        self.all_results.append(metrics)

            # 卸载模型
            del self.model
            del self.tokenizer
            self.clear_cache()
            self.logger.info(f"🏁 模型 {model_name} 测试完成。\n")

        try:
            self.load_model(MODELS_TO_TEST[0])
            examples = self.run_examples(2)
            self.all_results["examples"] = examples
        except:
            pass
        self.save_final_results()


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()
