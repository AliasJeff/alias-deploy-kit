import os
# 设置环境变量以减少显存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 禁用 vLLM 默认的用法统计，加快启动速度并减少网络请求
os.environ["VLLM_NO_USAGE_STATS"] = "1"

import json
import time
import torch
import logging
import gc
from datetime import datetime
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

MODELS_TO_TEST = [
    {
        "name": "Qwen3-4B-GPTQ-4bit",
        "path": "./models/Qwen3-4B-gptq-4bit",
    },
    {
        "name": "Qwen3-4B-AWQ-GEMM-4bit",
        "path": "./models/Qwen3-4B-awq-gemm-4bit",
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
        "name": "Qwen3-1.7B-AWQ-GEMM-4bit",
        "path": "./models/Qwen3-1.7B-awq-gemm-4bit",
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
        "name": "Qwen2-1.5B-AWQ-GEMM-4bit",
        "path": "./models/Qwen2-1.5B-awq-gemm-4bit",
    },
    {
        "name": "Qwen2-1.5B",
        "path": "./models/Qwen2-1.5B",
    },
]

CONFIG = {
    "data_path": "./data/test_benchmark.json",
    "result_dir": "./benchmark_results_vllm",
    "log_dir": "./benchmark_logs",
    "max_vram_gb": 7.5,
    "max_model_len": 4096,
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
        self.tokenizer = None
        self.model = None
        self.test_data = self.load_test_data()

    def setup_logger(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(CONFIG["log_dir"],
                                f"bench_vllm_{today_str}.log")

        logger = logging.getLogger("Benchmark")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

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

    def clear_cache(self):
        """强制清理显存和垃圾回收 (包含 vLLM 状态清理)"""
        # 如果模型存在，彻底删除以释放显存
        if self.model is not None:
            del self.model
            self.model = None
            # vLLM 需要清理分布式状态才能完全释放
            destroy_model_parallel()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # 强制同步以确保释放
            torch.cuda.synchronize()

    def _calculate_gpu_utilization(self):
        """将设定的 GB 转换为 vLLM 的 gpu_memory_utilization 比例"""
        if not torch.cuda.is_available():
            return 0.9  # 默认值

        max_gb = CONFIG.get("max_vram_gb", None)
        if max_gb is None or max_gb <= 0:
            return 0.9

        device_idx = torch.cuda.current_device()
        total_bytes = torch.cuda.get_device_properties(device_idx).total_memory
        total_gb = total_bytes / (1024**3)

        # vLLM 需要留一部分给 PyTorch 运行时的开销
        utilization = max(0.1, min(0.99, max_gb / total_gb))
        self.logger.info(
            f"🧩 vLLM 显存分配比例 (gpu_memory_utilization): {utilization:.4f} (约 {max_gb}GB / {total_gb:.2f}GB)"
        )
        return utilization

    def load_model(self, model_conf):
        self.logger.info(
            f"📥 正在加载模型 (vLLM): {model_conf['name']} ({model_conf['path']})...")
        self.clear_cache()

        gpu_util = self._calculate_gpu_utilization()

        try:
            # 显式加载 tokenizer，用于套用 chat_template
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_conf['path'], trust_remote_code=True)

            # vLLM 加载模型
            # vLLM 会自动识别 AWQ/GPTQ，无需手动指定 quantization
            self.model = LLM(
                model=model_conf['path'],
                trust_remote_code=True,
                gpu_memory_utilization=gpu_util,
                max_model_len=CONFIG["max_model_len"],
                enforce_eager=False,  # 设为 False 默认启用 CUDA Graph，极大提升小 batch 推理速度
                disable_log_stats=True  # 压测时关闭 vLLM 默认的控制台指标输出，避免刷屏
            )

            self.logger.info("✅ 模型加载成功 (预分配显存完毕)")
            return True
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            return False

    def prepare_batch_inputs(self, batch_size):
        """
        为 vLLM 准备 Batch 输入数据。
        与 HF 不同，vLLM 直接接受字符串列表，不需要我们手动 tokenization 和 padding。
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

        return formatted_texts, target_prompts

    def run_benchmark_step(self, model_name, batch_size, new_tokens):
        prompts, _ = self.prepare_batch_inputs(batch_size)
        if prompts is None: return None

        self.logger.info(f"🚀 开始测试: BS={batch_size}, MaxTokens={new_tokens}")

        # 设置 vLLM 采样参数
        # 压测模式下，设置 ignore_eos=True 强制生成指定的 new_tokens 数量，保证测试一致性
        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=new_tokens,
                                         ignore_eos=True)

        torch.cuda.reset_peak_memory_stats()

        try:
            t_start = time.perf_counter()

            # vLLM 的 generate() 会自动批处理并返回结果列表
            outputs = self.model.generate(prompts,
                                          sampling_params,
                                          use_tqdm=False)

            total_latency = time.perf_counter() - t_start
            mem_peak = torch.cuda.max_memory_allocated() / (1024**3)

            # --- 从 vLLM 请求级指标提取详细时间 ---
            # vLLM >= 0.4.x 支持 RequestOutput.metrics
            total_input_tokens = 0
            total_output_tokens = 0
            prefill_times = []
            decode_times = []

            for output in outputs:
                input_len = len(output.prompt_token_ids)
                output_len = len(output.outputs[0].token_ids)
                total_input_tokens += input_len
                total_output_tokens += output_len

                # 提取详细指标 (TTFT = Time To First Token)
                if hasattr(output, 'metrics') and output.metrics is not None:
                    m = output.metrics
                    if m.first_token_time and m.arrival_time:
                        prefill_times.append(m.first_token_time -
                                             m.arrival_time)
                    if m.finished_time and m.first_token_time:
                        # 减去生成最后 token 的开销才是纯粹的 decode 时间跨度
                        decode_times.append(m.finished_time -
                                            m.first_token_time)

            # 计算平均/总计指标
            # 如果能提取到指标，则精确计算；如果提取不到，则做粗略估算
            if prefill_times and decode_times:
                # 在连续批处理中，Batch 的 Prefill 时间近似等于最慢的那个 request 的 TTFT
                avg_prefill = sum(prefill_times) / len(prefill_times)
                # Decode 时间是整体耗时减去首 token 耗时
                total_decode_latency = max(0.0001, total_latency - avg_prefill)
                total_prefill_latency = avg_prefill
            else:
                # 兼容旧版本 vLLM fallback
                total_prefill_latency = total_latency * 0.15  # 粗略假设
                total_decode_latency = total_latency * 0.85

            prefill_tps = total_input_tokens / total_prefill_latency if total_prefill_latency > 0 else 0
            decode_tps = total_output_tokens / total_decode_latency if total_decode_latency > 0 else 0
            rps = batch_size / total_latency if total_latency > 0 else 0

            metrics = {
                "model": model_name,
                "config": {
                    "batch_size": batch_size,
                    "max_new_tokens": new_tokens
                },
                "metrics": {
                    "total_time_s": round(total_latency, 4),
                    "prefill_time_s": round(total_prefill_latency, 4),
                    "decode_time_s": round(total_decode_latency, 4),
                    "total_input_tokens": total_input_tokens,
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

        except Exception as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                self.logger.error(f"💥 OOM at BS={batch_size}")
                return {
                    "model": model_name,
                    "error": "OOM",
                    "config": {
                        "batch_size": batch_size,
                        "new_tokens": new_tokens
                    }
                }
            else:
                self.logger.error(f"❌ 测试出错: {e}", exc_info=True)
                return None

    def run_examples(self, batch_size, num_examples=2):
        """运行少量样例用于人工检查"""
        self.logger.info("🧪 生成样例中...")
        prompts, raw_prompts = self.prepare_batch_inputs(
            min(batch_size, num_examples))
        if not prompts: return []

        try:
            # 样例生成使用正常的采样策略，允许遇到 EOS 停止
            sampling_params = SamplingParams(temperature=0.7,
                                             max_tokens=256,
                                             ignore_eos=False)

            outputs = self.model.generate(prompts,
                                          sampling_params,
                                          use_tqdm=False)

            examples = []
            for i, output in enumerate(outputs):
                examples.append({
                    "prompt": raw_prompts[i],
                    "response": output.outputs[0].text
                })
            return examples
        except Exception as e:
            self.logger.error(f"❌ 样例生成失败: {e}")
            return []

    def save_final_results(self):
        filename = "benchmark_vllm_results.json"
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
            dummy_prompt = ["Hello"]
            dummy_sp = SamplingParams(max_tokens=1)
            self.model.generate(dummy_prompt, dummy_sp, use_tqdm=False)

            for bs in CONFIG["n_batch_size"]:
                for nt in CONFIG["new_tokens"]:
                    metrics = self.run_benchmark_step(model_name, bs, nt)
                    if metrics:
                        self.all_results.append(metrics)

            # 测试完当前模型，进行深度清理以便加载下一个模型
            self.clear_cache()
            self.logger.info(f"🏁 模型 {model_name} 测试完成。\n")

        self.logger.info("🧪 准备运行样例生成测试...")
        # 重新加载第一个模型跑样例
        if len(MODELS_TO_TEST) > 0 and self.load_model(MODELS_TO_TEST[0]):
            try:
                examples = self.run_examples(2)
                if examples:
                    self.all_results.append({
                        "model": MODELS_TO_TEST[0]['name'],
                        "type": "example_outputs",
                        "data": examples
                    })
            except Exception as e:
                self.logger.error(f"❌ 运行样例阶段发生错误: {e}")
            finally:
                self.clear_cache()
        else:
            self.logger.warning("⚠️ 模型重新加载失败，跳过样例生成阶段。")

        self.save_final_results()


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()
