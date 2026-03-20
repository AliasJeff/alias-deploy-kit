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
]

CONFIG = {
    "data_path":
    "./data/test_benchmark.json",
    "result_dir":
    "./benchmark_results_vllm",
    "log_dir":
    "./benchmark_logs",
    "max_vram_gb":
    7.4,
    "max_model_len":
    4096,
    "n_batch_size": [
        50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375,
        400, 425, 450, 475, 500
    ],
    "new_tokens": [80, 100, 128, 140, 160],
    "system_prompt":
    "直接生成html代码，不要输出其他任何内容 </no_think>"
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
        """动态计算 vLLM 的 gpu_memory_utilization 比例，避免超过真实可用显存"""
        if not torch.cuda.is_available():
            return 0.9

        device_idx = torch.cuda.current_device()

        # 获取真实空闲显存和物理总显存
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)

        max_gb = CONFIG.get("max_vram_gb", 7.5)

        # 核心修改：实际分配绝对不能超过当前空闲显存，并预留 0.3GB 防爆显存
        safe_alloc_gb = min(max_gb, free_gb - 0.3)
        if safe_alloc_gb <= 0:
            safe_alloc_gb = 0.5  # 兜底防止报错

        # vLLM 需要的参数是基于“物理总显存”的比例
        utilization = max(0.1, min(0.99, safe_alloc_gb / total_gb))

        self.logger.info(
            f"🧩 vLLM 显存分配比例: {utilization:.4f} "
            f"(计划分配约 {safe_alloc_gb:.2f}GB / 真实空闲 {free_gb:.2f}GB / 总计 {total_gb:.2f}GB)"
        )
        return float(utilization)

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

    def calibrate_time_ratio(self, batch_size):
        """
        通过探针请求获取模型的 prefill 和 decode 时间的基准比例，代替默认的 0.15 假设。
        算法：
        1. measure latency for input_len=1000, output_len=1 -> 得到 prefill latency (1000 tokens)
        2. measure latency for input_len=1, output_len=1 -> 得到 average latency A
        3. measure latency for input_len=1, output_len=1000 -> 得到 average latency B
        4. (B - A) / 999 得到 decode latency per token
        """
        self.logger.info(
            f"⚖️ 开始耗时探针校准，计算真实 Prefill/Decode 比例 (BS={batch_size})...")
        try:
            # 尝试获取一个合法的 token_id，避免 OOV 或者特殊的 EOS token 导致提前结束
            try:
                token_id = self.tokenizer.encode("Hello")[-1]
            except Exception:
                token_id = 10

            def _measure(in_len, out_len):
                token_ids = [[token_id] * in_len for _ in range(batch_size)]
                sp = SamplingParams(temperature=0.0,
                                    max_tokens=out_len,
                                    ignore_eos=True)

                t0 = time.perf_counter()
                try:
                    # 较新版本 vLLM 支持传入 Dict 形式
                    prompts_dict = [{
                        "prompt_token_ids": ids
                    } for ids in token_ids]
                    self.model.generate(prompts=prompts_dict,
                                        sampling_params=sp,
                                        use_tqdm=False)
                except Exception:
                    # 兼容旧版本 vLLM 直接传 prompt_token_ids 的方式
                    self.model.generate(prompts=None,
                                        prompt_token_ids=token_ids,
                                        sampling_params=sp,
                                        use_tqdm=False)
                return time.perf_counter() - t0

            # 预热一下，避免首次推理包含 CUDA 初始化的额外开销
            _measure(1, 1)

            # 1. measure latency for input_len=1000, output_len=1
            latency_1000_1 = _measure(1000, 1)

            # 2. measure latency for input_len=1, output_len=1 (A)
            latency_A = _measure(1, 1)

            # 3. measure latency for input_len=1, output_len=1000 (B)
            latency_B = _measure(1, 1000)

            # 计算单 token 的 Decode 耗时
            self.calib_decode_per_token = max((latency_B - latency_A) / 999.0,
                                              1e-6)

            # latency_1000_1 包含 1000 个 token 的 prefill 和 1 个 token 的 decode
            # 我们按照你的算法思路，将其除以 1000 得到单个 input_token 的 prefill 平均耗时估算
            self.calib_prefill_per_token = max(latency_1000_1 / 1000.0, 1e-6)

            self.logger.info(
                f"✅ 校准完成: 单Token Prefill耗时 ≈ {self.calib_prefill_per_token*1000:.3f}ms, "
                f"单Token Decode耗时 ≈ {self.calib_decode_per_token*1000:.3f}ms")

        except Exception as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                self.logger.warning(
                    f"⚠️ 探针校准时 OOM (BS={batch_size})，退回 0.15 假设。")
            else:
                self.logger.warning(f"⚠️ 探针校准失败: {e}，退回 0.15 假设。")
            self.calib_prefill_per_token = None
            self.calib_decode_per_token = None

    def prepare_batch_inputs(self, batch_size):
        """
        为 vLLM 准备 Batch 输入数据。
        与 HF 不同，vLLM 直接接受字符串列表，不需要我们手动 tokenization 和 padding。
        """
        sorted_data = sorted(self.test_data, key=lambda x: len(x['prompt']))

        raw_prompts = [item['prompt'] for item in sorted_data]
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
                # ====== 用探针得出的系数动态推算耗时分布 ======
                calib_prefill = getattr(self, 'calib_prefill_per_token', None)
                calib_decode = getattr(self, 'calib_decode_per_token', None)

                if calib_prefill is not None and calib_decode is not None:
                    # 根据实际批次生成的 tokens 总数估算相对耗时。
                    # （注：这里用 total_tokens 相当于分子分母都多乘了一个 batch_size，
                    #  但在求 prefill_ratio 的百分比时会被完美抵消，结果完全等价且精确）
                    est_prefill = total_input_tokens * calib_prefill
                    est_decode = total_output_tokens * calib_decode
                    total_est = est_prefill + est_decode

                    prefill_ratio = est_prefill / total_est if total_est > 0 else 0.15
                else:
                    # 兼容旧版本 vLLM fallback（极端报错时的兜底）
                    prefill_ratio = 0.15

                total_prefill_latency = total_latency * prefill_ratio
                total_decode_latency = total_latency * (1.0 - prefill_ratio)
                # =========================================================

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

                self.calibrate_time_ratio(bs)

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
