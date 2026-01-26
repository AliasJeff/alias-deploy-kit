import MNN.llm as mnnllm
import os
import time
import json
import logging
import numpy as np
from datetime import datetime
from typing import List, Dict, Any


class MNNConfig:

    MODEL_PATH = "./models/Qwen3-1.7B-MNN/config.json"

    RESULT_DIR = "results_mnn"
    LOG_DIR = os.path.join(RESULT_DIR, "logs")

    LOOPS = 2

    # 设置生成的最大 Token 数
    MAX_NEW_TOKENS = 256

    TEST_PROMPTS = [
        "设计一个科技公司官网首页，直接输出html代码，不要输出任何其他内容。 </no_think>",
        "设计一个电商平台首页，直接输出html代码，不要输出任何其他内容。 </no_think>",
        "设计一个企业级SaaS后台管理系统仪表盘，直接输出html代码，不要输出任何其他内容。 </no_think>",
        "设计一个新闻资讯类网站首页，直接输出html代码，不要输出任何其他内容。 </no_think>",
    ]

    MODEL_CONFIG = {
        "llm_model": "llm.mnn",
        "llm_weight": "llm.mnn.weight",
        "backend_type": "cpu",
        "thread_num": 4,
        "precision": "low",
        "memory": "low",
        "sampler_type": "mixed",
        "mixed_samplers": ["penalty", "topK", "topP", "min_p", "temperature"],
        "penalty": 1.1,
        "temperature": 0.6,
        "topP": 0.95,
        "topK": 20,
        "min_p": 0,
        "max_new_tokens": 256,
    }


class MNNBenchmark:

    def __init__(self):
        self.cfg = MNNConfig()
        self.results: List[Dict[str, Any]] = []

        # 创建目录
        os.makedirs(self.cfg.RESULT_DIR, exist_ok=True)
        os.makedirs(self.cfg.LOG_DIR, exist_ok=True)

        self.logger = self.setup_logger()
        self.logger.info(f"🚀 初始化 MNN 基准测试")

    def setup_logger(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.cfg.LOG_DIR, f"{today_str}.log")

        logger = logging.getLogger("MNN_Benchmark")
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

    def run(self):
        if not os.path.exists(self.cfg.MODEL_PATH):
            self.logger.error(f"❌ 配置文件 {self.cfg.MODEL_PATH} 不存在。")
            return

        # 1. 加载模型
        self.logger.info(f"⏳ 正在加载模型: {self.cfg.MODEL_PATH} ...")
        start_load = time.time()
        try:
            model = mnnllm.create(self.cfg.MODEL_PATH)
            model.load()
            self.logger.info("🔧 模型加载成功，正在设置配置...")
            model.set_config(self.cfg.MODEL_CONFIG)

            load_time = time.time() - start_load
            self.logger.info(f"✅ 模型加载完毕，耗时: {load_time:.2f}s")
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            return

        # 2. 预热 (Warm-up)
        self.logger.info("🔥 正在预热 (Warm-up)...")
        try:
            # 预热也使用 apply_chat_template
            warmup_msgs = [{"role": "user", "content": "Hello"}]
            warmup_prompt = model.apply_chat_template(warmup_msgs)
            model.response(warmup_prompt, False)
            self.logger.info("✅ 预热完成")
        except Exception as e:
            self.logger.warning(f"⚠️ 预热失败: {e}")

        self.logger.info("-" * 40)
        total_start_time = time.time()

        # 3. 循环执行推理
        count = 0
        for loop_idx in range(self.cfg.LOOPS):
            self.logger.info(
                f"🔄 --- 第 {loop_idx + 1} / {self.cfg.LOOPS} 轮测试 ---")

            for i, p in enumerate(self.cfg.TEST_PROMPTS):
                count += 1
                req_id = f"L{loop_idx}-P{i}"

                # --- 关键修改：使用 apply_chat_template ---
                # 构造消息列表
                messages = [{"role": "user", "content": p}]

                try:
                    # 将 List[Dict] 转换为模型所需的 Prompt 字符串
                    formatted_prompt = model.apply_chat_template(messages)
                except Exception as e:
                    self.logger.error(f"❌ Template 应用失败: {e}")
                    formatted_prompt = p  # 降级处理

                prompt_preview = p[:20].replace('\n', ' ') + "..."
                self.logger.info(
                    f"[{count}] ID:{req_id} | Prompt: {prompt_preview}")

                t0 = time.time()
                error_msg = None
                output = ""
                latency = 0.0
                tokens_per_sec = 0.0
                generated_tokens_count = 0

                try:
                    # 执行推理
                    output = model.response(formatted_prompt, False)
                    latency = time.time() - t0

                    # --- 关键修改：精确获取 Token 数量 ---
                    # 从 context 对象中获取本次生成的 tokens 列表
                    # 你的库代码显示 Context 有 output_tokens 属性
                    try:
                        output_tokens_list = model.context.output_tokens
                        generated_tokens_count = len(output_tokens_list)
                    except:
                        # 兜底：如果 context 获取失败，回退到字符估算
                        generated_tokens_count = int(len(output) / 1.5)

                    # 如果获取到的 token 列表为空但有输出字符串，说明 context 未刷新，回退估算
                    if generated_tokens_count == 0 and len(output) > 0:
                        generated_tokens_count = int(len(output) / 1.5)

                    if latency > 0:
                        tokens_per_sec = generated_tokens_count / latency

                    self.logger.info(
                        f"    -> ✅ 完成 | 耗时: {latency:.2f}s | Tokens: {generated_tokens_count} | Speed: {tokens_per_sec:.2f} t/s"
                    )

                except Exception as e:
                    latency = time.time() - t0
                    error_msg = str(e)
                    self.logger.error(f"    -> ❌ 推理出错: {e}")

                self.results.append({
                    "req_id":
                    req_id,
                    "prompt":
                    p,
                    "output_preview":
                    output[:50] if output else "",
                    "latency":
                    latency,
                    "generated_tokens":
                    generated_tokens_count,
                    "tokens_per_second":
                    tokens_per_sec,
                    "timestamp":
                    time.time(),
                    "error":
                    error_msg
                })

        total_duration = time.time() - total_start_time
        self.save_report(total_duration)

    def save_report(self, total_duration):
        if not self.results:
            return

        success_results = [r for r in self.results if not r['error']]
        avg_speed = np.mean([r['tokens_per_second'] for r in success_results
                             ]) if success_results else 0
        total_tokens = sum([r['generated_tokens'] for r in success_results])

        report = {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "model": self.cfg.MODEL_PATH,
                "max_new_tokens": self.cfg.MAX_NEW_TOKENS
            },
            "summary": {
                "total_requests": len(self.results),
                "total_tokens_generated": total_tokens,
                "avg_tokens_per_sec": round(avg_speed, 2),
                "total_duration_s": round(total_duration, 2)
            },
            "details": self.results
        }

        filename = f"benchmark_mnn_{int(time.time())}.json"
        output_file = os.path.join(self.cfg.RESULT_DIR, filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.logger.info(f"📊 报告已保存: {output_file}")


if __name__ == "__main__":
    benchmark = MNNBenchmark()
    benchmark.run()
