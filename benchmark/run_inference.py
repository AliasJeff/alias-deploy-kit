import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

CONFIG = {
    # 模型配置
    "model_name":
    "Qwen3-1.7B-GPTQ-2bit",
    "model_path":
    "./models/Qwen3-1.7B-gptq-2bit",

    # 输出与日志配置
    "output_json_path":
    "./qa_results.json",
    "log_dir":
    "./logs/chat_logs",

    # 长度限制
    "max_model_len":
    4096,

    # 采样参数
    "temperature":
    0.7,
    "top_p":
    0.9,
    "max_new_tokens":
    1024,
    "system_prompt":
    "你是一个人工智能助手，请清晰、准确、有逻辑地回答用户的问题。",
    "questions": [
        "请简要介绍一下什么是大型语言模型（LLM）？",
        "用Python写一个快速排序算法，并写上详细注释。",
        "量子计算和传统计算最大的区别是什么？",
        "给我讲一个关于程序员和咖啡的冷笑话。",
    ]
}


class VLLMAutoQA:

    def __init__(self):
        os.makedirs(CONFIG["log_dir"], exist_ok=True)
        self.logger = self.setup_logger()
        self.tokenizer = None
        self.model = None
        self.results = []

    def setup_logger(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(CONFIG["log_dir"], f"auto_qa_{today_str}.log")

        logger = logging.getLogger("VLLMAutoQA")
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

    def clear_cache(self):
        """强制清理显存和垃圾回收"""
        if self.model is not None:
            del self.model
            self.model = None
            destroy_model_parallel()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def load_model(self):
        self.logger.info(f"📥 正在加载模型: {CONFIG['model_name']} ...")
        self.clear_cache()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                CONFIG['model_path'], trust_remote_code=True)

            self.model = LLM(model=CONFIG['model_path'],
                             trust_remote_code=True,
                             max_model_len=CONFIG["max_model_len"],
                             enforce_eager=False,
                             disable_log_stats=True)
            self.logger.info("✅ 模型加载成功！")
            return True
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {e}")
            return False

    def process_questions(self):
        """遍历并处理所有预设问题"""
        questions = CONFIG.get("questions", [])
        if not questions:
            self.logger.warning("⚠️ 没有找到预填的问题。")
            return

        sampling_params = SamplingParams(temperature=CONFIG["temperature"],
                                         top_p=CONFIG["top_p"],
                                         max_tokens=CONFIG["max_new_tokens"],
                                         ignore_eos=False)

        self.logger.info(f"🚀 开始批量回答，共 {len(questions)} 个问题...")

        for idx, q in enumerate(questions, 1):
            self.logger.info(f"[{idx}/{len(questions)}] 正在生成: {q[:20]}...")

            # 每次提问重置上下文，保证问题独立性 (Single-turn)
            messages = [{
                "role": "system",
                "content": CONFIG["system_prompt"]
            }, {
                "role": "user",
                "content": q
            }]

            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            try:
                t_start = time.perf_counter()
                outputs = self.model.generate([prompt],
                                              sampling_params,
                                              use_tqdm=False)
                latency = time.perf_counter() - t_start

                output = outputs[0]
                response_text = output.outputs[0].text.strip()
                input_tokens = len(output.prompt_token_ids)
                output_tokens = len(output.outputs[0].token_ids)

                # 计算生成速度 (TPS)
                tps = output_tokens / latency if latency > 0 else 0

                # 保存结果
                result_item = {
                    "id": idx,
                    "question": q,
                    "response": response_text,
                    "metrics": {
                        "total_latency_s": round(latency, 4),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "tokens_per_second": round(tps, 2)
                    },
                    "timestamp": datetime.now().isoformat()
                }
                self.results.append(result_item)

                self.logger.info(
                    f"✅ 完成 | 耗时: {latency:.2f}s | 输出Tokens: {output_tokens} | TPS: {tps:.2f}\n"
                )

            except Exception as e:
                self.logger.error(f"❌ 处理问题时出错: {e}")
                self.results.append({
                    "id": idx,
                    "question": q,
                    "error": str(e)
                })

    def save_results(self):
        """将结果和指标保存为 JSON 文件"""
        output_path = CONFIG["output_json_path"]
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"💾 所有问题处理完毕！结果已保存至: {output_path}")
        except Exception as e:
            self.logger.error(f"❌ 保存 JSON 失败: {e}")

    def run(self):
        if not self.load_model():
            return

        self.process_questions()
        self.save_results()
        self.clear_cache()


if __name__ == "__main__":
    app = VLLMAutoQA()
    app.run()
