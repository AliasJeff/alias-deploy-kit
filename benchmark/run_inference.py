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
    "models": [
        {
            "name": "Qwen3-1.7B-GPTQ-4bit",
            "path": "./models/Qwen3-1.7B-gptq-4bit"
        },
        {
            "name": "Qwen3-1.7B",
            "path": "./models/Qwen3-1.7B"
        },
    ],

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
    2048,
    "system_prompt":
    "You are a helpful assistant",
    "questions": [
        "设计订单确认页面，意图：提交订单",
        "设计订单列表页面，意图：查看所有订单",
        "设计餐厅列表页，意图：查看餐厅详情",
        "设计支付页面，意图：完成支付",
        "设计新闻详情页，意图：点赞新闻",
        "设计搜索页面，意图：查看搜索历史",
        "设计评价页面，意图：提交评价",
        "设计商家入驻页，意图：提交资料",
        "设计叫车等待页，意图：查看附近车辆",
        "设计行程列表页，意图：查看行程详情",
    ]
}


class VLLMAutoQA:

    def __init__(self):
        os.makedirs(CONFIG["log_dir"], exist_ok=True)
        self.logger = self.setup_logger()
        self.tokenizer = None
        self.model = None
        self.all_results = {}

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
        if self.model is not None:
            del self.model
            self.model = None
            destroy_model_parallel()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def load_model(self, model_info):
        model_name = model_info["name"]
        model_path = model_info["path"]

        self.logger.info(f"📥 正在加载模型: {model_name} (路径: {model_path}) ...")
        self.clear_cache()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True)

            self.model = LLM(model=model_path,
                             trust_remote_code=True,
                             max_model_len=CONFIG["max_model_len"],
                             enforce_eager=False,
                             disable_log_stats=True)
            self.logger.info(f"✅ 模型 {model_name} 加载成功！")
            return True
        except Exception as e:
            self.logger.error(f"❌ 模型 {model_name} 加载失败: {e}")
            return False

    def process_questions(self, model_info):
        model_name = model_info["name"]
        questions = CONFIG.get("questions", [])
        if not questions:
            self.logger.warning("⚠️ 没有找到预填的问题。")
            return

        self.all_results[model_name] = []

        sampling_params = SamplingParams(temperature=CONFIG["temperature"],
                                         top_p=CONFIG["top_p"],
                                         max_tokens=CONFIG["max_new_tokens"],
                                         ignore_eos=False)

        self.logger.info(
            f"🚀 开始让 {model_name} 进行批量回答，共 {len(questions)} 个问题...")

        for idx, q in enumerate(questions, 1):
            self.logger.info(f"[{idx}/{len(questions)}] 正在生成: {q[:20]}...")

            messages = [
                {
                    "role": "system",
                    "content": CONFIG["system_prompt"]
                },
                {
                    "role": "user",
                    "content": q
                },
            ]

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

                tps = output_tokens / latency if latency > 0 else 0

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
                self.all_results[model_name].append(result_item)

                self.logger.info(
                    f"✅ 完成 | 耗时: {latency:.2f}s | 输出Tokens: {output_tokens} | TPS: {tps:.2f}\n"
                )

            except Exception as e:
                self.logger.error(f"❌ 处理问题时出错: {e}")
                self.all_results[model_name].append({
                    "id": idx,
                    "question": q,
                    "error": str(e)
                })

    def save_results(self):
        output_path = CONFIG["output_json_path"]
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.all_results, f, ensure_ascii=False, indent=4)
            self.logger.info(f"💾 所有模型测试完毕！汇总结果已保存至: {output_path}")
        except Exception as e:
            self.logger.error(f"❌ 保存 JSON 失败: {e}")

    def run(self):
        models_to_test = CONFIG.get("models", [])
        if not models_to_test:
            self.logger.error("❌ 配置文件中未发现任何模型！")
            return

        for model_info in models_to_test:
            self.logger.info(
                f"\n{'='*40}\n开始评测模型: {model_info['name']}\n{'='*40}")

            if self.load_model(model_info):
                self.process_questions(model_info)
            else:
                self.logger.warning(f"⚠️ 跳过模型 {model_info['name']} 的测试。")

            self.clear_cache()
            self.logger.info(f"🧹 模型 {model_info['name']} 的显存已清理。")

        self.save_results()


if __name__ == "__main__":
    app = VLLMAutoQA()
    app.run()
