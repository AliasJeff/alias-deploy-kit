import os
import json
import time
import torch
import logging
import gc
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# 优化显存分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    "output_json_path":
    "./qa_results.json",
    "log_dir":
    "./logs/chat_logs",
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


class torchAutoQA:

    def __init__(self):
        os.makedirs(CONFIG["log_dir"], exist_ok=True)
        self.logger = self.setup_logger()
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.all_results = {}

    def setup_logger(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(CONFIG["log_dir"], f"auto_qa_{today_str}.log")
        logger = logging.getLogger("TorchAutoQA")
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()

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
        # 彻底释放显存
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def load_model(self, model_info):
        model_name = model_info["name"]
        model_path = model_info["path"]

        self.logger.info(f"📥 正在加载模型: {model_name}...")
        self.clear_cache()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True)

            # 自动识别设备映射，GPTQ 模型会自动识别并加载
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True).eval()  # 切换到推理模式

            self.logger.info(f"✅ 模型 {model_name} 加载成功！设备: {self.model.device}")
            return True
        except Exception as e:
            self.logger.error(f"❌ 模型 {model_name} 加载失败: {e}")
            return False

    def process_questions(self, model_info):
        model_name = model_info["name"]
        questions = CONFIG.get("questions", [])
        self.all_results[model_name] = []

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

            # 使用 chat_template 构造输入
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text],
                                          return_tensors="pt").to(self.device)

            try:
                t_start = time.perf_counter()

                # 开始推理
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=CONFIG["max_new_tokens"],
                        temperature=CONFIG["temperature"],
                        top_p=CONFIG["top_p"],
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id)

                latency = time.perf_counter() - t_start

                # 提取生成的回复部分（去掉输入的 prompt 部分）
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in
                    zip(model_inputs.input_ids, generated_ids)
                ]

                response_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True)[0]

                input_tokens = model_inputs.input_ids.shape[1]
                output_tokens = len(generated_ids[0])
                tps = output_tokens / latency if latency > 0 else 0

                result_item = {
                    "id": idx,
                    "question": q,
                    "response": response_text.strip(),
                    "metrics": {
                        "total_latency_s": round(latency, 4),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "tokens_per_second": round(tps, 2)
                    },
                    "timestamp": datetime.now().isoformat()
                }
                self.all_results[model_name].append(result_item)
                self.logger.info(f"✅ 完成 | 耗时: {latency:.2f}s | TPS: {tps:.2f}")

            except Exception as e:
                self.logger.error(f"❌ 处理问题时出错: {e}")

    def save_results(self):
        output_path = CONFIG["output_json_path"]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, ensure_ascii=False, indent=4)
        self.logger.info(f"💾 结果已保存至: {output_path}")

    def run(self):
        for model_info in CONFIG.get("models", []):
            self.logger.info(f"\n{'='*40}\n评测: {model_info['name']}\n{'='*40}")
            if self.load_model(model_info):
                self.process_questions(model_info)
            self.clear_cache()
        self.save_results()


if __name__ == "__main__":
    app = torchAutoQA()
    app.run()
