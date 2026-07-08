#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised Fine-Tuning (SFT) + LoRA for Qwen3-0.6B
Dataset : training_data.json  (instruction / input / output)
GPU     : RTX 5090  32 GB VRAM
"""

import os
import json
import logging

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── paths ───────────────────────────────────────────────────────────────────
MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
DATA_PATH = "/root/autodl-tmp/training_data.json"
OUTPUT_DIR = "/root/autodl-tmp/qwen3-0.6b-lora-sft"
MERGED_DIR = "/root/autodl-tmp/qwen3-0.6b-lora-merged"

# ─── hyper-params ────────────────────────────────────────────────────────────
SEED = 42
MAX_SEQ_LEN = 4096
TRAIN_RATIO = 0.95

# LoRA
LORA_R = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
LORA_TARGET = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
]

# Training
NUM_EPOCHS = 8
BATCH_SIZE = 4  # per device
GRAD_ACCUM = 4  # effective batch = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"
SAVE_STEPS = 200
LOGGING_STEPS = 20
EVAL_STEPS = 200
FP16 = False
BF16 = True  # RTX 5090 supports BF16

# ─── prompt template ─────────────────────────────────────────────────────────
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n")

PROMPT_TEMPLATE_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n")


def build_prompt(instruction: str, inp: str = "") -> str:
    if inp.strip():
        return PROMPT_TEMPLATE_WITH_INPUT.format(instruction=instruction,
                                                 input=inp)
    return PROMPT_TEMPLATE.format(instruction=instruction)


def tokenize_fn(examples, tokenizer):
    input_ids_list = []
    labels_list = []
    attention_masks = []

    for instruction, inp, output in zip(examples["instruction"],
                                        examples["input"], examples["output"]):
        if inp.strip():
            user_content = f"{instruction}\n\n{inp}"
        else:
            user_content = instruction

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": output
            },
        ]

        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": user_content
                },
            ],
            tokenize=True,
            add_generation_prompt=True,
        )

        prompt_len = len(prompt_ids)
        input_ids = full_ids[:MAX_SEQ_LEN]

        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[:MAX_SEQ_LEN]

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_masks.append([1] * len(input_ids))

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_masks,
    }


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)

    # 1. Load tokenizer
    logger.info("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load dataset
    logger.info("Loading dataset ...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset = Dataset.from_list(raw)
    split = dataset.train_test_split(test_size=1 - TRAIN_RATIO, seed=SEED)
    train_ds = split["train"]
    eval_ds = split["test"]
    logger.info(f"Train: {len(train_ds)}  |  Eval: {len(eval_ds)}")

    # 3. Tokenise
    logger.info("Tokenising ...")
    tok_fn = lambda examples: tokenize_fn(examples, tokenizer)
    train_ds = train_ds.map(
        tok_fn,
        batched=True,
        batch_size=500,
        remove_columns=train_ds.column_names,
        desc="Tokenising train",
    )
    eval_ds = eval_ds.map(
        tok_fn,
        batched=True,
        batch_size=500,
        remove_columns=eval_ds.column_names,
        desc="Tokenising eval",
    )

    # 4. Load model
    logger.info("Loading base model ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.enable_input_require_grads()

    # 5. Attach LoRA
    logger.info("Attaching LoRA adapters ...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        fp16=FP16,
        bf16=BF16,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=SEED,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
    )

    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    # 9. Train
    logger.info("Starting training ...")
    trainer.train()

    # 10. Save LoRA adapter + tokenizer
    logger.info(f"Saving LoRA adapter to {OUTPUT_DIR} ...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 11. Merge LoRA weights into base model and save
    logger.info("Merging LoRA weights into base model ...")
    merged = model.merge_and_unload()
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)
    logger.info(f"Merged model saved to {MERGED_DIR}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
