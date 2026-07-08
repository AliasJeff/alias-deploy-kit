#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference for the fine-tuned Qwen3-0.6B model.

Usage examples:
    # Interactive mode (merged model, default)
    python inference.py

    # Single prompt
    python inference.py --prompt "设计登录页面，意图：用户登录"

    # Load LoRA adapter directly (without merged model)
    python inference.py --use_lora --prompt "设计商品首页"
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MERGED_DIR = "./models/qwen3-0.6b-ours"
LORA_DIR = "./models/qwen3-0.6b-ours"
BASE_DIR = "./models/qwen3-0.6b-ours"

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


def load_model(use_lora: bool = False):
    if use_lora:
        print(f"[INFO] Loading base model + LoRA adapter ...")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_DIR,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, LORA_DIR)
        tokenizer = AutoTokenizer.from_pretrained(LORA_DIR,
                                                  trust_remote_code=True)
    else:
        print(f"[INFO] Loading merged model from {MERGED_DIR} ...")
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_DIR,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR,
                                                  trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("[INFO] Model loaded.")
    return model, tokenizer


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    instruction: str,
    inp: str = "",
    max_new_tokens: int = 4096,
    temperature: float = 0.1,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    prompt = build_prompt(instruction, inp)
    # prompt = instruction
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=(temperature > 0),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B SFT Inference")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Instruction prompt (if omitted, enters interactive mode)")
    parser.add_argument("--input",
                        type=str,
                        default="",
                        help="Optional context input")
    parser.add_argument("--use_lora",
                        action="store_true",
                        help="Load LoRA adapter instead of merged model")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    model, tokenizer = load_model(use_lora=args.use_lora)

    if args.prompt:
        # ── single-shot ──────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(f"Instruction : {args.prompt}")
        if args.input:
            print(f"Input       : {args.input}")
        print("=" * 60)
        response = generate(
            model,
            tokenizer,
            instruction=args.prompt,
            inp=args.input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\nOutput:\n{response}\n")
    else:
        # ── interactive loop ─────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Interactive inference  (type 'quit' or Ctrl-C to exit)")
        print("=" * 60)
        while True:
            try:
                instruction = input("\nInstruction: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                sys.exit(0)

            if instruction.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            if not instruction:
                continue

            inp = input("Input (leave blank if none): ").strip()

            print("\nGenerating ...")
            response = generate(
                model,
                tokenizer,
                instruction=instruction,
                inp=inp,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"\nOutput:\n{response}")


if __name__ == "__main__":
    main()
