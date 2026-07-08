#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 推理服务，监听 8000 端口
启动: python3 vllm_server.py
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from vllm import LLM, SamplingParams
import uvicorn

MERGED_DIR = "/root/autodl-tmp/qwen3-0.6b-lora-merged/MO"

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task.\n"
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

app = FastAPI(title="Qwen3-0.6B vLLM Service")

print("[INFO] Loading model with vLLM ...")
llm = LLM(
    model=MERGED_DIR,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.6,
)
print("[INFO] Model loaded.")


class GenerateRequest(BaseModel):
    instruction: str
    max_new_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.1


class GenerateResponse(BaseModel):
    output: str
    instruction: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MERGED_DIR}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    prompt = PROMPT_TEMPLATE.format(instruction=req.instruction)
    sampling_params = SamplingParams(
        max_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )
    outputs = llm.generate([prompt], sampling_params)
    text = outputs[0].outputs[0].text
    return GenerateResponse(output=text, instruction=req.instruction)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

