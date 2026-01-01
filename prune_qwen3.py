import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from third_party.sparsegpt.sparsegpt import SparseGPT
from third_party.sparsegpt.datautils import get_c4

model_id = "Qwen/Qwen3-1.7B"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Preparing calibration data...")
dataloader, _ = get_c4(
    nsamples=128,
    seed=0,
    seqlen=2048,
    model=model_id,
    tokenizer=tokenizer,
)

print("Starting 2:4 Sparsity Pruning...")
gpts = SparseGPT(model)
gpts.fasterprune(dataloader, prunen=2, prunem=4, percdamp=0.01, blocksize=128)

save_path = "./outputs/models/qwen3-1.7b-sparse-2by4"
print(f"Saving sparse model to {save_path}...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Pruning complete! Saved to {save_path}")
