from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False, device_map={"": "cuda:0"})

prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")

print("Input IDs:", input_ids.tolist())

# Token-by-token greedy generation
with torch.no_grad():
    for i in range(20):
        logits = model(input_ids).logits
        next_id = logits[0, -1].argmax().item()
        print(f"  pos {input_ids.shape[1]-1} -> {next_id} ({tokenizer.decode([next_id])})")
        if next_id == 151645:
            break
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device="cuda:0")], dim=1)