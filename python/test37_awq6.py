# Python reference: what token does pos 0 ("The", token 785) predict?
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False, device_map={"": "cuda:0"})

input_ids = torch.tensor([[785]], device="cuda:0")
with torch.no_grad():
    logits = model(input_ids).logits
    top5 = logits[0, 0].topk(5)
    print("Token 785 ('The') -> top 5 predictions:")
    for i in range(5):
        tid = top5.indices[i].item()
        val = top5.values[i].item()
        print(f"  {tid} ({tokenizer.decode([tid])}) logit={val:.4f}")
