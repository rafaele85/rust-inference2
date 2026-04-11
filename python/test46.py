import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/data/models/Qwen3.5-122B-A10B-q4-affine-g64-mlx"
# Or use the original HF model if you have it

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokens = tokenizer.encode("The capital of France is")
print("tokens:", tokens)

# If you can load the model in Python (may need lots of RAM):
# model = AutoModelForCausalLM.from_pretrained(model_path)
# with torch.no_grad():
#     out = model(torch.tensor([tokens]))
#     next_token = out.logits[0, -1].argmax().item()
#     print("next token:", next_token)
#     print("decoded:", tokenizer.decode([next_token]))
