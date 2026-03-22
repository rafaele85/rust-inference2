import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)
model.eval()

token_id = 1  # BOS
embed = model.model.embed_tokens.weight[token_id]
print("embed[:5]:", embed[:5])

# RMSNorm
weight = model.model.layers[0].input_layernorm.weight
variance = embed.pow(2).mean(-1, keepdim=True)
normed = embed * torch.rsqrt(variance + 1e-5) * weight
print("rms_norm[:5]:", normed[:5])
