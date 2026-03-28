import torch
from awq import AutoAWQForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False, device_map={"": "cuda:0"})
m = model.model.model

embed = m.embed_tokens(torch.tensor([[785]], device="cuda:0")).half()
normed = m.layers[0].input_layernorm(embed)

q_proj = m.layers[0].self_attn.q_proj
k_proj = m.layers[0].self_attn.k_proj

with torch.no_grad():
    q = q_proj(normed)
    k = k_proj(normed)
print(f"q[:5]: {q[0, :5].float().tolist()}")
print(f"k[:5]: {k[0, :5].float().tolist()}")