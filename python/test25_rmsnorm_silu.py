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

# After the rms_norm from before
q_proj = model.model.layers[0].self_attn.q_proj.weight
q = normed @ q_proj.T
print("q[:5]:", q[:5])

gate_proj = model.model.layers[0].mlp.gate_proj.weight
up_proj = model.model.layers[0].mlp.up_proj.weight
gate = normed @ gate_proj.T
up = normed @ up_proj.T
silu_gate = torch.nn.functional.silu(gate)
print("silu_gate[:5]:", silu_gate[:5])