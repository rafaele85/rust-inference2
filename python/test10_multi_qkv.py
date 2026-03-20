import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./model", torch_dtype=torch.float32)
model.eval()

input_ids = torch.tensor([[464, 3139, 286, 4881, 318]])
with torch.no_grad():
    wte = model.transformer.wte(input_ids)
    wpe = model.transformer.wpe(torch.arange(5).unsqueeze(0))
    hidden = wte + wpe
    ln_out = model.transformer.h[0].ln_1(hidden)
    qkv_weight = model.transformer.h[0].attn.c_attn.weight
    qkv_bias = model.transformer.h[0].attn.c_attn.bias
    qkv = ln_out @ qkv_weight + qkv_bias
    q, k, v = qkv.split(768, dim=-1)

for v in q[0][-1][:10].tolist():
    print(f"{v:.4f}")
