from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)
model.eval()

with torch.no_grad():
    embed = model.model.embed_tokens(torch.tensor([[1]]))
    normed = model.model.layers[0].input_layernorm(embed)
    v = model.model.layers[0].self_attn.v_proj(normed)
    print("v[:5]:", v[0, 0, :5])
