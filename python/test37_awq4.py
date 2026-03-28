from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False, device_map={"": "cuda:0"})

# Get the embedding for token 785 ("The")
embed = model.model.model.embed_tokens.weight[785].float().cpu()
print(f"embed shape: {embed.shape}")
print(f"embed[:10]: {embed[:10].tolist()}")

# Run q_proj on it
layer = model.model.model.layers[0].self_attn.q_proj
with torch.no_grad():
    inp = embed.unsqueeze(0).half().to("cuda:0")
    out = layer(inp)
    out = out.squeeze(0).float().cpu()
print(f"q_proj output shape: {out.shape}")
print(f"q_proj output[:10]: {out[:10].tolist()}")


# What dtype is the bias?
f = safe_open(path, framework="numpy")
t = f.get_tensor("model.layers.0.self_attn.q_proj.bias")
print(f"bias dtype: {t.dtype}")