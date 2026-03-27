from safetensors import safe_open
from awq import AutoAWQForCausalLM
import torch, os

model_name = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False, device_map={"": "cuda:0"})

# Get the dequantized weight from layer 0 q_proj
layer = model.model.model.layers[0].self_attn.q_proj


x = torch.eye(896, device="cuda:0", dtype=torch.float16)
with torch.no_grad():
    y = layer(x)
print(f"Shape: {y.shape}")
print(f"First row, first 10: {y[0, :10].tolist()}")
print(f"Last row, first 10: {y[-1, :10].tolist()}")