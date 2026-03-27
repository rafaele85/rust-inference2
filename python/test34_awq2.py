import json, struct, numpy as np
from safetensors import safe_open

model_name = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
import os
home = os.environ["HOME"]
sanitized = model_name.replace("/", "--")
model_dir = f"{home}/.cache/huggingface/hub/models--{sanitized}"
ref = open(f"{model_dir}/refs/main").read().strip()
path = f"{model_dir}/snapshots/{ref}/model.safetensors"

f = safe_open(path, framework="numpy")

# Check i32 loading
name = "model.layers.0.self_attn.q_proj.qweight"
t = f.get_tensor(name)
print(f"{name}: shape={t.shape} dtype={t.dtype}")
print("First 10 values:", t.flatten()[:10].tolist())

# Check f16 loading
name2 = "model.layers.0.self_attn.q_proj.scales"
t2 = f.get_tensor(name2)
print(f"\n{name2}: shape={t2.shape} dtype={t2.dtype}")
print("First 10 values:", t2.flatten()[:10].tolist())
