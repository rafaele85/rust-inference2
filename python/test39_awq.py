import torch, json, os
from safetensors import safe_open

def resolve_hf_path(model_name):
    home = os.environ["HOME"]
    sanitized = model_name.replace("/", "--")
    model_dir = f"{home}/.cache/huggingface/hub/models--{sanitized}"
    with open(f"{model_dir}/refs/main") as f:
        h = f.read().strip()
    return f"{model_dir}/snapshots/{h}"

path = resolve_hf_path("TechxGenus/DeepSeek-V2-Lite-Chat-AWQ")

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
prompt = "<｜begin▁of▁sentence｜>User: What is the capital of France?\n\nAssistant:"
ids = tok.encode(prompt, add_special_tokens=False)
print(f"Token IDs ({len(ids)}): {ids}")
print(f"First token: {ids[0]} = {tok.decode([ids[0]])}")
print(f"Last token: {ids[-1]} = {tok.decode([ids[-1]])}")
