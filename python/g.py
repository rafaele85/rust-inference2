from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeGatedDeltaNet,
)
from transformers import AutoConfig
from safetensors import safe_open
from pathlib import Path
import torch

path = "/mnt/data/models/Qwen3.5-122B-A10B-AWQ-4bit"
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
tc = config.text_config

print("Loading weights...")
tensors = {}
for f in sorted(Path(path).glob("*.safetensors")):
    with safe_open(str(f), framework="pt") as st:
        for k in st.keys():
            if "layers.0.linear_attn" in k or "layers.0.input_layernorm" in k:
                tensors[k] = st.get_tensor(k)

print(f"Loaded {len(tensors)} tensors")

gdn = Qwen3_5MoeGatedDeltaNet(tc, layer_idx=0)
gdn_state = {}
for k, v in tensors.items():
    if "layers.0.linear_attn." in k:
        short_k = k.replace("model.language_model.layers.0.linear_attn.", "")
        gdn_state[short_k] = v

missing, unexpected = gdn.load_state_dict(gdn_state, strict=False)
print(f"Missing: {missing}")
print(f"Unexpected: {unexpected}")

gdn = gdn.float().cuda()

hidden = torch.full((1, 1, 3072), 0.01).cuda()

input_norm_w = tensors["model.language_model.layers.0.input_layernorm.weight"].float().cuda()
norm = Qwen3_5MoeRMSNorm(3072, eps=tc.rms_norm_eps).cuda()
norm.weight = torch.nn.Parameter(input_norm_w)
normed = norm(hidden)
print(f"normed first 5: {normed[0,0,:5].tolist()}")

with torch.no_grad():
    attn_out = gdn(normed)

if isinstance(attn_out, tuple):
    attn_out = attn_out[0]

print(f"attn_out first 10: {attn_out[0,0,:10].tolist()}")
print(f"attn_out max: {attn_out.abs().max().item()}")

result = hidden + attn_out
print(f"result first 10: {result[0,0,:10].tolist()}")
