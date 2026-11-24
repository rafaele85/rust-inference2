from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeAttention,
)
from transformers import AutoConfig
from safetensors import safe_open
from pathlib import Path
import torch

path = "/mnt/data/models/Qwen3.5-122B-A10B-AWQ-4bit"
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
tc = config.text_config

# Load layer 3 attention weights
print("Loading weights...")
tensors = {}
for f in sorted(Path(path).glob("*.safetensors")):
    with safe_open(str(f), framework="pt") as st:
        for k in st.keys():
            if "layers.3.self_attn" in k or "layers.3.input_layernorm" in k:
                tensors[k] = st.get_tensor(k)

attn = Qwen3_5MoeAttention(tc, layer_idx=3)
attn_state = {}
for k, v in tensors.items():
    if "layers.3.self_attn." in k:
        short_k = k.replace("model.language_model.layers.3.self_attn.", "")
        attn_state[short_k] = v
attn.load_state_dict(attn_state, strict=False)
attn = attn.float()

# Dummy input
hidden = torch.full((1, 1, 3072), 0.01)

# Apply input layernorm
input_norm_w = tensors["model.language_model.layers.3.input_layernorm.weight"].float()
norm = Qwen3_5MoeRMSNorm(3072, eps=tc.rms_norm_eps)
norm.weight = torch.nn.Parameter(input_norm_w)
normed = norm(hidden)
print(f"normed first 5: {normed[0,0,:5].tolist()}")

# RoPE for position 0
rope_dim = int(tc.head_dim * tc.rope_parameters["partial_rotary_factor"])
half = rope_dim // 2
theta = tc.rope_parameters["rope_theta"]
freqs = 1.0 / (theta ** (torch.arange(half, dtype=torch.float32) / half))
cos_vals = torch.cat([freqs.cos(), freqs.cos()]).unsqueeze(0).unsqueeze(0)  # pos=0: angle=0, cos=1
sin_vals = torch.cat([freqs.sin(), freqs.sin()]).unsqueeze(0).unsqueeze(0)  # pos=0: sin=0


with torch.no_grad():
    q_gate_flat = attn.q_proj(normed)  # [1, 1, 16384]
    print(f"q_gate_flat[0..5]: {q_gate_flat[0,0,:5].tolist()}")
    print(f"q_gate_flat[256..261]: {q_gate_flat[0,0,256:261].tolist()}")