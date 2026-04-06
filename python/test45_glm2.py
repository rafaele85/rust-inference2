import torch
import math

from safetensors import safe_open
import json, os, glob

path = os.path.expanduser("~/.cache/huggingface/hub/models--cyankiwi--GLM-4.7-Flash-AWQ-4bit")
refs = open(f"{path}/refs/main").read().strip()
model_path = f"{path}/snapshots/{refs}"

# Load weights
tensors = {}
st_files = sorted(glob.glob(f"{model_path}/model*.safetensors"))
for f in st_files:
    with safe_open(f, framework="pt") as st:
        for key in st.keys():
            if "layers.0." in key or key in ["model.embed_tokens.weight", "model.norm.weight"]:
                tensors[key] = st.get_tensor(key)

# Config
eps = 1e-5
rope_theta = 1000000.0
qk_nope_head_dim = 192
qk_rope_head_dim = 64
num_heads = 20
head_dim = qk_nope_head_dim + qk_rope_head_dim  # 256
v_head_dim = 256
kv_lora_rank = 512

# Embed token 27
embed = tensors["model.embed_tokens.weight"].float()
hidden = embed[27].clone()

# RMS norm
norm_w = tensors["model.layers.0.input_layernorm.weight"].float()
variance = hidden.pow(2).mean()
normed = hidden * torch.rsqrt(variance + eps) * norm_w
print(f"normed[:5] = {normed[:5].tolist()}")

# Q LoRA path
q_a_w = tensors["model.layers.0.self_attn.q_a_proj.weight"].float()
q_a_ln_w = tensors["model.layers.0.self_attn.q_a_layernorm.weight"].float()
q_latent = normed @ q_a_w.T
variance = q_latent.pow(2).mean()
q_normed = q_latent * torch.rsqrt(variance + eps) * q_a_ln_w
print(f"q_normed[:5] = {q_normed[:5].tolist()}")

# Q expand
q_b_w = tensors["model.layers.0.self_attn.q_b_proj.weight"].float()
q_full = q_normed @ q_b_w.T
print(f"q_full[:5] = {q_full[:5].tolist()}")

# KV path
kv_a_w = tensors["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"].float()
kv_a_ln_w = tensors["model.layers.0.self_attn.kv_a_layernorm.weight"].float()
kv_a = normed @ kv_a_w.T
c_cache = kv_a[:kv_lora_rank]
k_pe = kv_a[kv_lora_rank:]

variance = c_cache.pow(2).mean()
c_normed = c_cache * torch.rsqrt(variance + eps) * kv_a_ln_w
print(f"c_normed[:5] = {c_normed[:5].tolist()}")
print(f"k_pe[:5] = {k_pe[:5].tolist()}")

# KV expand
kv_b_w = tensors["model.layers.0.self_attn.kv_b_proj.weight"].float()
kv_b = c_normed @ kv_b_w.T
print(f"kv_b[:5] = {kv_b[:5].tolist()}")

# RoPE freqs for position 0
half = qk_rope_head_dim // 2
freqs = torch.tensor([1.0 / (rope_theta ** (i / half)) for i in range(half)])
angles = 0 * freqs
cos_vals = torch.cos(angles)
sin_vals = torch.sin(angles)

# Apply interleaved RoPE to Q
q_roped = q_full.clone()
for h in range(num_heads):
    offset = h * head_dim + qk_nope_head_dim
    for i in range(half):
        i1 = i * 2
        i2 = i * 2 + 1
        x1 = q_full[offset + i1]
        x2 = q_full[offset + i2]
        q_roped[offset + i1] = x1 * cos_vals[i] - x2 * sin_vals[i]
        q_roped[offset + i2] = x2 * cos_vals[i] + x1 * sin_vals[i]

# Apply interleaved RoPE to k_pe
k_pe_roped = torch.zeros(qk_rope_head_dim)
for i in range(half):
    i1 = i * 2
    i2 = i * 2 + 1
    x1 = k_pe[i1]
    x2 = k_pe[i2]
    k_pe_roped[i1] = x1 * cos_vals[i] - x2 * sin_vals[i]
    k_pe_roped[i2] = x2 * cos_vals[i] + x1 * sin_vals[i]

print(f"q_roped[:5] = {q_roped[:5].tolist()}")
print(f"k_pe_roped[:5] = {k_pe_roped[:5].tolist()}")

# Build full_k and full_v
h_dim = qk_nope_head_dim + v_head_dim  # 448
full_k = torch.zeros(num_heads * head_dim)
full_v = torch.zeros(num_heads * v_head_dim)

for h in range(num_heads):
    kv_b_off = h * h_dim
    k_off = h * head_dim
    v_off = h * v_head_dim
    full_k[k_off:k_off + qk_nope_head_dim] = kv_b[kv_b_off:kv_b_off + qk_nope_head_dim]
    full_k[k_off + qk_nope_head_dim:k_off + head_dim] = k_pe_roped
    full_v[v_off:v_off + v_head_dim] = kv_b[kv_b_off + qk_nope_head_dim:kv_b_off + h_dim]

print(f"full_k[:5] = {full_k[:5].tolist()}")
print(f"full_v[:5] = {full_v[:5].tolist()}")

# Attention (single position, so softmax is trivial — just 1.0)
# attn_out = full_v (since there's only one position, probs=[1.0])
attn_out = full_v.clone()
print(f"attn_out[:5] = {attn_out[:5].tolist()}")

# O projection
o_proj_w = tensors["model.layers.0.self_attn.o_proj.weight"].float()
projected = attn_out @ o_proj_w.T
print(f"projected[:5] = {projected[:5].tolist()}")

# Residual
hidden_after_attn = hidden + projected
print(f"hidden_after_attn[:5] = {hidden_after_attn[:5].tolist()}")


# Post-attention layernorm
post_norm_w = tensors["model.layers.0.post_attention_layernorm.weight"].float()
variance = hidden_after_attn.pow(2).mean()
normed2 = hidden_after_attn * torch.rsqrt(variance + eps) * post_norm_w
print(f"normed2[:5] = {normed2[:5].tolist()}")

# Dense FFN (SwiGLU)
gate_w = tensors["model.layers.0.mlp.gate_proj.weight"].float()
up_w = tensors["model.layers.0.mlp.up_proj.weight"].float()
down_w = tensors["model.layers.0.mlp.down_proj.weight"].float()

gate = normed2 @ gate_w.T
up = normed2 @ up_w.T
silu_gate = gate * torch.sigmoid(gate)
ffn_act = silu_gate * up
ffn_out = ffn_act @ down_w.T

print(f"gate[:5] = {gate[:5].tolist()}")
print(f"up[:5] = {up[:5].tolist()}")
print(f"ffn_out[:5] = {ffn_out[:5].tolist()}")

hidden_after_ffn = hidden_after_attn + ffn_out
print(f"hidden_after_ffn[:5] = {hidden_after_ffn[:5].tolist()}")

print("gate_proj" in str(tensors.keys()))
print([k for k in tensors.keys() if "mlp" in k and "layers.0" in k])