from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeRMSNormGated,
    Qwen3_5MoeGatedDeltaNet,
)
from transformers import AutoConfig
from safetensors import safe_open
from pathlib import Path
import torch
import torch.nn.functional as F

path = "/mnt/data/models/Qwen3.5-122B-A10B-AWQ-4bit"
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
tc = config.text_config

tensors = {}
for f in sorted(Path(path).glob("*.safetensors")):
    with safe_open(str(f), framework="pt") as st:
        for k in st.keys():
            if "layers.0.linear_attn" in k or "layers.0.input_layernorm" in k:
                tensors[k] = st.get_tensor(k)

def get(name):
    return tensors[f"model.language_model.layers.0.{name}"].float()

hidden = torch.full((3072,), 0.01)
input_norm_w = get("input_layernorm.weight")
var = (hidden * hidden).mean()
normed = (1.0 + input_norm_w) * hidden * torch.rsqrt(var + tc.rms_norm_eps)

qkv_w = get("linear_attn.in_proj_qkv.weight")
z_w = get("linear_attn.in_proj_z.weight")
a_w = get("linear_attn.in_proj_a.weight")
b_w = get("linear_attn.in_proj_b.weight")
out_w = get("linear_attn.out_proj.weight")
norm_w = get("linear_attn.norm.weight")
conv1d_w = get("linear_attn.conv1d.weight").squeeze(1)
a_log = get("linear_attn.A_log")
dt_bias = get("linear_attn.dt_bias")

nkh = tc.linear_num_key_heads
khd = tc.linear_key_head_dim
nvh = tc.linear_num_value_heads
vhd = tc.linear_value_head_dim
q_dim = nkh * khd
v_dim = nvh * vhd
qkv_dim = q_dim * 2 + v_dim

qkv_proj = normed @ qkv_w.T
conv_out = qkv_proj * conv1d_w[:, -1]
qkv_act = conv_out * torch.sigmoid(conv_out)

q_raw = qkv_act[:q_dim]
k_raw = qkv_act[q_dim:q_dim*2]
v = qkv_act[q_dim*2:]

q = q_raw.view(nkh, khd)
q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
q = q.view(-1)
k = k_raw.view(nkh, khd)
k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
k = k.view(-1)

alpha_raw = normed @ a_w.T
beta = torch.sigmoid(normed @ b_w.T)
dt = torch.where(alpha_raw + dt_bias > 20, alpha_raw + dt_bias, torch.log1p(torch.exp(alpha_raw + dt_bias)))
decay = torch.exp(-torch.exp(a_log) * dt)

hpk = nvh // nkh
state = torch.zeros(nvh, khd, vhd)
output = torch.zeros(v_dim)

for h in range(nvh):
    kh = h // hpk
    kv = k[kh*khd:(kh+1)*khd]
    vv = v[h*vhd:(h+1)*vhd]
    qv = q[kh*khd:(kh+1)*khd]
    # Decay first (doesn't matter for state=0)
    state[h] = decay[h] * state[h]
    sk = state[h].T @ kv
    dv = vv - sk
    state[h] = state[h] + beta[h] * torch.outer(kv, dv)
    output[h*vhd:(h+1)*vhd] = state[h].T @ qv

print(f"recurrent_out head0 first 5: {output[:5].tolist()}")

z_raw = normed @ z_w.T
z_silu = z_raw * torch.sigmoid(z_raw)
print(f"z_raw head0 first 5: {z_raw[:5].tolist()}")
print(f"z (silu) head0 first 5: {z_silu[:5].tolist()}")
print(f"norm_w first 5: {norm_w[:5].tolist()}")

# Standard RMSNorm (Qwen3_5MoeRMSNormGated uses weight * x * scale)
gated = torch.zeros(v_dim)
for h in range(nvh):
    o = h * vhd
    head_slice = output[o:o+vhd]
    var_h = (head_slice * head_slice).mean()
    normed_h = norm_w * head_slice * torch.rsqrt(var_h + tc.rms_norm_eps)
    if h == 0:
        print(f"normed_head0 first 5: {normed_h[:5].tolist()}")
    gated[o:o+vhd] = normed_h * z_silu[o:o+vhd]

print(f"gated head0 first 5: {gated[:5].tolist()}")

projected = gated @ out_w.T
print(f"projected first 5: {projected[:5].tolist()}")
