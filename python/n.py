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

tensors = {}
for f in sorted(Path(path).glob("*.safetensors")):
    with safe_open(str(f), framework="pt") as st:
        for k in st.keys():
            if "layers.0.linear_attn" in k or "layers.0.input_layernorm" in k:
                tensors[k] = st.get_tensor(k)

gdn = Qwen3_5MoeGatedDeltaNet(tc, layer_idx=0)
gdn_state = {}
for k, v in tensors.items():
    if "layers.0.linear_attn." in k:
        short_k = k.replace("model.language_model.layers.0.linear_attn.", "")
        gdn_state[short_k] = v
gdn.load_state_dict(gdn_state, strict=False)
gdn = gdn.float().cuda()

# Check norm weight
print(f"HF norm.weight first 5: {gdn.norm.weight[:5].tolist()}")
print(f"HF norm.weight mean: {gdn.norm.weight.mean().item():.4f}")
print(f"HF norm type: {type(gdn.norm)}")

# Check: run forward with hooks to see intermediates
hidden = torch.full((1, 1, 3072), 0.01).cuda()
input_norm_w = tensors["model.language_model.layers.0.input_layernorm.weight"].float().cuda()
norm = Qwen3_5MoeRMSNorm(3072, eps=tc.rms_norm_eps).cuda()
norm.weight = torch.nn.Parameter(input_norm_w)
normed = norm(hidden)

# Manually step through GatedDeltaNet forward
with torch.no_grad():
    batch_size, seq_len, _ = normed.shape
    mixed_qkv = gdn.in_proj_qkv(normed)
    mixed_qkv = mixed_qkv.transpose(1, 2)
    z = gdn.in_proj_z(normed)
    z = z.reshape(batch_size, seq_len, -1, gdn.head_v_dim)
    b = gdn.in_proj_b(normed)
    a = gdn.in_proj_a(normed)
    
    # Conv1d (torch fallback)
    import torch.nn.functional as F
    mixed_qkv = F.silu(gdn.conv1d(mixed_qkv)[:, :, :mixed_qkv.shape[-1]])
    mixed_qkv = mixed_qkv.transpose(1, 2)
    
    query, key, value = torch.split(mixed_qkv, [gdn.key_dim, gdn.key_dim, gdn.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, gdn.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, gdn.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, gdn.head_v_dim)
    
    beta = b.sigmoid()
    g = -gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)
    
    if gdn.num_v_heads // gdn.num_k_heads > 1:
        query = query.repeat_interleave(gdn.num_v_heads // gdn.num_k_heads, dim=2)
        key = key.repeat_interleave(gdn.num_v_heads // gdn.num_k_heads, dim=2)
    
    print(f"query shape after repeat: {query.shape}")
    print(f"query[0,0,0,:5]: {query[0,0,0,:5].tolist()}")
    print(f"beta shape: {beta.shape}, first 5: {beta[0,0,:5].tolist()}")
    print(f"g shape: {g.shape}, first 5: {g[0,0,:5].tolist()}")
    
    # Use recurrent version manually for single token
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_recurrent_gated_delta_rule
    core_attn_out, _ = torch_recurrent_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True
    )
    
    print(f"recurrent out shape: {core_attn_out.shape}")
    print(f"recurrent out[0,0,:5]: {core_attn_out[0,0,:5].tolist()}")
    
    # Apply norm
    core_attn_out_flat = core_attn_out.reshape(-1, gdn.head_v_dim)
    z_flat = z.reshape(-1, gdn.head_v_dim)
    normed_out = gdn.norm(core_attn_out_flat, z_flat)
    print(f"after norm first 5: {normed_out[0,:5].tolist()}")
    
    normed_out = normed_out.reshape(batch_size, seq_len, -1)
    output = gdn.out_proj(normed_out)
    print(f"output first 5: {output[0,0,:5].tolist()}")



from fla.modules.fused_norm_gate import FusedRMSNormGated
import torch

# Test with known values
norm = FusedRMSNormGated(4, eps=1e-6, activation="silu").cuda().float()
norm.weight = torch.nn.Parameter(torch.tensor([0.5, 0.5, 0.5, 0.5]).cuda())

x = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).cuda()
z = torch.tensor([[0.1, 0.2, 0.3, 0.4]]).cuda()
out = norm(x, z)
print(f"FLA output: {out.tolist()}")

# Compare formulas
var = (x * x).mean(dim=-1, keepdim=True)
scale = torch.rsqrt(var + 1e-6)
silu_z = z * torch.sigmoid(z)

# Standard: weight * x * scale * silu(z)
std = 0.5 * x * scale * silu_z
print(f"Standard (w*x): {std.tolist()}")

# Zero-centered: (1+weight) * x * scale * silu(z)
zc = 1.5 * x * scale * silu_z
print(f"Zero-centered ((1+w)*x): {zc.tolist()}")
