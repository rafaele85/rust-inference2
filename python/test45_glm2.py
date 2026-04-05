import json
import torch
import numpy as np
from pathlib import Path
from safetensors import safe_open

CKPT_PATH = Path.home() / ".cache/huggingface/hub/models--cyankiwi--GLM-4.7-Flash-AWQ-4bit/snapshots/fd1d0bbf48f24628352d78108a351a4c2287763a"

def load_all_tensors():
    """Load tensors from all shards"""
    st_files = list(CKPT_PATH.glob("*.safetensors"))
    all_tensors = {}

    print(f"Loading {len(st_files)} shard files...")
    for st_file in st_files:
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key not in all_tensors:
                    all_tensors[key] = f.get_tensor(key)
    return all_tensors

def generate_step2_kv_projection():
    """Generate reference for Step 2: KV-projection path"""
    tensors = load_all_tensors()

    kv_a_proj = tensors["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"]  # [576, 2048]
    kv_b_proj = tensors["model.layers.0.self_attn.kv_b_proj.weight"]          # [8960, 512]
    kv_ln = tensors["model.layers.0.self_attn.kv_a_layernorm.weight"]         # [512]

    torch.manual_seed(42)
    hidden = torch.randn(2048, dtype=torch.bfloat16)
    eps = 1e-5

    # 1. Compress: [2048] @ [2048, 576] -> [576]
    kv_latent = torch.matmul(hidden, kv_a_proj.t())

    # 2. Split
    c_cache = kv_latent[:512]  # compressed cache
    k_pe = kv_latent[512:]     # rope shared (64)

    # 3. RMSNorm
    variance = c_cache.pow(2).mean()
    c_cache_norm = c_cache * torch.rsqrt(variance + eps) * kv_ln

    # 4. Expand: [512] @ [512, 8960] -> [8960]
    kv_full = torch.matmul(c_cache_norm, kv_b_proj.t())

    # 5. Split: 3840 keys + 5120 values
    keys_nope = kv_full[:3840]
    values = kv_full[3840:]

    np.savez("step2_kv_proj_reference.npz",
        input=hidden.float().numpy(),
        kv_a_proj=kv_a_proj.float().numpy(),
        kv_b_proj=kv_b_proj.float().numpy(),
        kv_ln=kv_ln.float().numpy(),
        c_cache=c_cache.float().numpy(),
        k_pe=k_pe.float().numpy(),
        c_cache_norm=c_cache_norm.float().numpy(),
        keys_nope=keys_nope.float().numpy(),
        values=values.float().numpy(),
    )
    print("✅ Saved step2_kv_proj_reference.npz")

# Call it in main:
generate_step2_kv_projection()
