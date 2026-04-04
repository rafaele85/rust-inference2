import torch
from pathlib import Path
from safetensors.torch import load_file
from compressed_tensors.compressors.quantized_compressors.pack_quantized import unpack_from_int32

import os
home = os.environ["HOME"]
base = f"{home}/.cache/huggingface/hub/models--cyankiwi--GLM-4.7-Flash-AWQ-4bit/snapshots/fd1d0bbf48f24628352d78108a351a4c2287763a"

for shard in sorted(Path(base).glob("model*.safetensors")):
    t = load_file(str(shard))
    if "model.layers.1.self_attn.q_a_proj.weight_packed" in t:
        packed = t["model.layers.1.self_attn.q_a_proj.weight_packed"]
        scale = t["model.layers.1.self_attn.q_a_proj.weight_scale"].float()
        unpacked = unpack_from_int32(packed, num_bits=4, shape=torch.Size([768, 2048]))
        scale_expanded = scale.repeat_interleave(32, dim=1)
        w = unpacked.float() * scale_expanded

        print(f"unpacked[0, :8]: {unpacked[0, :8].tolist()}")
        print(f"packed[0, 0] hex: {packed[0,0].item() & 0xFFFFFFFF:#010x}")
        print(f"scale[0, 0]: {scale[0, 0].item()}")
        print(f"w[0, :10]: {w[0, :10].tolist()}")
        print(f"w[0, -10:]: {w[0, -10:].tolist()}")
        print(f"w[767, :10]: {w[767, :10].tolist()}")
        break