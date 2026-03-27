from safetensors import safe_open
import os, numpy as np

model_name = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
home = os.environ["HOME"]
sanitized = model_name.replace("/", "--")
model_dir = f"{home}/.cache/huggingface/hub/models--{sanitized}"
ref = open(f"{model_dir}/refs/main").read().strip()
path = f"{model_dir}/snapshots/{ref}/model.safetensors"

f = safe_open(path, framework="numpy")

qw = f.get_tensor("model.layers.0.self_attn.q_proj.qweight")
qz = f.get_tensor("model.layers.0.self_attn.q_proj.qzeros")
sc = f.get_tensor("model.layers.0.self_attn.q_proj.scales")

val = int(qw[0][0])
uval = val & 0xFFFFFFFF
print(f"qweight[0][0] = {val} (0x{uval:08x})")

AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
nibbles = []
for k in range(8):
    shift = AWQ_ORDER[k] * 4
    n = (uval >> shift) & 0xF
    nibbles.append(n)
print(f"Unpacked nibbles (AWQ order): {nibbles}")

zval = int(qz[0][0])
uval_z = zval & 0xFFFFFFFF
print(f"\nqzeros[0][0] = {zval} (0x{uval_z:08x})")
z_nibbles = []
for k in range(8):
    shift = AWQ_ORDER[k] * 4
    n = (uval_z >> shift) & 0xF
    z_nibbles.append(n)
print(f"Unpacked zero nibbles: {z_nibbles}")

print(f"\nscales[0][0:8] = {sc[0, 0:8].tolist()}")

print(f"\nDequantized [in=0, out=0..7]:")
for k in range(8):
    d = (float(nibbles[k]) - float(z_nibbles[k])) * float(sc[0, k])
    print(f"  out={k}: ({nibbles[k]} - {z_nibbles[k]}) * {sc[0,k]} = {d}")


print(f"qweight shape: {qw.shape}")
print(f"scales shape: {sc.shape}")
print(f"qweight[0][0:5] = {qw[0, 0:5].tolist()}")
print(f"qweight shape[0] is in_dim or out_dim?")
print(f"hidden_size=896, out_features of q_proj=896")
print(f"So qweight[896, 112]: first dim={qw.shape[0]}, 112*8={112*8}")

qw_k = f.get_tensor("model.layers.0.self_attn.k_proj.qweight")
sc_k = f.get_tensor("model.layers.0.self_attn.k_proj.scales")
print(f"k_proj qweight shape: {qw_k.shape}")
print(f"k_proj scales shape: {sc_k.shape}")
print(f"k_proj: in=896 (hidden_size), out=128 (kv_heads * head_dim = 2*64)")

print("\nSequential nibbles (no AWQ_ORDER):")
for k in range(8):
    shift = k * 4
    n = (uval >> shift) & 0xF
    print(f"  k={k}: nibble={n}")


