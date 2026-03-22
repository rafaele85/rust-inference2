from safetensors import safe_open
f = safe_open("/home/rambo/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/model.safetensors", framework="pt")
for name in sorted(f.keys()):
    t = f.get_tensor(name)
    print(name, t.shape, t.dtype)
