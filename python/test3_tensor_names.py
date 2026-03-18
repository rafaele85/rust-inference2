from safetensors import safe_open

model = safe_open("./model/model.safetensors", framework="pt")
for name in sorted(model.keys()):
    shape = list(model.get_tensor(name).shape)
    print(f"{name} {shape}")
