from safetensors import safe_open

model = safe_open("./model/model.safetensors", framework="pt")
for name in sorted(model.keys()):
    print(name)
