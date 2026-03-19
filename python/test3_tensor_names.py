from safetensors import safe_open

model = safe_open("./model/model.safetensors", framework="pt")
tensor = model.get_tensor("ln_f.bias")
for v in tensor[:10].tolist():
    print(f"{v:.6f}")
