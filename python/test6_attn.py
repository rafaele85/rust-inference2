import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./model", torch_dtype=torch.float32)
model.eval()

input_ids = torch.tensor([[464]])
with torch.no_grad():
    wte = model.transformer.wte(input_ids)
    wpe = model.transformer.wpe(torch.tensor([[0]]))
    hidden = wte + wpe
    attn_out = model.transformer.h[0].attn(model.transformer.h[0].ln_1(hidden))[0]

for v in attn_out[0][0][:10].tolist():
    print(f"{v:.4f}")