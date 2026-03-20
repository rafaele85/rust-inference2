import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./model", torch_dtype=torch.float32)
model.eval()

input_ids = torch.tensor([[464, 3139, 286, 4881, 318]])
with torch.no_grad():
    wte = model.transformer.wte(input_ids)
    wpe = model.transformer.wpe(torch.arange(5).unsqueeze(0))
    hidden = wte + wpe
    hidden = model.transformer.h[0](hidden)

for v in hidden[0][-1][:10].tolist():
    print(f"{v:.4f}")