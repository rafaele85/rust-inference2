import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./model", torch_dtype=torch.float32)
model.eval()

input_ids = torch.tensor([[464]])
with torch.no_grad():
    wte = model.transformer.wte(input_ids)
    wpe = model.transformer.wpe(torch.tensor([[0]]))
    hidden = wte + wpe

for v in hidden[0][0][:10].tolist():
    print(f"{v:.6f}")