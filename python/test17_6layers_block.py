import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./model", torch_dtype=torch.float32)
model.eval()

input_ids = torch.tensor([[464, 3139, 286, 4881, 318]])
with torch.no_grad():
    wte = model.transformer.wte(input_ids)
    wpe = model.transformer.wpe(torch.arange(5).unsqueeze(0))
    hidden = wte + wpe
    for i in range(6):
        hidden = model.transformer.h[i](hidden)
    ln_out = model.transformer.ln_f(hidden)
    logits = ln_out @ model.lm_head.weight.T

top_id = torch.argmax(logits[0][-1]).item()
print(top_id)
