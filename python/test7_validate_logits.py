import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./model", torch_dtype=torch.float32)
model.eval()

input_ids = torch.tensor([[464]])
with torch.no_grad():
    output = model(input_ids)
    logits = output.logits

top_values, top_indices = torch.topk(logits[0][0], 5)
for v, i in zip(top_values.tolist(), top_indices.tolist()):
    print(f"{i} {v:.4f}")
