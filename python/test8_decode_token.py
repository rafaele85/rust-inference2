import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./model", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("./model")
model.eval()

input_ids = torch.tensor([[464]])
with torch.no_grad():
    output = model(input_ids)
    logits = output.logits

top_id = torch.argmax(logits[0][0]).item()
print(f"{top_id}")
