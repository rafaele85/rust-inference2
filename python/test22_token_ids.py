import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./model", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("./model")
model.eval()

input_ids = tokenizer.encode("The capital of France is")
input_tensor = torch.tensor([input_ids])
with torch.no_grad():
    output = model.generate(input_tensor, max_new_tokens=5, do_sample=False)

print(tokenizer.decode(output[0]))
