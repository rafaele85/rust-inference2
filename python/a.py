import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./model", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("./model")
model.eval()

input_ids = torch.tensor([[464, 3139, 286, 4881, 318]])
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=15, do_sample=False)

print(tokenizer.decode(output[0]))
