from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("./model", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("./model")

input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
print(tokenizer.decode(output[0]))
