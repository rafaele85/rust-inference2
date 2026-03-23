from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

with torch.no_grad():
    input_ids = torch.tensor([[1]])
    outputs = model(input_ids)
    logits = outputs.logits
    top_id = logits[0, -1].argmax().item()
    print("top token id:", top_id)
    print("top token:", tokenizer.decode(top_id))
    print("logits[:5]:", logits[0, 0, :5])
