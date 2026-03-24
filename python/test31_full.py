from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

input_ids = [1]  # BOS
with torch.no_grad():
    for i in range(10):
        outputs = model(torch.tensor([input_ids]))
        next_id = outputs.logits[0, -1].argmax().item()
        input_ids.append(next_id)
        print(f"pos={i} id={next_id} token={tokenizer.decode(next_id)}")

print(tokenizer.decode(input_ids))
