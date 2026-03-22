from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
for name, param in model.named_parameters():
    print(name, param.shape, param.dtype)
