# test_deepseek_reference.py
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

# Monkey-patch for transformers >= 4.49 compatibility with old custom model code
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: None
if not hasattr(DynamicCache, "seen_tokens"):
    @property
    def _seen_tokens_compat(self):
        return self.get_seq_length()
    DynamicCache.seen_tokens = _seen_tokens_compat

model_name = "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
)

prompt = "User: Explain quantum entanglement in simple terms.\n\nAssistant:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_ids = inputs.input_ids
print(f"Input IDs ({input_ids.shape[1]}): {input_ids[0].tolist()}")

eos = model.config.eos_token_id
pad_id = eos[0] if isinstance(eos, list) else eos

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=pad_id,
    )

input_len = input_ids.shape[1]
generated_ids = output[0][input_len:].tolist()
text = tokenizer.decode(output[0], skip_special_tokens=False)

print(f"\n=== Python reference output ===")
print(text)
print(f"\n=== Generated token IDs ({len(generated_ids)}) ===")
print(generated_ids)

with open("reference_output.json", "w") as f:
    json.dump({
        "input_ids": input_ids[0].tolist(),
        "generated_ids": generated_ids,
        "text": text,
    }, f, indent=2)

print("\nSaved to reference_output.json")