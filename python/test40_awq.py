import torch, json, os

def resolve_hf_path(model_name):
    home = os.environ["HOME"]
    sanitized = model_name.replace("/", "--")
    model_dir = f"{home}/.cache/huggingface/hub/models--{sanitized}"
    with open(f"{model_dir}/refs/main") as f:
        h = f.read().strip()
    return f"{model_dir}/snapshots/{h}"

path = resolve_hf_path("TechxGenus/DeepSeek-V2-Lite-Chat-AWQ")

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()

prompt = "<｜begin▁of▁sentence｜>User: What is the capital of France?\n\nAssistant:"
ids = tok.encode(prompt, add_special_tokens=False)
input_tensor = torch.tensor([ids]).to(model.device)

with torch.no_grad():
    out = model(input_tensor, output_hidden_states=True)
    
    # Embedding output
    h0 = out.hidden_states[0][0, 0, :5]
    print(f"embed[0:5]: {h0.tolist()}")
    
    # After layer 0
    h1 = out.hidden_states[1][0, 0, :5]
    print(f"after_layer0[0:5]: {h1.tolist()}")
    
    # Top 5 logits
    logits = out.logits[0, -1, :]
    top5 = torch.topk(logits, 5)
    for i in range(5):
        print(f"top {top5.indices[i].item()}: {top5.values[i].item():.4f} = {tok.decode([top5.indices[i].item()])}")
