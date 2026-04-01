import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name = "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False, trust_remote_code=True, device_map={"": 0})
model.model.eval()

device = torch.device("cuda:0")

prompt = "<｜begin▁of▁sentence｜>User: What is the capital of France?\n\nAssistant:"
input_ids = tokenizer.encode(prompt, add_special_tokens=False)
token_1 = input_ids[1]
print(f"Token 1 ID: {token_1}")

single_token = torch.tensor([[token_1]]).to(device)
position_ids = torch.tensor([[1]]).to(device)
attention_mask = torch.zeros(1, 1, 1, 1, dtype=torch.float16).to(device)

captured = {}

def make_hook(layer_idx):
    def hook(module, input, output):
        hidden = output[0]
        captured[layer_idx] = hidden[0, 0, :5].tolist()
    return hook

hooks = []
for i in range(len(model.model.model.layers)):
    h = model.model.model.layers[i].register_forward_hook(make_hook(i))
    hooks.append(h)

with torch.no_grad():
    embed = model.model.model.embed_tokens(single_token)
    print(f"embed[0:5]: {embed[0, 0, :5].tolist()}")

    hidden = embed.unsqueeze(0) if embed.dim() == 2 else embed
    for i, layer in enumerate(model.model.model.layers):
        hidden = layer(hidden, attention_mask=attention_mask, position_ids=position_ids)[0]

    normed = model.model.model.norm(hidden)
    logits = model.model.lm_head(normed)
    print(f"Output token: {logits[0, 0].argmax().item()}")

    for i in sorted(captured.keys()):
        print(f"L{i} hidden[0:5]: {captured[i]}")

for h in hooks:
    h.remove()