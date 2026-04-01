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

# Hook to capture intermediates at layer 1, token 0
captured = {}

def hook_layer1_input(module, input, output):
    # output is the hidden state after this layer
    captured['after_layer1'] = output[0][0, 0, :10].detach().cpu().float().tolist()

def hook_layer1_attn_output(module, input, output):
    captured['layer1_attn_out'] = output[0][0, 0, :10].detach().cpu().float().tolist()

def hook_layer1_mlp_output(module, input, output):
    captured['layer1_mlp_out'] = output[0, 0, :10].detach().cpu().float().tolist()

def hook_layer1_mlp_input(module, input):
    captured['layer1_mlp_input'] = input[0][0, 0, :10].detach().cpu().float().tolist()

layer1 = model.model.layers[1]
layer1.register_forward_hook(hook_layer1_input)
layer1.self_attn.register_forward_hook(hook_layer1_attn_output)
layer1.mlp.register_forward_hook(hook_layer1_mlp_output)
layer1.mlp.register_forward_pre_hook(hook_layer1_mlp_input)

with torch.no_grad():
    out = model(input_tensor, output_hidden_states=True)

    print(f"embed[0:10]: {out.hidden_states[0][0, 0, :10].float().tolist()}")
    print(f"after_layer0[0:10]: {out.hidden_states[1][0, 0, :10].float().tolist()}")
    print(f"after_layer1[0:10]: {out.hidden_states[2][0, 0, :10].float().tolist()}")
    print(f"layer1_attn_out[0:10]: {captured.get('layer1_attn_out', 'N/A')}")
    print(f"layer1_mlp_input[0:10]: {captured.get('layer1_mlp_input', 'N/A')}")
    print(f"layer1_mlp_out[0:10]: {captured.get('layer1_mlp_out', 'N/A')}")

    logits = out.logits[0, -1, :]
    top5 = torch.topk(logits, 5)
    print(f"\ntop5 logits:")
    for i in range(5):
        tid = top5.indices[i].item()
        print(f"  {tid}: {top5.values[i].item():.4f} = {tok.decode([tid])}")



from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
messages = [{"role": "user", "content": "What is the capital of France?"}]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(repr(prompt))

with torch.no_grad():
    gen_ids = model.generate(input_tensor, max_new_tokens=20, do_sample=False, use_cache=False)
    print(tok.decode(gen_ids[0]))
    h26 = out.hidden_states[26][0, 0, :10].float().tolist()
    print(f"after_layer25[0:10]: {h26}")