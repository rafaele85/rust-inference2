import torch
from awq import AutoAWQForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False, device_map={"": "cuda:0"})

m = model.model.model

input_ids = torch.tensor([[785]], device="cuda:0")
embed = m.embed_tokens(input_ids).half()

# Get position embeddings from rotary_emb
pos_ids = torch.tensor([[0]], device="cuda:0")
position_embeddings = m.rotary_emb(embed, pos_ids)

with torch.no_grad():
    out = m.layers[0](embed, position_ids=pos_ids, position_embeddings=position_embeddings)
    h = out[0]
    print(f"after layer 0: {h[0, :5].float().tolist()}")


# Run attention only
layer0 = m.layers[0]
normed = layer0.input_layernorm(embed)
attn_out, _, _ = layer0.self_attn(normed, position_ids=pos_ids, position_embeddings=position_embeddings)
h_after_attn = embed + attn_out
print(f"after attn: {h_after_attn[0, :5].float().tolist()}")