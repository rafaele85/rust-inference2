from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)
model.eval()

with torch.no_grad():
    embed = model.model.embed_tokens(torch.tensor([[1]]))
    normed = model.model.layers[0].input_layernorm(embed)

    attn = model.model.layers[0].self_attn
    q = attn.q_proj(normed).view(1, 1, 32, 64).transpose(1, 2)
    k = attn.k_proj(normed).view(1, 1, 4, 64).transpose(1, 2)

    position_ids = torch.tensor([[1]])
    cos, sin = model.model.rotary_emb(k, position_ids)

    print("cos[:5]:", cos[0, 0, :5])
    print("sin[:5]:", sin[0, 0, :5])

    q_rope, k_rope = attn.rotary_fn(q, k, cos, sin)

    print("q_rope head0[:5]:", q_rope[0, 0, 0, :5])
    print("k_rope head0[:5]:", k_rope[0, 0, 0, :5])