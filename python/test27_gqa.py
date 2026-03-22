from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)
model.eval()

with torch.no_grad():
    input_ids = torch.tensor([[1]])
    embed = model.model.embed_tokens(input_ids)
    normed = model.model.layers[0].input_layernorm(embed)

    attn = model.model.layers[0].self_attn
    q = attn.q_proj(normed).view(1, 1, 32, 64).transpose(1, 2)
    k = attn.k_proj(normed).view(1, 1, 4, 64).transpose(1, 2)
    v = attn.v_proj(normed).view(1, 1, 4, 64).transpose(1, 2)

    k_expanded = k.repeat_interleave(8, dim=1)
    v_expanded = v.repeat_interleave(8, dim=1)

    attn_raw = v_expanded.transpose(1, 2).reshape(1, 1, 2048)

    o = attn.o_proj(attn_raw)
    print("manual attn[:5]:", o[0, 0, :5])

    # Compare with SDK
    position_ids = torch.tensor([[0]])
    cos, sin = model.model.rotary_emb(normed, position_ids)
    attn_output = model.model.layers[0].self_attn(normed, position_embeddings=(cos, sin))
    print("sdk attn[:5]:", attn_output[0][0, 0, :5])
