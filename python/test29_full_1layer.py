from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)
model.eval()

with torch.no_grad():
    embed = model.model.embed_tokens(torch.tensor([[1]]))
    position_ids = torch.tensor([[0]])
    cos, sin = model.model.rotary_emb(embed, position_ids)

    layer_out = model.model.layers[0](embed, position_embeddings=(cos, sin))
    print("layer0_out[:5]:", layer_out[0][0, :5])
