from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
import inspect
from awq.modules.linear.gemm import WQLinearMMFunction


model_name = "Qwen/Qwen2.5-0.5B-Instruct-AWQ"
model = AutoAWQForCausalLM.from_quantized(model_name, fuse_layers=False, device_map={"": "cuda:0"})


from awq.utils.packing_utils import dequantize_gemm
import torch

layer = model.model.model.layers[0].self_attn.q_proj
w = dequantize_gemm(layer.qweight, layer.qzeros, layer.scales, 4, 128)
print(f"dequantize_gemm shape: {w.shape}")
print(f"[0, :10]: {w[0, :10].tolist()}")
print(f"[:10, 0]: {w[:10, 0].tolist()}")