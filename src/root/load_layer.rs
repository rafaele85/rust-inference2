use crate::root::arch::llama::llama_moe::{llama_load_experts};
use crate::root::arch::safetensors::SafeTensor;
use crate::root::common_types::{Vecf32, WeightF16};
use crate::root::gpu::Gpu;
use crate::root::moe::StandardExpert;

pub struct CommonWeights {
    pub input_norm_w: Vecf32,
    pub post_norm_w: Vecf32,

    pub q_proj_w: WeightF16,          // hidden → num_heads * (nope + rope)
    pub o_proj_w: WeightF16,          // output projection

    pub gate_proj_w: Option<WeightF16>,
    pub up_proj_w: Option<WeightF16>,
    pub down_proj_w: Option<WeightF16>,

    pub expert_router_w : Option<Vecf32>,
    pub experts: Option<Vec<StandardExpert>>,
}
pub fn load_common_weights(safe_tensor: &mut SafeTensor, layer_no: usize, gpu: &mut Gpu) -> CommonWeights{
    let input_norm_w = safe_tensor.load_layer_weight( "input_layernorm", layer_no);
    let post_norm_w = safe_tensor.load_layer_weight( "post_attention_layernorm", layer_no);
    let q_proj_w = gpu.upload_f16_from_f32(&safe_tensor.load_layer_weight( "self_attn.q_proj", layer_no),  &format!("q_proj_w{}", layer_no));
    let o_proj_w = gpu.upload_f16_from_f32(&safe_tensor.load_layer_weight( "self_attn.o_proj", layer_no),  &format!("o_proj_w{}", layer_no));
    let (gate_proj_w, up_proj_w, down_proj_w) = safe_tensor.load_dense_ffns(layer_no, gpu);

    let expert_router_w = safe_tensor.load_weight_optional("mlp.gate", layer_no);
    let experts = llama_load_experts(safe_tensor, layer_no, &gpu);

    CommonWeights {input_norm_w, post_norm_w, q_proj_w, o_proj_w, gate_proj_w, up_proj_w, down_proj_w, experts, expert_router_w}
}