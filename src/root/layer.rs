use std::borrow::Cow;
use crate::root::arch::safetensors::SafeTensor;
use crate::root::common_types::{Arrf32, Vecf32, WeightF16};
use crate::root::config::{MLAConfig, NormConfig};
use crate::root::gpu::Gpu;
use crate::root::moe::{load_experts_awq, load_experts_bf16, StandardExpert};

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
    let bf16_key = format!("model.layers.{}.mlp.experts.0.gate_proj.weight", layer_no);
    let experts = if safe_tensor.entries.contains_key(&bf16_key) {
        load_experts_bf16(safe_tensor, layer_no, gpu)
    } else {
        load_experts_awq(safe_tensor, layer_no, gpu)
    };

    CommonWeights {input_norm_w, post_norm_w, q_proj_w, o_proj_w, gate_proj_w, up_proj_w, down_proj_w, experts, expert_router_w}
}
pub trait MLAKVLayer {
    fn get_kv_a_proj_w(&self) -> &WeightF16;
    fn get_kv_b_proj_w(&self) -> &WeightF16;
    fn get_kv_a_layernorm_w(&self) -> &Vecf32;
    fn get_o_proj_w(&self) -> &WeightF16;
    fn compute_q_raw_normed<'a, TC: NormConfig + MLAConfig>(&'a self, normed: &'a Vecf32, config: &TC, gpu: &mut Gpu) -> Cow<'a, Arrf32>;

    fn get_q_raw_proj_w(&self) -> &WeightF16;
}

pub trait FFNLayer {
    fn get_input_norm_w(&self) -> &Vecf32;
    fn get_post_norm_w(&self) -> &Vecf32;
    fn get_gate_proj_w(&self) -> &Option<WeightF16>;
    fn get_up_proj_w(&self) -> &Option<WeightF16>;
    fn get_down_proj_w(&self) -> &Option<WeightF16>;
}

pub trait MOELayer {
    fn get_expert_router_w(&self) -> &Option<Vecf32>;
    fn get_experts(&self) -> &Option<Vec<StandardExpert>>;
}
pub trait SharedExpertLayer {
    fn get_shared_gate_proj_w(&self) -> &Option<WeightF16>;
    fn get_shared_up_proj_w(&self) -> &Option<WeightF16>;
    fn get_shared_down_proj_w(&self) -> &Option<WeightF16>;
}
