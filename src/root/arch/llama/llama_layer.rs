use crate::root::arch::common::LoadLayer;
use crate::root::arch::llama::llama_load_weight::{llama_load_bias};
use crate::root::arch::safetensors::SafeTensor;
use crate::root::common_types::{Vecf32, WeightF16};
use crate::root::gpu::Gpu;
use crate::root::layer::{load_ffn_weights, load_norm_weights, load_q_weights, FFNWeights, NormWeights};
use crate::root::moe::StandardExpert;

pub struct LlamaLayer {
    pub input_norm_w: Vecf32,
    pub post_norm_w: Vecf32,
    pub q_proj_w: WeightF16,
    pub k_proj_w: WeightF16,
    pub v_proj_w: WeightF16,
    pub q_proj_b: Option<Vecf32>,
    pub k_proj_b: Option<Vecf32>,
    pub v_proj_b: Option<Vecf32>,

    pub o_proj_w: WeightF16,
    pub gate_proj_w: Option<WeightF16>,
    pub up_proj_w: Option<WeightF16>,
    pub down_proj_w: Option<WeightF16>,

    pub q_norm_w: Option<Vecf32>,
    pub k_norm_w: Option<Vecf32>,

    pub expert_router_w: Option<Vecf32>,
    pub experts: Option<Vec<StandardExpert>>
}


impl LoadLayer for LlamaLayer {
    fn load_layer(safe_tensor: &mut SafeTensor, layer: usize, gpu: &mut Gpu) -> Self {
        let norm_weights = load_norm_weights(safe_tensor, layer, gpu);
        let NormWeights {input_norm_w, post_norm_w } = norm_weights;
        let ffn_weights = load_ffn_weights(safe_tensor, layer, gpu);
        let FFNWeights {
            gate_proj_w, up_proj_w, down_proj_w, o_proj_w,
            experts, expert_router_w,
        } = ffn_weights;

        let q_proj_w = load_q_weights(safe_tensor, layer, gpu);
        let q_proj_b = llama_load_bias(safe_tensor, "self_attn.q_proj", layer);

        let k_proj_w = gpu.upload_f16_from_f32( &safe_tensor.load_layer_weight( "self_attn.k_proj", layer), &format!("k_proj_w{}", &layer) );
        let k_proj_b = llama_load_bias(safe_tensor, "self_attn.k_proj", layer);

        let v_proj_w = gpu.upload_f16_from_f32( &safe_tensor.load_layer_weight( "self_attn.v_proj", layer), &format!("v_proj_w{}", &layer) );
        let v_proj_b = llama_load_bias(safe_tensor, "self_attn.v_proj", layer);

        let q_norm_w = safe_tensor.load_weight_optional("self_attn.q_norm", layer);
        let k_norm_w = safe_tensor.load_weight_optional("self_attn.k_norm", layer);

        Self {
            input_norm_w, post_norm_w, q_proj_w, k_proj_w, v_proj_w, q_proj_b, k_proj_b, v_proj_b, o_proj_w,
            gate_proj_w, up_proj_w, down_proj_w, q_norm_w, k_norm_w,
            expert_router_w, experts
        }
    }
}
