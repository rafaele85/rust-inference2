use crate::arch::common::LoadLayer;
use crate::arch::llama::llama_moe::{llama_load_experts, LlamaExpert};
use crate::arch::llama::llama_load_weight::{llama_load_bias, llama_load_weight, llama_load_weight_optional};
use crate::arch::safetensors::SafeTensor;
use crate::common_types::{Vecf32,  WeightF16};
use crate::gpu::Gpu;


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
    pub experts: Option<Vec<LlamaExpert>>

}

fn load_dense_ffns(safe_tensor: &mut SafeTensor, layer_no: usize, gpu: &Gpu) -> (Option<WeightF16>, Option<WeightF16>, Option<WeightF16>) {
    let has_dense_ffns = safe_tensor.entries.contains_key(
        &format!("model.layers.{}.mlp.gate_proj.weight", layer_no)
    );
    if has_dense_ffns  {
        let gate_proj_w = gpu.upload_f16_from_f32( &llama_load_weight(safe_tensor, "mlp.gate_proj", layer_no), "gate_proj_w" );
        let up_proj_w = gpu.upload_f16_from_f32( &llama_load_weight(safe_tensor, "mlp.up_proj", layer_no), "up_proj_w");
        let down_proj_w = gpu.upload_f16_from_f32( &llama_load_weight(safe_tensor, "mlp.down_proj", layer_no), "down_proj_w" );

        ( Some(gate_proj_w), Some(up_proj_w), Some(down_proj_w) )
    } else {
        (None, None, None)
    }
}


impl LoadLayer for LlamaLayer {
    fn load_layer(safe_tensor: &mut SafeTensor, layer: usize, gpu: &Gpu) -> Self {
        let input_norm_w = llama_load_weight(safe_tensor, "input_layernorm", layer);
        let post_norm_w = llama_load_weight(safe_tensor, "post_attention_layernorm", layer);

        let q_proj_w = gpu.upload_f16_from_f32( &llama_load_weight(safe_tensor, "self_attn.q_proj", layer), &format!("q_proj_w{}", &layer) );
        let q_proj_b = llama_load_bias(safe_tensor, "self_attn.q_proj", layer);

        let k_proj_w = gpu.upload_f16_from_f32( &llama_load_weight(safe_tensor, "self_attn.k_proj", layer), &format!("k_proj_w{}", &layer) );
        let k_proj_b = llama_load_bias(safe_tensor, "self_attn.k_proj", layer);

        let v_proj_w = gpu.upload_f16_from_f32( &llama_load_weight(safe_tensor, "self_attn.v_proj", layer), &format!("v_proj_w{}", &layer) );
        let v_proj_b = llama_load_bias(safe_tensor, "self_attn.v_proj", layer);

        let o_proj_w = gpu.upload_f16_from_f32( &llama_load_weight(safe_tensor, "self_attn.o_proj", layer), "o_proj_w" );

        let (gate_proj_w, up_proj_w, down_proj_w) = load_dense_ffns(safe_tensor, layer, gpu);

        let q_norm_w = llama_load_weight_optional(safe_tensor, "self_attn.q_norm", layer);
        let k_norm_w = llama_load_weight_optional(safe_tensor, "self_attn.k_norm", layer);

        let expert_router_w = llama_load_weight_optional(safe_tensor, "mlp.gate", layer);

        let experts = llama_load_experts(safe_tensor, layer, &gpu );


        Self {
            input_norm_w, post_norm_w, q_proj_w, k_proj_w, v_proj_w, q_proj_b, k_proj_b, v_proj_b, o_proj_w,
            gate_proj_w, up_proj_w, down_proj_w, q_norm_w, k_norm_w,
            expert_router_w, experts
        }
    }
}
