use crate::arch::common::LoadLayer;
use crate::arch::llama::llama::load_tensor_bf16;
use crate::arch::safetensors::SafeTensor;
use crate::common_types::{Vecf32, Weight};
use crate::gpu::Gpu;


fn load_bias(safe_tensor : &mut SafeTensor, name: &str, layer: usize) -> Option<Vecf32> {
    let entries = &safe_tensor.entries;
    let key = format!("model.layers.{}.{}.bias", layer, name);
    if entries.contains_key(&key) {
        let b = load_tensor_bf16(safe_tensor, &key);
        return Some(b);
    }
    None
}

fn load_weight_optional(safe_tensor : &mut SafeTensor, name: &str, layer: usize) -> Option<Vecf32> {
    let entries = &safe_tensor.entries;
    let key = format!("model.layers.{}.{}.weight", layer, name);
    if entries.contains_key(&key) {
        let b = load_tensor_bf16(safe_tensor, &key);
        return Some(b);
    }
    None
}
fn load_weight(safe_tensor: &mut SafeTensor, name: &str, layer: usize) -> Vecf32 {
    let prefix = format!("model.layers.{}", layer);
    let w = load_tensor_bf16(safe_tensor, &format!("{}.{}.weight", prefix, name));
    w
}
pub struct LlamaLayer {
    pub input_norm_w: Vecf32,
    pub post_norm_w: Vecf32,
    pub q_proj_w: Weight,
    pub k_proj_w: Weight,
    pub v_proj_w: Weight,
    pub q_proj_b: Option<Vecf32>,
    pub k_proj_b: Option<Vecf32>,
    pub v_proj_b: Option<Vecf32>,

    pub o_proj_w: Weight,
    pub gate_proj_w: Weight,
    pub up_proj_w: Weight,
    pub down_proj_w: Weight,

    pub q_norm_w: Option<Vecf32>,
    pub k_norm_w: Option<Vecf32>,
}

impl LoadLayer for LlamaLayer {
    fn load_layer(safe_tensor: &mut SafeTensor, layer: usize, gpu: &Gpu) -> Self {
        let input_norm_w = load_weight(safe_tensor, "input_layernorm", layer);
        let post_norm_w = load_weight(safe_tensor, "post_attention_layernorm", layer);

        let q_proj_w = gpu.upload( &load_weight(safe_tensor, "self_attn.q_proj", layer), &format!("q_proj_w{}", &layer) );
        let q_proj_b = load_bias(safe_tensor, "self_attn.q_proj", layer);

        let k_proj_w = gpu.upload( &load_weight(safe_tensor, "self_attn.k_proj", layer), &format!("k_proj_w{}", &layer) );
        let k_proj_b = load_bias(safe_tensor, "self_attn.k_proj", layer);

        let v_proj_w = gpu.upload( &load_weight(safe_tensor, "self_attn.v_proj", layer), &format!("v_proj_w{}", &layer) );
        let v_proj_b = load_bias(safe_tensor, "self_attn.v_proj", layer);

        let o_proj_w = gpu.upload( &load_weight(safe_tensor, "self_attn.o_proj", layer), "o_proj_w" );
        let gate_proj_w = gpu.upload( &load_weight(safe_tensor, "mlp.gate_proj", layer), "gate_proj_w" );
        let up_proj_w = gpu.upload( &load_weight(safe_tensor, "mlp.up_proj", layer), "up_proj_w");
        let down_proj_w = gpu.upload( &load_weight(safe_tensor, "mlp.down_proj", layer), "down_proj_w" );

        let q_norm_w = load_weight_optional(safe_tensor, "self_attn.q_norm", layer);
        let k_norm_w = load_weight_optional(safe_tensor, "self_attn.k_norm", layer);


        Self {input_norm_w, post_norm_w, q_proj_w, k_proj_w, v_proj_w, q_proj_b, k_proj_b, v_proj_b, o_proj_w, gate_proj_w, up_proj_w, down_proj_w, q_norm_w, k_norm_w}
    }
}
