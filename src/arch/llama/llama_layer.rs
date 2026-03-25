use crate::arch::common::LoadLayer;
use crate::arch::llama::llama::load_tensor_bf16;
use crate::arch::safetensors::SafeTensor;
use crate::common_types::{Vecf32, Weight};
use crate::gpu::Gpu;


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
    pub o_proj_w: Weight,
    pub gate_proj_w: Weight,
    pub up_proj_w: Weight,
    pub down_proj_w: Weight,
}

impl LoadLayer for LlamaLayer {
    fn load_layer(safe_tensor: &mut SafeTensor, layer: usize, gpu: &Gpu) -> Self {
        let input_norm_w = load_weight(safe_tensor, "input_layernorm", layer);
        let post_norm_w = load_weight(safe_tensor, "post_attention_layernorm", layer);
        let q_proj_w = gpu.upload( &load_weight(safe_tensor, "self_attn.q_proj", layer) );
        let k_proj_w = gpu.upload( &load_weight(safe_tensor, "self_attn.k_proj", layer) );
        let v_proj_w = gpu.upload( &load_weight(safe_tensor, "self_attn.v_proj", layer) );
        let o_proj_w = gpu.upload( &load_weight(safe_tensor, "self_attn.o_proj", layer) );
        let gate_proj_w = gpu.upload( &load_weight(safe_tensor, "mlp.gate_proj", layer) );
        let up_proj_w = gpu.upload( &load_weight(safe_tensor, "mlp.up_proj", layer) );
        let down_proj_w = gpu.upload( &load_weight(safe_tensor, "mlp.down_proj", layer) );


        Self {input_norm_w, post_norm_w, q_proj_w, k_proj_w, v_proj_w, o_proj_w, gate_proj_w, up_proj_w, down_proj_w}
    }
}
