use std::borrow::Cow;
use crate::root::arch::safetensors::{LoadLayerWeightArgs, SafeTensor};
use crate::root::common_types::{Arrf32, Vecf32, WeightF16};
use crate::root::config::{MLAConfig, NormConfig};
use crate::root::gpu::Gpu;
use crate::root::moe::{load_experts_awq, load_experts_bf16, load_experts_ct, StandardExpert};

pub struct NormWeights {
    pub input_norm_w: Vecf32,
    pub post_norm_w: Vecf32,
}

pub struct FFNWeights {
    pub gate_proj_w: Option<WeightF16>,
    pub up_proj_w: Option<WeightF16>,
    pub down_proj_w: Option<WeightF16>,
    pub o_proj_w: WeightF16,

    pub expert_router_w : Option<Vecf32>,
    pub experts: Option<Vec<StandardExpert>>,
}
pub struct CommonWeights {
    pub q_proj_w: WeightF16,          // hidden → num_heads * (nope + rope)
    pub o_proj_w: WeightF16,          // output projection
}

pub fn load_norm_weights(safe_tensor: &mut SafeTensor, layer_no: usize, _gpu: &mut Gpu) -> NormWeights {
    let input_norm_w = safe_tensor.load_layer_weight(LoadLayerWeightArgs { name: "input_layernorm", layer_no});
    let post_norm_w = safe_tensor.load_layer_weight(LoadLayerWeightArgs { name: "post_attention_layernorm", layer_no});
    NormWeights { input_norm_w, post_norm_w }
}

pub fn load_ffn_weights(safe_tensor: &mut SafeTensor, layer_no: usize, gpu: &mut Gpu) -> FFNWeights {
    let (gate_proj_w, up_proj_w, down_proj_w) = safe_tensor.load_dense_ffns(layer_no, gpu);
    let o_proj_w = gpu.upload_f16_from_f32(
        &safe_tensor.load_layer_weight(LoadLayerWeightArgs { name: "self_attn.o_proj", layer_no}),
        &format!("o_proj_w{}", layer_no)
    );

    let expert_router_w = safe_tensor.load_weight_optional("mlp.gate", layer_no);

    let bf16_key = format!("model.layers.{}.mlp.experts.0.gate_proj.weight", layer_no);
    let awq_key = format!("model.layers.{}.mlp.experts.0.gate_proj.qweight", layer_no);
    let ct_key = format!("model.layers.{}.mlp.experts.0.gate_proj.weight_packed", layer_no);

    let experts = if safe_tensor.entries.contains_key(&bf16_key) {
        load_experts_bf16(safe_tensor, layer_no, gpu)
    } else if safe_tensor.entries.contains_key(&awq_key) {
        load_experts_awq(safe_tensor, layer_no, gpu)
    } else if safe_tensor.entries.contains_key(&ct_key) {
        load_experts_ct(safe_tensor, layer_no, gpu)
    } else {
        None
    };

    FFNWeights {gate_proj_w, up_proj_w, down_proj_w, o_proj_w, experts, expert_router_w }
}

pub fn load_q_weights(safe_tensor: &mut SafeTensor, layer_no: usize,  gpu: &mut Gpu) -> WeightF16 {
    let q_proj_w = gpu.upload_f16_from_f32(
        &safe_tensor.load_layer_weight(LoadLayerWeightArgs { name: "self_attn.q_proj", layer_no}),
        &format!("q_proj_w{}", layer_no)
    );
    q_proj_w
}


pub fn load_shared_experts(safe_tensor: &mut SafeTensor, layer_no: usize, gpu: &mut Gpu) -> (Option<WeightF16>, Option<WeightF16>, Option<WeightF16>) {
    let key = format!("model.layers.{}.mlp.shared_experts.gate_proj.qweight", layer_no);
    let key_bf16 = format!("model.layers.{}.mlp.shared_experts.gate_proj.weight", layer_no);

    if safe_tensor.entries.contains_key(&key) || safe_tensor.entries.contains_key(&key_bf16) {
        let gate = gpu.upload_f16_from_f32(
            &safe_tensor.load_layer_weight(LoadLayerWeightArgs { name: "mlp.shared_experts.gate_proj", layer_no}),
            &format!("shared_gate{}", layer_no)
        );
        let up = gpu.upload_f16_from_f32(
            &safe_tensor.load_layer_weight(LoadLayerWeightArgs { name: "mlp.shared_experts.up_proj", layer_no}),
            &format!("shared_up{}", layer_no)
        );
        let down = gpu.upload_f16_from_f32(
            &safe_tensor.load_layer_weight(LoadLayerWeightArgs { name: "mlp.shared_experts.down_proj", layer_no}),
            &format!("shared_down{}", layer_no)
        );

        return (Some(gate), Some(up), Some(down));
    }

    (None, None, None)
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
