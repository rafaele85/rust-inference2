use std::borrow::Cow;
use crate::root::arch::common::LoadLayer;
use crate::root::arch::safetensors::SafeTensor;
use crate::root::common_types::{Arrf32, Vecf32, WeightF16};
use crate::root::config::{MLAConfig, NormConfig};
use crate::root::gpu::Gpu;
use crate::root::layer::{load_common_weights, load_shared_experts, CommonWeights, FFNLayer, MLAKVLayer, MOELayer, SharedExpertLayer};
use crate::root::moe::StandardExpert;

pub struct DeepseekLayer {
    pub input_norm_w: Vecf32,
    pub post_norm_w: Vecf32,

    pub q_proj_w: WeightF16,          // hidden → num_heads * (nope + rope)
    pub o_proj_w: WeightF16,          // output projection
    pub kv_a_proj_w: WeightF16,       // hidden → kv_lora_rank + rope_dim
    pub kv_b_proj_w: WeightF16,       // kv_lora_rank → num_heads * (nope + v_head_dim)

    pub kv_a_layernorm_w: Vecf32,     //norm on compressed KV, size kv_lora_rank

    pub gate_proj_w: Option<WeightF16>,
    pub up_proj_w: Option<WeightF16>,
    pub down_proj_w: Option<WeightF16>,

    pub expert_router_w : Option<Vecf32>,
    pub experts: Option<Vec<StandardExpert>>,

    pub shared_gate_proj_w: Option<WeightF16>,
    pub shared_up_proj_w: Option<WeightF16>,
    pub shared_down_proj_w: Option<WeightF16>,
}

impl FFNLayer for DeepseekLayer {
    fn get_input_norm_w(&self) -> &Vecf32 {
        &self.input_norm_w
    }

    fn get_post_norm_w(&self) -> &Vecf32 {
        &self.post_norm_w
    }

    fn get_gate_proj_w(&self) -> &Option<WeightF16> {
        &self.gate_proj_w
    }

    fn get_up_proj_w(&self) -> &Option<WeightF16> {
        &self.up_proj_w
    }

    fn get_down_proj_w(&self) -> &Option<WeightF16> {
        &self.down_proj_w
    }
}

impl MOELayer for DeepseekLayer {
    fn get_expert_router_w(&self) -> &Option<Vecf32> {
        &self.expert_router_w
    }

    fn get_experts(&self) -> &Option<Vec<StandardExpert>> {
        &self.experts
    }
}

impl MLAKVLayer for DeepseekLayer {
    fn get_kv_a_layernorm_w(&self) -> &Vecf32 {
        &self.kv_a_layernorm_w
    }

    fn get_kv_a_proj_w(&self) -> &WeightF16 {
        &self.kv_a_proj_w
    }

    fn get_kv_b_proj_w(&self) -> &WeightF16 {
        &self.kv_b_proj_w
    }

    fn get_o_proj_w(&self) -> &WeightF16 {
        &self.o_proj_w
    }

    fn compute_q_raw_normed<'a, TC: NormConfig + MLAConfig>(&'a self, normed: &'a Vecf32, _config: &TC, _gpu: &mut Gpu) -> Cow<'a, Arrf32> {
        Cow::Borrowed(normed)
    }

    fn get_q_raw_proj_w(&self) -> &WeightF16 {
        &self.q_proj_w
    }
}

impl SharedExpertLayer for DeepseekLayer {
    fn get_shared_gate_proj_w(&self) -> &Option<WeightF16> {
        &self.shared_gate_proj_w
    }

    fn get_shared_up_proj_w(&self) -> &Option<WeightF16> {
        &self.shared_up_proj_w
    }

    fn get_shared_down_proj_w(&self) -> &Option<WeightF16> {
        &self.shared_down_proj_w
    }
}


impl LoadLayer for DeepseekLayer {
    fn load_layer(safe_tensor: &mut SafeTensor, layer_no: usize, gpu: &mut Gpu) -> Self {
        let common = load_common_weights(safe_tensor, layer_no, gpu);
        let CommonWeights {input_norm_w, post_norm_w, q_proj_w, o_proj_w, gate_proj_w, up_proj_w, down_proj_w, experts, expert_router_w} = common;

        let kv_a_layernorm_w = safe_tensor.load_layer_weight("self_attn.kv_a_layernorm", layer_no);
        let kv_a_proj_w = gpu.upload_f16_from_f32(&safe_tensor.load_layer_weight("self_attn.kv_a_proj_with_mqa", layer_no), &format!("kv_a_proj_w{}", layer_no));
        let kv_b_proj_w = gpu.upload_f16_from_f32(&safe_tensor.load_layer_weight("self_attn.kv_b_proj", layer_no), &format!("kv_b_proj_w{}", layer_no));
        let (shared_gate_proj_w, shared_up_proj_w, shared_down_proj_w) = load_shared_experts(safe_tensor, layer_no, gpu);

        Self {
            input_norm_w, post_norm_w,
            q_proj_w, kv_a_proj_w, kv_a_layernorm_w, kv_b_proj_w, o_proj_w,
            gate_proj_w, up_proj_w, down_proj_w,
            expert_router_w, experts,
            shared_gate_proj_w,
            shared_up_proj_w,
            shared_down_proj_w,
        }
    }
}
