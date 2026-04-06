use std::borrow::Cow;
use crate::root::arch::common::LoadLayer;
use crate::root::arch::safetensors::{SafeTensor};
use crate::root::common_types::{Arrf32, Vecf32, WeightF16};
use crate::root::config::{MLAConfig, NormConfig};
use crate::root::gpu::Gpu;
use crate::root::layer::{load_shared_experts, FFNLayer, MLAKVLayer, MOELayer, SharedExpertLayer};
use crate::root::matmul::matmul_transposed_f16;
use crate::root::moe::{load_experts_awq, load_experts_bf16, load_experts_ct, StandardExpert};
use crate::root::norm::rms_norm;

pub struct GLMLayer {
    // Attention
    pub input_norm_w: Vecf32,
    pub post_norm_w: Vecf32,
    pub q_a_proj_w: WeightF16,        // [2048, 768]
    pub q_b_proj_w: WeightF16,        // [768, 5120]
    pub q_a_layernorm_w: Vecf32,      // [768]
    pub kv_a_proj_w: WeightF16, // [2048, 576] (512 + 64)
    pub kv_b_proj_w: WeightF16,       // [512, 8960] (20 * (192+256))
    pub kv_a_layernorm_w: Vecf32,     // [512]
    pub o_proj_w: WeightF16,          // [5120, 2048]
    // FFN (GLM uses SwiGLU like LLaMA)
    pub gate_proj_w: Option<WeightF16>,       // [2048, intermediate_size]
    pub up_proj_w: Option<WeightF16>,         // [2048, intermediate_size]
    pub down_proj_w: Option<WeightF16>,       // [intermediate_size, 2048]

    // MoE
    pub expert_router_w: Option<Vecf32>,
    pub experts: Option<Vec<StandardExpert>>,

    pub shared_gate_proj_w: Option<WeightF16>,
    pub shared_up_proj_w: Option<WeightF16>,
    pub shared_down_proj_w: Option<WeightF16>,
}

impl LoadLayer for GLMLayer {
    fn load_layer(safe_tensor: &mut SafeTensor, layer_no: usize, gpu: &mut Gpu) -> Self {

        let input_norm_w = safe_tensor.load_layer_weight("input_layernorm", layer_no);
        let post_norm_w = safe_tensor.load_layer_weight("post_attention_layernorm", layer_no);

        // MLA Query projections (GLM-specific LoRA path)
        let q_a_proj_w = gpu.upload_f16_from_f32(
            &safe_tensor.load_layer_weight("self_attn.q_a_proj", layer_no),
            &format!("q_a_proj_{}", layer_no)
        );
        let q_b_proj_w = gpu.upload_f16_from_f32(
            &safe_tensor.load_layer_weight("self_attn.q_b_proj", layer_no),
            &format!("q_b_proj_{}", layer_no)
        );
        let q_a_layernorm_w = safe_tensor.load_layer_weight("self_attn.q_a_layernorm", layer_no);

        // MLA KV projections
        let kv_a_proj_w = gpu.upload_f16_from_f32(
            &safe_tensor.load_layer_weight("self_attn.kv_a_proj_with_mqa", layer_no),
            &format!("kv_a_proj_{}", layer_no)
        );
        let kv_b_proj_w = gpu.upload_f16_from_f32(
            &safe_tensor.load_layer_weight("self_attn.kv_b_proj", layer_no),
            &format!("kv_b_proj_{}", layer_no)
        );
        let kv_a_layernorm_w = safe_tensor.load_layer_weight("self_attn.kv_a_layernorm", layer_no);

        let o_proj_w = gpu.upload_f16_from_f32(
            &safe_tensor.load_layer_weight("self_attn.o_proj", layer_no),
            &format!("o_proj_{}", layer_no)
        );

        // FFN
        let (gate_proj_w, up_proj_w, down_proj_w) = safe_tensor.load_dense_ffns(layer_no, gpu);

        // MoE
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

        // Shared experts
        let (shared_gate_proj_w, shared_up_proj_w, shared_down_proj_w) =
            load_shared_experts(safe_tensor, layer_no, gpu);

        Self {
            input_norm_w, post_norm_w,
            q_a_proj_w, q_b_proj_w, q_a_layernorm_w,
            kv_a_proj_w, kv_b_proj_w, kv_a_layernorm_w,
            o_proj_w,
            gate_proj_w, up_proj_w, down_proj_w,
            expert_router_w, experts,
            shared_gate_proj_w, shared_up_proj_w, shared_down_proj_w,
        }
    }
}
impl MLAKVLayer for GLMLayer {
    fn get_kv_a_proj_w(&self) -> &WeightF16 {
        &self.kv_a_proj_w
    }

    fn get_kv_b_proj_w(&self) -> &WeightF16 {
        &self.kv_b_proj_w
    }

    fn get_kv_a_layernorm_w(&self) -> &Vecf32 {
        &self.kv_a_layernorm_w
    }

    fn get_o_proj_w(&self) -> &WeightF16 {
        &self.o_proj_w
    }

    // In GLMLayer's compute_q_raw_normed:
    fn compute_q_raw_normed<'a, TC: NormConfig + MLAConfig>(
        &'a self, normed: &'a Vecf32, config: &TC, gpu: &mut Gpu
    ) -> Cow<'a, Arrf32> {
        let q_lora_rank = self.q_a_layernorm_w.len();
        let latent = matmul_transposed_f16(normed, &self.q_a_proj_w, q_lora_rank, gpu);
        let normed_latent = rms_norm(&latent, &self.q_a_layernorm_w, config.get_rms_norm_eps());

        Cow::Owned(normed_latent)
    }

    fn get_q_raw_proj_w(&self) -> &WeightF16 {
        &self.q_b_proj_w
    }
}




impl FFNLayer for GLMLayer {
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

impl MOELayer for GLMLayer {
    fn get_expert_router_w(&self) -> &Option<Vecf32> {
        &self.expert_router_w
    }

    fn get_experts(&self) -> &Option<Vec<StandardExpert>> {
        &self.experts
    }
}

impl SharedExpertLayer for GLMLayer {
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
