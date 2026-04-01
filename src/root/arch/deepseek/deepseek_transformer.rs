use crate::root::activation::silu;
use crate::root::arch::deepseek::deepseek::DeepseekModel;
use crate::root::arch::deepseek::deepseek_attention::{deepseek_mla_attention, DeepseekMLAAttentionArgs};
use crate::root::arch::deepseek::deepseek_config::DeepseekConfig;
use crate::root::arch::deepseek::deepseek_layer::DeepseekLayer;
use crate::root::common_types::{Vecf32, WeightF16};
use crate::root::gpu::{get_default_gpu, get_gpu, Gpu};
use crate::root::kv_cache::KVCache;
use crate::root::linear::{add_to_vector, hadamard};
use crate::root::logits::logits;
use crate::root::matmul::matmul_transposed_f16;
use crate::root::moe::{moe_ffn, MoeFFN};
use crate::root::norm::rms_norm;

pub struct DeepseekGenerateTokenArgs<'a> {
    pub token_id: usize,
    pub position: usize,
    pub model: &'a DeepseekModel,
    pub kv_cache: &'a mut KVCache,
    pub gpus: &'a mut Vec<Gpu>,
}

pub fn deepseek_generate_token(args: DeepseekGenerateTokenArgs) -> usize {
    let DeepseekGenerateTokenArgs {token_id, position, model, kv_cache, mut gpus} = args;
    let DeepseekModel {config, embed_weights, layers, final_norm_w, lm_head_w, ..} = model;
    let token_start = token_id * config.hidden_size;

    let embed_token = &embed_weights[token_start..token_start + config.hidden_size];

    let mut hidden = embed_token.to_vec();

    for i in 0..config.num_hidden_layers {
        let gpu = get_gpu(gpus, i, config.num_hidden_layers);
        deepseek_layer(DeepseekLayerArgs {
            hidden: &mut hidden,
            layer: &layers[i],
            config,
            kv_cache,
            layer_no: i,
            position,
            gpu,
        });
    }

    let default_gpu = get_default_gpu(&mut gpus, "logits");
    let output_token_id = logits(hidden, final_norm_w, lm_head_w, config.vocab_size, config.rms_norm_eps, default_gpu);

    output_token_id
}


pub struct DeepseekLayerArgs<'a> {
    pub hidden: &'a mut Vecf32,
    pub layer: &'a DeepseekLayer,
    pub config: &'a DeepseekConfig,
    pub kv_cache: &'a mut KVCache,
    pub layer_no: usize,
    pub position: usize,
    pub gpu: &'a mut Gpu,
}


pub fn deepseek_layer(args: DeepseekLayerArgs) {
    let DeepseekLayerArgs { layer, config, kv_cache, layer_no, position, mut gpu, hidden} = args;

    let mut normed = rms_norm(hidden, &layer.input_norm_w, config.rms_norm_eps);

    deepseek_mla_attention(DeepseekMLAAttentionArgs {config, gpu: &mut gpu, layer, hidden, kv_cache, layer_no, position, normed: &mut normed});

    if let (Some(router_w), Some(experts)) = (&layer.expert_router_w, &layer.experts) {
        let normed2 = rms_norm(hidden, &layer.post_norm_w, config.rms_norm_eps);
        moe_ffn(MoeFFN {
            hidden, normed: &normed2, router_w, experts, layer_no,
            num_experts_per_tok: config.num_experts_per_tok, moe_intermediate_size: config.moe_intermediate_size, hidden_size: config.hidden_size, gpu
        });
        deepseek_shared_experts(DeepseekSharedExpertsArgs {hidden, normed: &normed2, layer, config, gpu, layer_no});
    } else {
        if let (Some(gate_proj_w), Some(up_proj_w), Some(down_proj_w)) = (&layer.gate_proj_w, &layer.up_proj_w, &layer.down_proj_w) {
            deepseek_ffn(hidden, layer, gate_proj_w, up_proj_w, down_proj_w, config, gpu);
        }
    }
}


fn deepseek_ffn(hidden: &mut Vecf32, layer: &DeepseekLayer, gate_proj_w: &WeightF16, up_proj_w: &WeightF16, down_proj_w: &WeightF16, config: &DeepseekConfig, gpu: &mut Gpu) {
    let normed2 = rms_norm(&hidden, &layer.post_norm_w, config.rms_norm_eps);

    let gate = matmul_transposed_f16(&normed2, &gate_proj_w, config.intermediate_size, gpu);

    let up = matmul_transposed_f16(&normed2, &up_proj_w, config.intermediate_size, gpu);

    let silu_gate = silu(&gate);

    let ffn_act = hadamard(silu_gate, up, config.intermediate_size);

    let ffn_out = matmul_transposed_f16(&ffn_act, &down_proj_w, config.hidden_size, gpu);

    add_to_vector(hidden, &ffn_out);
}

struct DeepseekSharedExpertsArgs<'a> {
    hidden: &'a mut Vecf32,
    normed: &'a Vecf32,
    layer: &'a DeepseekLayer,
    config: &'a DeepseekConfig,
    gpu: &'a mut Gpu,
    layer_no: usize
}
fn deepseek_shared_experts(args: DeepseekSharedExpertsArgs) {
    let DeepseekSharedExpertsArgs {hidden, normed, layer, config, gpu, layer_no} = args;

    if let (Some(gate_w), Some(up_w), Some(down_w)) = (&layer.shared_gate_proj_w, &layer.shared_up_proj_w, &layer.shared_down_proj_w) {
        let shared_intermediate = config.n_shared_experts * config.moe_intermediate_size;
        let gate = matmul_transposed_f16(normed, gate_w, shared_intermediate, gpu);
        let silu_gate = silu(&gate);
        let up = matmul_transposed_f16(normed, up_w, shared_intermediate, gpu);
        let ffn_act = hadamard(silu_gate, up, shared_intermediate);
        let ffn_out = matmul_transposed_f16(&ffn_act, down_w, config.hidden_size, gpu);

        add_to_vector(hidden, &ffn_out);
    }
}