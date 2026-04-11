use crate::root::activation::silu;
use crate::root::arch::llama::llama::LlamaModel;
use crate::root::arch::llama::llama_attention::{llama_gqa, LlamaGQAArgs};
use crate::root::arch::llama::llama_config::LlamaConfig;
use crate::root::arch::llama::llama_layer::LlamaLayer;
use crate::root::common_types::{Vecf32, WeightF16};
use crate::root::gpu::{get_default_gpu, get_gpu, Gpu};
use crate::root::kv_cache::KVCache;
use crate::root::linear::{add_to_vector, hadamard};
use crate::root::logits::logits;
use crate::root::matmul::matmul_transposed_f16;
use crate::root::moe::{moe_ffn, MoeFFN};
use crate::root::norm::rms_norm;

fn llama_ffn(hidden: &mut Vecf32, layer: &LlamaLayer, gate_proj_w: &WeightF16, up_proj_w: &WeightF16, down_proj_w: &WeightF16, config: &LlamaConfig, gpu: &mut Gpu) {
    let normed2 = rms_norm(&hidden, &layer.post_norm_w, config.rms_norm_eps);

    let gate = matmul_transposed_f16(&normed2, &gate_proj_w, config.intermediate_size, gpu);

    let up = matmul_transposed_f16(&normed2, &up_proj_w, config.intermediate_size, gpu);

    let silu_gate = silu(&gate);

    let ffn_act = hadamard(silu_gate, up, config.intermediate_size);

    let ffn_out = matmul_transposed_f16(&ffn_act, &down_proj_w, config.hidden_size, gpu);

    add_to_vector(hidden, &ffn_out);
}

pub struct LlamaLayerArgs<'a> {
    pub hidden: &'a mut Vecf32,
    pub layer: &'a LlamaLayer,
    pub config: &'a LlamaConfig,
    pub kv_cache: &'a mut KVCache,
    pub layer_no: usize,
    pub position: usize,
    pub gpu: &'a mut Gpu,
}


pub fn llama_layer(args: LlamaLayerArgs) {
    let LlamaLayerArgs { layer, config, kv_cache, layer_no, position, mut gpu, hidden} = args;

    let mut normed = rms_norm(hidden, &layer.input_norm_w, config.rms_norm_eps);

    llama_gqa(LlamaGQAArgs {config, gpu: &mut gpu, layer, hidden, kv_cache, layer_no, position, normed: &mut normed});

    if let (Some(router_w), Some(experts)) = (&layer.expert_router_w, &layer.experts) {
        let normed2 = rms_norm(hidden, &layer.post_norm_w, config.rms_norm_eps);
        moe_ffn(MoeFFN {
            hidden, normed: &normed2, router_w: &router_w, experts: &experts,
            config,
            gpu,
        });
    } else {
        if let (Some(gate_proj_w), Some(up_proj_w), Some(down_proj_w)) = (&layer.gate_proj_w, &layer.up_proj_w, &layer.down_proj_w) {
            llama_ffn(hidden, layer, gate_proj_w, up_proj_w, down_proj_w, config, gpu);
        }
    }
}

pub struct LlamaGenerateTokenArgs<'a> {
    pub token_id: usize,
    pub position: usize,
    pub model: &'a LlamaModel,
    pub kv_cache: &'a mut KVCache,
    pub gpus: &'a mut Vec<Gpu>,
}

pub fn llama_generate_token(args: LlamaGenerateTokenArgs) -> usize {
    let LlamaGenerateTokenArgs {token_id, position, model, kv_cache, mut gpus} = args;
    let LlamaModel {config, embed_weights, layers, final_norm_w, lm_head_w, ..} = model;
    let token_start = token_id * config.hidden_size;
    let embed_token = &embed_weights[token_start..token_start + config.hidden_size];

    let mut hidden = embed_token.to_vec();

    for i in 0..config.num_hidden_layers {
        let gpu = get_gpu(gpus, i, config.num_hidden_layers);
        llama_layer(LlamaLayerArgs {
            kv_cache,
            layer: &layers[i],
            config,
            layer_no: i,
            position,
            gpu,
            hidden: &mut hidden
        });
    }

    let default_gpu = get_default_gpu(&mut gpus, "logits");
    let output_token_id = logits(&hidden, final_norm_w, lm_head_w, config.vocab_size, config.rms_norm_eps, default_gpu);

    output_token_id
}