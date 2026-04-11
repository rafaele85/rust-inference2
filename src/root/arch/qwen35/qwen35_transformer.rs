use crate::root::arch::qwen35::qwen35::{Qwen35Layer, Qwen35Model};
use crate::root::arch::qwen35::qwen35_attention::qwen35_full_attention;
use crate::root::arch::qwen35::qwen35_config::TextConfig;
use crate::root::arch::qwen35::qwen35_layer::{Qwen35MoeWeights};
use crate::root::arch::qwen35::qwen35_linear_attention::qwen35_linear_attention;
use crate::root::arch::qwen35::qwen35_moe::qwen35_moe_ffn;
use crate::root::common_types::Vecf32;
use crate::root::gpu::{get_default_gpu, get_gpu, Gpu};
use crate::root::kv_cache::KVCache;
use crate::root::logits::logits;
use crate::root::norm::rms_norm;

pub fn qwen35_generate_token(
    token_id: usize,
    position: usize,
    model: &Qwen35Model,
    kv_cache: &mut KVCache,
    gpus: &mut Vec<Gpu>
) -> usize {

    let tc = &model.config.text_config;
    let token_start = token_id * tc.hidden_size;
    let embed_token = &model.embed_weights[token_start..(token_start + tc.hidden_size)];
    let mut hidden = embed_token.to_vec();

    for i in 0..tc.num_hidden_layers {
        let gpu = get_gpu(gpus, i, tc.num_hidden_layers);
        let input_norm_w = layer_input_norm(model, i);
        let normed = rms_norm(&hidden, input_norm_w, tc.rms_norm_eps);
        layer_linear_attention(model, i, &mut hidden, &normed, &tc, kv_cache, position, gpu);
        let post_norm_w = layer_post_norm(model, i);
        let post_normed = rms_norm(&hidden, post_norm_w, tc.rms_norm_eps);
        let moe = layer_moe(model, i);
        qwen35_moe_ffn(&mut hidden, &post_normed, moe, tc.num_experts_per_tok, gpu);
    }

    let default_gpu = get_default_gpu(gpus, "logits");
    let res = logits(hidden, &model.final_norm_w, &model.lm_head_w, tc.vocab_size, tc.rms_norm_eps, default_gpu);

    res
}


fn layer_input_norm(model: &Qwen35Model, layer_no: usize) -> &Vecf32 {
    match &model.layers[layer_no] {
        Qwen35Layer::Linear(l) => &l.input_norm_w,
        Qwen35Layer::Full(l) => &l.input_norm_w,
    }
}

fn layer_linear_attention(
    model: &Qwen35Model, layer_no: usize, hidden: &mut Vecf32, normed: &Vecf32, tc: &TextConfig,
    kv_cache: &mut KVCache, position: usize, gpu: &mut Gpu
) {
    match &model.layers[layer_no] {
        Qwen35Layer::Linear(l) => {
            qwen35_linear_attention(hidden, &normed, l, tc, kv_cache, position, gpu);
        },
        Qwen35Layer::Full(l) => {
            qwen35_full_attention(hidden, &normed, l, tc, kv_cache, layer_no, position, gpu);
        }
    }
}

fn layer_post_norm(model: &Qwen35Model, layer_no: usize) -> &Vecf32 {
    match &model.layers[layer_no] {
        Qwen35Layer::Linear(l) => &l.post_norm_w,
        Qwen35Layer::Full(l) => &l.post_norm_w
    }
}

fn layer_moe(model: &Qwen35Model, layer_no: usize) -> &Qwen35MoeWeights {
    match &model.layers[layer_no] {
        Qwen35Layer::Linear(l) => &l.moe,
        Qwen35Layer::Full(l) => &l.moe
    }
}