use crate::root::arch::glm::glm::GLMModel;
use crate::root::generate::{apply_layer, ApplyLayerArgs};
use crate::root::gpu::{get_default_gpu, get_gpu, Gpu};
use crate::root::kv_cache::KVCache;
use crate::root::logits::logits;

pub struct GLMGenerateTokenArgs<'a> {
    pub token_id: usize,
    pub position: usize,
    pub model: &'a GLMModel,
    pub kv_cache: &'a mut KVCache,
    pub gpus: &'a mut Vec<Gpu>,
}

pub fn glm_generate_token(args: GLMGenerateTokenArgs) -> usize {
    let GLMGenerateTokenArgs {token_id, position, model, kv_cache, mut gpus} = args;
    let GLMModel {config, embed_weights, layers, final_norm_w, lm_head_w, ..} = model;
    let token_start = token_id * config.hidden_size;

    let embed_token = &embed_weights[token_start..token_start + config.hidden_size];

    let mut hidden = embed_token.to_vec();

    for i in 0..config.num_hidden_layers {
        let gpu = get_gpu(gpus, i, config.num_hidden_layers);
        apply_layer(ApplyLayerArgs {
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


