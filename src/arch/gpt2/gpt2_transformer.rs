use crate::arch::gpt2::gpt2_attention::{gpt2_cached_attention, gpt2_extract_qkv, GPT2CachedAttentionArgs, GPT2QKV};
use crate::common_types::{Vecf32};
use crate::arch::gpt2::gpt2_embeddings::gpt2_combine_embedding_weights;
use crate::gpu::{get_default_gpu, get_gpu, Gpu};
use crate::kv_cache::KVCache;
use crate::arch::gpt2::gpt2_linear::{normalize_and_project, project_and_residual, NormalizeAndProject, ProjectAndResidualArgs};
use crate::arch::gpt2::gpt2_logits::{gpt2_compute_logits, GPT2ComputeLogitsArgs};
use crate::arch::gpt2::gpt2::{GPT2Model};
use crate::arch::gpt2::gpt2_activation::gelu;
use crate::arch::gpt2::gpt2_layer::GPT2Layer;

pub struct GenerateTokenSingleLayerArgs<'a> {
    pub layer_no: usize,
    pub qkv_size: usize,
    pub n_heads: usize,
    pub dim: usize,
    pub hidden: &'a mut Vecf32,
    pub layer: &'a GPT2Layer,
    pub kv_cache: &'a mut KVCache,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub gpu: &'a Gpu,
    pub eps: f32
}

fn generate_token_single_layer(args: GenerateTokenSingleLayerArgs) {
    let GenerateTokenSingleLayerArgs {
        hidden, layer, layer_no, qkv_size,
        kv_cache, n_heads, dim, head_dim, ffn_dim, gpu, eps
    } = args;

    let qkv = normalize_and_project(NormalizeAndProject {
        hidden, norm_weights: &layer.ln1_w, norm_bias: &layer.ln1_b,
        projection_weights: &layer.qkv_w, projection_bias: &layer.qkv_b, projection_dim: qkv_size, gpu,
        eps
    });

    let GPT2QKV {q, k, v} = gpt2_extract_qkv(&qkv);

    kv_cache.append(layer_no, k, v);

    let k_cache = &kv_cache.k[layer_no];
    let v_cache = &kv_cache.v[layer_no];
    let cache_len = k_cache.len();
    let attn_raw = gpt2_cached_attention( GPT2CachedAttentionArgs {
        q: &q,
        k_cache,
        v_cache,
        cache_len,
        n_heads,
        dim,
        head_dim,
    });

    project_and_residual(ProjectAndResidualArgs {
        activation_weights: &attn_raw, projection_weights: &layer.proj_w, projection_bias: &layer.proj_b, dim, hidden, gpu
    });

    let ffn_up = normalize_and_project(NormalizeAndProject {
       hidden, norm_weights: &layer.ln2_w, norm_bias: &layer.ln2_b, projection_weights: &layer.ffn_up_w, projection_bias: &layer.ffn_up_b, projection_dim: ffn_dim, gpu, eps
    });

    let ffn_act = gelu(&ffn_up);

    project_and_residual(ProjectAndResidualArgs {
        activation_weights: &ffn_act, projection_weights: &layer.ffn_down_w, projection_bias: &layer.ffn_down_b, dim, hidden, gpu
    });
}

pub struct GPT2GenerateTokenArgs<'a> {
    pub token_id: usize,
    pub position: usize,
    pub kv_cache: &'a mut KVCache,
    pub gpus: &'a Vec<Gpu>,
    pub model: &'a GPT2Model
}
pub fn gpt2_generate_token(args: GPT2GenerateTokenArgs) -> usize {
    let GPT2GenerateTokenArgs {
        token_id, position,
        kv_cache,
        gpus, model
    } = args;

    let config = &model.config;
    let wte_start = token_id * model.config.n_embd;
    let wpe_start = position * model.config.n_embd;
    let wte_row = &model.wte[wte_start..wte_start + config.n_embd];
    let wpe_row = &model.wpe[wpe_start..wpe_start + config.n_embd];
    let qkv_size = config.n_embd * 3;

    let mut hidden = gpt2_combine_embedding_weights(wte_row, wpe_row);

    for layer_no in 0..config.n_layer {
        let layer = &model.layers[layer_no];
        let gpu = get_gpu(gpus, layer_no, config.n_layer);
        generate_token_single_layer(GenerateTokenSingleLayerArgs {
            layer_no, layer, qkv_size, hidden: &mut hidden,
            kv_cache,
            n_heads: config.n_head, ffn_dim: config.ffn_dim, dim: config.n_embd, head_dim: config.head_dim,
            gpu, eps: config.layer_norm_epsilon
        });
    }
    let default_gpu = get_default_gpu(gpus, "logits");
    let top_id = gpt2_compute_logits(GPT2ComputeLogitsArgs {hidden: &hidden, gpt2_model: &model, last_pos: 0, dim: config.n_embd, vocab_size: config.vocab_size, gpu: default_gpu});

    top_id
}