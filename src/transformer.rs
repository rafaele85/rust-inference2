use crate::activation::gelu;
use crate::attention::{cached_attention, extract_qkv, CachedAttentionArgs, QKV};
use crate::common_types::{Vecf32};
use crate::embeddings::combine_embedding_weights;
use crate::gpu::Gpu;
use crate::kv_cache::KVCache;
use crate::linear::{normalize_and_project, project_and_residual, NormalizeAndProject, ProjectAndResidualArgs};
use crate::logits::{compute_logits, ComputeLogitsArgs};
use crate::model::gpt2::{GPT2Layer, GPT2Model};

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
    pub gpu: &'a Gpu
}

pub fn generate_token_single_layer(args: GenerateTokenSingleLayerArgs) {
    let GenerateTokenSingleLayerArgs {
        hidden, layer, layer_no, qkv_size,
        kv_cache, n_heads, dim, head_dim, ffn_dim, gpu
    } = args;

    let qkv = normalize_and_project(NormalizeAndProject {
        hidden, norm_weights: &layer.ln1_w, norm_bias: &layer.ln1_b,
        projection_weights: &layer.qkv_w, projection_bias: &layer.qkv_b, projection_dim: qkv_size, gpu
    });

    let QKV {q, k, v} = extract_qkv(&qkv);

    kv_cache.append(layer_no, k, v);

    let k_cache = &kv_cache.k[layer_no];
    let v_cache = &kv_cache.v[layer_no];
    let cache_len = k_cache.len();
    let attn_raw = cached_attention( CachedAttentionArgs {
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
       hidden, norm_weights: &layer.ln2_w, norm_bias: &layer.ln2_b, projection_weights: &layer.ffn_up_w, projection_bias: &layer.ffn_up_b, projection_dim: ffn_dim, gpu
    });

    let ffn_act = gelu(&ffn_up);

    project_and_residual(ProjectAndResidualArgs {
        activation_weights: &ffn_act, projection_weights: &layer.ffn_down_w, projection_bias: &layer.ffn_down_b, dim, hidden, gpu
    });
}

pub struct GenerateTokenArgs<'a> {
    pub token_id: usize,
    pub dim: usize,
    pub position: usize,
    pub num_layers: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub kv_cache: &'a mut KVCache,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub gpu: &'a Gpu,
    pub gpt2_model: &'a GPT2Model
}
pub fn generate_token(args: GenerateTokenArgs) -> usize {
    let GenerateTokenArgs {
        token_id, dim, position,num_layers,
        n_heads, ffn_dim, kv_cache, head_dim,
        vocab_size, gpu, gpt2_model
    } = args;

    let wte_start = token_id * dim;
    let wpe_start = position * dim;
    let wte_row = &gpt2_model.wte[wte_start..wte_start + dim];
    let wpe_row = &gpt2_model.wpe[wpe_start..wpe_start + dim];
    let qkv_size = dim * 3;

    let mut hidden = combine_embedding_weights(wte_row, wpe_row);

    for layer_no in 0..num_layers {
        let layer = &gpt2_model.layers[layer_no];
        generate_token_single_layer(GenerateTokenSingleLayerArgs {
            layer_no, layer, qkv_size, hidden: &mut hidden, n_heads, ffn_dim, kv_cache, dim, head_dim, gpu
        });
    }
    let top_id = compute_logits(ComputeLogitsArgs {hidden: &hidden, gpt2_model: &gpt2_model, last_pos: 0, dim, vocab_size, gpu});

    top_id
}