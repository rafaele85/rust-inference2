use crate::activation::gelu;
use crate::attention::{cached_attention, extract_qkv, extract_v, multi_head_attention, CachedAttentionArgs, QKV};
use crate::common_types::{Arrf32, Vecf32};
use crate::embeddings::combine_embedding_weights;
use crate::kv_cache::KVCache;
use crate::linear::{normalize_and_project, project_and_residual, NormalizeAndProject, ProjectAndResidualArgs};
use crate::logits::{compute_logits, ComputeLogitsArgs};
use crate::matmul::matmul_transposed;
use crate::norm::layer_norm;
use crate::safetensors::Layer;

pub struct TransformerPassArgs<'a> {
    pub hidden: &'a mut Vec<f32>,
    pub wte: &'a Vec<f32>,
    pub wpe: &'a Vec<f32>,
    pub layer: &'a Layer,
    pub norm_final_weight: &'a Vec<f32>,
    pub norm_final_bias: &'a Vec<f32>,
    pub dim: usize,
}

pub fn transformer_pass(args: TransformerPassArgs) {
    let ffn_dim = 3072;
    let qkv_combined_dim = args.dim * 3; // all 3 weights are of the same size

    let layer = &args.layer;

    let qkv = normalize_and_project(NormalizeAndProject {
        hidden: args.hidden, norm_weights: &layer.ln1_w, norm_bias: &layer.ln1_b, projection_weights: &layer.qkv_w, projection_bias: &layer.qkv_b, projection_dim: qkv_combined_dim
    });

    let v = extract_v(&qkv);

    project_and_residual(ProjectAndResidualArgs {activation_weights: &v, projection_weights: &layer.proj_w, projection_bias: &layer.proj_b, dim: args.dim, hidden: args.hidden });

    let ffn_up = normalize_and_project(NormalizeAndProject {
        hidden: args.hidden, norm_weights: &layer.ln2_w, norm_bias: &layer.ln2_b, projection_weights: &layer.ffn_up_w, projection_bias: &layer.ffn_up_b, projection_dim: ffn_dim
    });

    let ffn_act = gelu(&ffn_up);

    project_and_residual(ProjectAndResidualArgs {activation_weights: &ffn_act, projection_weights: &layer.ffn_down_w, projection_bias: &layer.ffn_down_b, dim: args.dim, hidden: args.hidden });
}

pub struct TransformerArgs<'a> {
    pub num_layers: usize,
    pub wte: &'a Vec<f32>,
    pub wpe: &'a Vec<f32>,
    pub layers: &'a Vec<Layer>,
    pub norm_final_weight: &'a Vec<f32>,
    pub norm_final_bias: &'a Vec<f32>,
    pub dim: usize,
    pub token_id: usize,
    pub ln_f_w: &'a Vec<f32>,
    pub ln_f_b: &'a Vec<f32>,
}

pub fn transformer(args: TransformerArgs) -> usize {
    let wte_start = args.token_id * args.dim;
    let wte_row = &args.wte[wte_start..wte_start + args.dim];
    let wpe_row = &args.wpe[0..args.dim];

    let mut hidden: Vec<f32> = combine_embedding_weights(wte_row, wpe_row);

    for i in 0..args.num_layers {
        transformer_pass(TransformerPassArgs {
            hidden: &mut hidden, wte: args.wte, wpe: args.wpe, layer: &args.layers[i], norm_final_weight: args.norm_final_weight, norm_final_bias: args.norm_final_bias, dim: args.dim
        });
    }

    hidden = layer_norm(&hidden, &args.ln_f_w, &args.ln_f_b);

    let vocab_size = 50257;
    let logits = matmul_transposed(&hidden, &args.wte, vocab_size);
    let top_id = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;

    top_id
}

pub struct InitializeHiddenArgs<'a> {
    pub hidden_size: usize,
    pub dim: usize,
    pub wte: &'a Vec<f32>,
    pub wpe: &'a Vec<f32>,
    pub token_ids: &'a Vec<usize>
}
pub fn initialize_hidden(args: InitializeHiddenArgs) -> Vec<f32> {
    let InitializeHiddenArgs {hidden_size, dim, wte, wpe, token_ids} = args;
    let mut hidden = vec![0.0f32; hidden_size];

    for (pos, &token_id) in token_ids.iter().enumerate() {
        for i in 0..dim {
            let offset = pos * dim;
            hidden[offset + i] = wte[token_id * dim + i] + wpe[offset + i];
        }
    }

    hidden
}

fn cache_kv_for_position(qkv: &Arrf32, dim: usize, kv_cache: &mut KVCache, layer_no: usize) {
    let k = qkv[dim..dim+dim].to_vec();
    let v = qkv[dim+dim..dim+dim+dim].to_vec();
    kv_cache.append(layer_no, k, v);
}

pub struct FillAllQKVArgs<'a> {
    pub all_qkv_size: usize,
    pub seq_len: usize,
    pub dim: usize,
    pub hidden: &'a Vecf32,
    pub layer: &'a Layer,
    pub qkv_size: usize,
    pub kv_cache: &'a mut KVCache,
    pub layer_no: usize
}

pub fn fill_all_qkv(args: FillAllQKVArgs) -> Vecf32 {
    let FillAllQKVArgs {all_qkv_size, seq_len, dim, hidden, layer, qkv_size, kv_cache, layer_no} = args;
    let mut all_qkv = vec![0.0f32; all_qkv_size];
    for pos in 0..seq_len {
        let offset = pos * dim;
        let mut h = &hidden[offset..offset + dim];
        let qkv = normalize_and_project(NormalizeAndProject {
            hidden: &mut h.to_vec(), norm_weights: &layer.ln1_w, norm_bias: &layer.ln1_b, projection_weights: &layer.qkv_w, projection_bias: &layer.qkv_b, projection_dim: qkv_size
        });
        let qkv_offset = pos * qkv_size;
        all_qkv[qkv_offset..qkv_offset + qkv_size].copy_from_slice(&qkv);
        cache_kv_for_position(&qkv, dim, kv_cache, layer_no);
    }

    all_qkv
}

pub struct AttentionForPositionArgs<'a> {
    pub hidden: &'a mut Vec<f32>,
    pub attn_raw: &'a Vec<f32>,
    pub layer: &'a Layer,
    pub pos: usize,
    pub dim: usize
}

fn apply_attention_to_position(args: AttentionForPositionArgs) {
    let AttentionForPositionArgs {hidden, attn_raw, layer, pos, dim} = args;
    let attn_start = pos * dim;
    let attn_end = attn_start + dim;
    let a = &attn_raw[attn_start..attn_end];

    let h_start = pos * dim;
    let h_end = h_start + dim;
    let mut h = &mut hidden[h_start..h_end];

    project_and_residual(ProjectAndResidualArgs {activation_weights: &a, projection_weights: &layer.proj_w, projection_bias: &layer.proj_b, dim, hidden: h});
}

pub struct AttentionBlockArgs<'a> {
    pub hidden: &'a mut Vec<f32>,
    pub layer: &'a Layer,
    pub seq_len: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub kv_cache: &'a mut KVCache,
    layer_no: usize
}

fn apply_attention_block(args: AttentionBlockArgs) {
    let AttentionBlockArgs {hidden, layer, seq_len, dim, n_heads, kv_cache, layer_no} = args;
    let qkv_size = dim * 3;
    let all_qkv_size = seq_len * qkv_size;

    let all_qkv = fill_all_qkv(FillAllQKVArgs {
        all_qkv_size, qkv_size, hidden, seq_len, dim, layer, kv_cache, layer_no
    });

    let attn_raw = multi_head_attention(&all_qkv, seq_len, dim, n_heads);

    for pos in 0..seq_len {
        apply_attention_to_position(AttentionForPositionArgs {
            hidden, attn_raw: &attn_raw, layer, pos, dim
        });
    }
}

pub struct FFNForPositionArgs<'a> {
    pub hidden: &'a mut Vec<f32>,
    pub layer: &'a Layer,
    pub pos: usize,
    pub dim: usize,
    pub ffn_dim: usize,
}

fn apply_ffn_to_position(args: FFNForPositionArgs) {
    let FFNForPositionArgs {hidden, layer, pos, dim, ffn_dim} = args;
    let Layer {ln2_w, ln2_b, ffn_up_w, ffn_up_b, ffn_down_w, ffn_down_b, ..} = layer;
    let offset = pos * dim;
    let end = offset + dim;
    let mut h = &mut hidden[offset..end];

    let ffn_up = normalize_and_project(NormalizeAndProject {
        hidden: &mut h,
        norm_weights: &layer.ln2_w,
        norm_bias: &layer.ln2_b,
        projection_weights: &layer.ffn_up_w,
        projection_bias: &layer.ffn_up_b,
        projection_dim: ffn_dim
    });

    let mut ffn_act = gelu(&ffn_up);
    let h_mut = &mut hidden[offset..end];
    project_and_residual(ProjectAndResidualArgs { activation_weights: &mut ffn_act, projection_weights: ffn_down_w, projection_bias: ffn_down_b, dim, hidden: h_mut });
}

pub struct TransformerLayerArgs<'a> {
    pub hidden: &'a mut Vec<f32>,
    pub layer: &'a Layer,
    pub seq_len: usize,
    pub dim: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub kv_cache: &'a mut KVCache,
    pub layer_no: usize,
}

pub fn apply_transformer_layer(args: TransformerLayerArgs) {
    let TransformerLayerArgs {hidden, layer, seq_len, dim, n_heads, ffn_dim, kv_cache, layer_no} = args;
    apply_attention_block(AttentionBlockArgs { hidden, layer, seq_len, dim, n_heads, kv_cache, layer_no });

    for pos in 0..seq_len {
        apply_ffn_to_position(FFNForPositionArgs { hidden, layer, pos, dim, ffn_dim });
    }
}




pub struct TransformerMultiArgs<'a> {
    pub num_layers: usize,
    pub wte: &'a Vec<f32>,
    pub wpe: &'a Vec<f32>,
    pub layers: &'a Vec<Layer>,
    pub ln_f_w: &'a Vec<f32>,
    pub ln_f_b: &'a Vec<f32>,
    pub dim: usize,
    pub token_ids: &'a Vec<usize>,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub kv_cache: &'a mut KVCache,
}
pub fn transformer_multi(args: TransformerMultiArgs) -> usize {
    let TransformerMultiArgs {
        n_heads, dim, num_layers, layers, ln_f_w, ln_f_b,
        wte, wpe,  token_ids, ffn_dim, vocab_size, kv_cache
    } = args;
    let seq_len = token_ids.len();
    let hidden_size = seq_len * dim;

    let last_pos = seq_len - 1;

    let mut hidden = initialize_hidden(InitializeHiddenArgs {hidden_size, dim, wpe, wte, token_ids});

    for layer_no in 0..num_layers {
        let layer = &layers[layer_no];

        apply_transformer_layer(TransformerLayerArgs {
           hidden: &mut hidden, layer, seq_len, dim, n_heads, ffn_dim, kv_cache, layer_no
        });

    }

    let top_id = compute_logits(ComputeLogitsArgs {hidden: &hidden, wte, ln_f_w, ln_f_b, last_pos, dim, vocab_size});

    top_id
}


pub struct GenerateTokenSingleLayerArgs<'a> {
    pub layer_no: usize,
    pub qkv_size: usize,
    pub n_heads: usize,
    pub dim: usize,
    pub hidden: &'a mut Vecf32,
    pub layer: &'a Layer,
    pub kv_cache: &'a mut KVCache,
    pub head_dim: usize,
    pub ffn_dim: usize,
}

pub fn generate_token_single_layer(args: GenerateTokenSingleLayerArgs) {
    let GenerateTokenSingleLayerArgs {
        hidden, layer, layer_no, qkv_size,
        kv_cache, n_heads, dim, head_dim, ffn_dim
    } = args;

    let qkv = normalize_and_project(NormalizeAndProject {
        hidden, norm_weights: &layer.ln1_w, norm_bias: &layer.ln1_b,
        projection_weights: &layer.qkv_w, projection_bias: &layer.qkv_b, projection_dim: qkv_size
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
        activation_weights: &attn_raw, projection_weights: &layer.proj_w, projection_bias: &layer.proj_b, dim, hidden
    });

    let ffn_up = normalize_and_project(NormalizeAndProject {
       hidden, norm_weights: &layer.ln2_w, norm_bias: &layer.ln2_b, projection_weights: &layer.ffn_up_w, projection_bias: &layer.ffn_up_b, projection_dim: ffn_dim
    });

    let ffn_act = gelu(&ffn_up);

    project_and_residual(ProjectAndResidualArgs {
        activation_weights: &ffn_act, projection_weights: &layer.ffn_down_w, projection_bias: &layer.ffn_down_b, dim, hidden
    });
}

pub struct GenerateTokenArgs<'a> {
    pub token_id: usize,
    pub dim: usize,
    pub position: usize,
    pub wte: &'a Vecf32,
    pub wpe: &'a Vecf32,
    pub num_layers: usize,
    pub layers: &'a Vec<Layer>,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub seq_len: usize,
    pub kv_cache: &'a mut KVCache,
    pub head_dim: usize,
    pub ln_f_w: &'a Vecf32,
    pub ln_f_b: &'a Vecf32,
    pub vocab_size: usize,
}
pub fn generate_token(args: GenerateTokenArgs) -> usize {
    let GenerateTokenArgs {
        token_id, dim, position, wte, wpe, num_layers, layers,
        n_heads, ffn_dim, seq_len, kv_cache, head_dim, ln_f_w, ln_f_b,
        vocab_size,
    } = args;

    let wte_start = token_id * dim;
    let wpe_start = position * dim;
    let wte_row = &wte[wte_start..wte_start + dim];
    let wpe_row = &wpe[wpe_start..wpe_start + dim];
    let qkv_size = dim * 3;

    let mut hidden = combine_embedding_weights(wte_row, wpe_row);

    for layer_no in 0..num_layers {
        let layer = &layers[layer_no];
        generate_token_single_layer(GenerateTokenSingleLayerArgs {
            layer_no, layer, qkv_size, hidden: &mut hidden, n_heads, ffn_dim, kv_cache, dim, head_dim
        });
    }
    let top_id = compute_logits(ComputeLogitsArgs {hidden: &hidden, wte, ln_f_w, ln_f_b, last_pos: 0, dim, vocab_size});

    top_id
}