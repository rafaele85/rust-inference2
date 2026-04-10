use std::borrow::Cow;
use crate::root::activation::softmax;
use crate::root::common_types::{Arrf32, Vecf32, WeightF16};
use crate::root::config::{AttnConfig, MLAConfig, ModelDimConfig, NormConfig};
use crate::root::gpu::Gpu;
use crate::root::kv_cache::{KVCache, PositionKVVectors};
use crate::root::layer::{MLAKVLayer};
use crate::root::linear::add_to_vector;
use crate::root::matmul::matmul_transposed_f16;
use crate::root::norm::rms_norm;
use crate::root::rope::{apply_rope, apply_rope_to_head, compute_rope_freqs, ApplyRopeArgs, ApplyRopeToHeadArgs};

pub struct ComputeQKScoresArgs<'a> {
    pub cache_len: usize,
    pub q_offset: usize,
    pub head_dim: usize,
    pub q: &'a Vecf32,
    pub kv_offset: usize,
    pub k_cache: &'a PositionKVVectors
}
pub fn compute_qk_scores(args: ComputeQKScoresArgs) -> Vecf32 {
    let ComputeQKScoresArgs {cache_len, head_dim, q_offset, kv_offset, q, k_cache} = args;
    let mut scores = vec![0.0f32; cache_len];
    for pos in 0..cache_len {
        let mut dot = 0.0f32;
        for i in 0..head_dim {
            dot += q[q_offset + i] * k_cache[pos][kv_offset + i];
        }
        scores[pos] = dot / (head_dim as f32).sqrt();
    }
    let probs = softmax(&scores);

    probs
}

pub fn project_v_to_attn(attn_out: &mut Vecf32, probs: &Vecf32, cache:  &PositionKVVectors, cache_len: usize, dim: usize, cache_offset: usize, attn_offset: usize) {
    for pos in 0..cache_len {
        for i in 0..dim {
            attn_out[attn_offset+i] += probs[pos] * cache[pos][cache_offset+i];
        }
    }
}

fn kv_down_project(normed: &Vecf32, kv_a_proj_w: &WeightF16, kv_a_total: usize, kv_lora_rank: usize, gpu: &mut Gpu) -> (Vecf32, Vecf32) {
    let kv_a = matmul_transposed_f16(normed, kv_a_proj_w, kv_a_total, gpu);
    let compressed_kv = kv_a[..kv_lora_rank].to_vec();
    let k_rope_shared = kv_a[kv_lora_rank..].to_vec();
    (compressed_kv, k_rope_shared)
}

fn kv_up_project(compressed_kv: &Vecf32, kv_a_layernorm_w: &Vecf32, kv_b_proj_w: &WeightF16, kv_b_total: usize, eps: f32, gpu: &mut Gpu) -> Vecf32 {
    let normed_kv = rms_norm(compressed_kv, kv_a_layernorm_w, eps);
    matmul_transposed_f16(&normed_kv, kv_b_proj_w, kv_b_total, gpu)
}

struct BuildKVForCachingArgs<'a, TConfig: MLAConfig + ModelDimConfig + AttnConfig> {
    config: &'a TConfig,
    kv_b: &'a Vecf32,
    k_rope_roped: &'a Arrf32,
    qk_head_dim: usize,
}
fn build_kv_for_caching<TConfig: MLAConfig + ModelDimConfig + AttnConfig>(args: BuildKVForCachingArgs<TConfig>) -> (Vecf32, Vecf32) {
    let BuildKVForCachingArgs {kv_b, k_rope_roped, config, qk_head_dim } = args;
    assert_eq!(k_rope_roped.len(), config.get_qk_rope_head_dim(), "k_rope_roped length mismatch");
    let mut full_k = vec![0.0f32; config.get_num_attention_heads() * qk_head_dim];
    let mut full_v = vec![0.0f32; config.get_num_attention_heads() * config.get_v_head_dim()];

    let h_dim = config.get_qk_nope_head_dim() + config.get_v_head_dim();
    for h in 0..config.get_num_attention_heads() {
        let kv_b_offset = h * h_dim;
        let k_offset = h * qk_head_dim;
        let k_rope_offset = k_offset + config.get_qk_nope_head_dim();
        let v_offset = h * config.get_v_head_dim();
        let v_src_offset = kv_b_offset + config.get_qk_nope_head_dim();

        let k_slice = &kv_b[kv_b_offset..kv_b_offset + config.get_qk_nope_head_dim()];
        full_k[k_offset..k_offset + config.get_qk_nope_head_dim()].copy_from_slice(k_slice);
        full_k[k_rope_offset..k_rope_offset + config.get_qk_rope_head_dim()].copy_from_slice(&k_rope_roped);
        let v_slice = &kv_b[v_src_offset..v_src_offset + config.get_v_head_dim()];
        full_v[v_offset..v_offset + config.get_v_head_dim()].copy_from_slice(v_slice);
    }

    (full_k, full_v)
}

struct ApplyMLARopeResult  {
    q: Vecf32,
    k_rope_roped: Vecf32,
}

struct ApplyMLARopeArgs<'a, TConfig: MLAConfig + AttnConfig> {
    q_raw: &'a Vecf32,
    k_rope_shared: &'a Vecf32,
    position: usize,
    config: &'a TConfig,
    qk_head_dim: usize,
}

fn apply_mla_rope<TConfig: MLAConfig + AttnConfig>(args: ApplyMLARopeArgs<TConfig>) -> ApplyMLARopeResult {
    let ApplyMLARopeArgs {q_raw, k_rope_shared, config, position, qk_head_dim } = args;
    let half = config.get_qk_rope_head_dim()/2;
    let (cos, sin) = compute_rope_freqs(half, position, config.get_rope_theta());

    let mut q = q_raw.clone();

    apply_rope(ApplyRopeArgs {
        out: &mut q, x: &q_raw, num_heads: config.get_num_attention_heads(), cos: &cos, sin: &sin,
        head_dim: qk_head_dim, nope_dim: config.get_qk_nope_head_dim(), interleaved: true
    });

    let mut k_rope_roped = vec![0.0f32; config.get_qk_rope_head_dim()];
    apply_rope_to_head(ApplyRopeToHeadArgs {x: &k_rope_shared, out: &mut k_rope_roped, offset: 0, cos: &cos, sin: &sin, interleaved: true});

    ApplyMLARopeResult {q, k_rope_roped}
}


struct MLAAttentionScoresArgs<'a> {
    q: &'a Vecf32,
    k_cache: &'a PositionKVVectors,
    v_cache: &'a PositionKVVectors,
    num_attention_heads: usize,
    v_head_dim: usize,
    qk_head_dim: usize,
}
fn mla_attention_scores(args: MLAAttentionScoresArgs) -> Vecf32 {
    let MLAAttentionScoresArgs {q, k_cache, v_cache, num_attention_heads,v_head_dim, qk_head_dim} = args;
    let cache_len = k_cache.len();

    let mut attn_out = vec![0.0f32; num_attention_heads * v_head_dim];

    for h in 0..num_attention_heads {
        let q_offset = h * qk_head_dim;
        let probs = compute_qk_scores(ComputeQKScoresArgs {cache_len, kv_offset: q_offset, q_offset, head_dim: qk_head_dim, q: &q, k_cache});

        let v_offset = h * v_head_dim;
        project_v_to_attn(&mut attn_out, &probs, &v_cache, cache_len, v_head_dim, v_offset, v_offset);
    }

    attn_out
}


fn q_project(normed: &Cow<Arrf32>, q_proj_w: &WeightF16, q_total: usize, gpu: &mut Gpu) -> Vecf32 {
    let res = matmul_transposed_f16(normed, q_proj_w, q_total, gpu);

    res
}

pub struct MLAAttentionArgs<'a, TLayer: MLAKVLayer, TConfig: NormConfig + ModelDimConfig + MLAConfig + AttnConfig> {
    pub hidden: &'a mut Vecf32,
    pub normed: &'a Vecf32,
    pub layer: &'a TLayer,
    pub kv_cache: &'a mut KVCache,
    pub layer_no: usize,
    pub position: usize,
    pub gpu: &'a mut Gpu,
    pub config: &'a TConfig,
}
pub fn mla_attention<TLayer: MLAKVLayer, TConfig: NormConfig + ModelDimConfig + MLAConfig + AttnConfig>(args: MLAAttentionArgs<TLayer, TConfig>) {  // reuse same args struct
    let MLAAttentionArgs {hidden, normed,
        layer,
        config,
        kv_cache, layer_no, position, gpu} = args;

    let kv_a_total = config.get_kv_lora_rank() + config.get_qk_rope_head_dim();
    let kv_b_total = config.get_num_attention_heads() * (config.get_qk_nope_head_dim() + config.get_v_head_dim());
    let qk_head_dim = config.get_qk_nope_head_dim() + config.get_qk_rope_head_dim();
    let q_total = config.get_num_attention_heads() * qk_head_dim;
    let q_raw_normed = layer.compute_q_raw_normed(&normed, config, gpu);
    let q_raw_proj_w = layer.get_q_raw_proj_w();
    let q_raw = q_project(&q_raw_normed, &q_raw_proj_w, q_total, gpu);

    let (compressed_kv, k_rope_shared) = kv_down_project(normed, layer.get_kv_a_proj_w(), kv_a_total, config.get_kv_lora_rank(), gpu);

    let kv_b = kv_up_project(&compressed_kv, &layer.get_kv_a_layernorm_w(), layer.get_kv_b_proj_w(), kv_b_total, config.get_rms_norm_eps(), gpu);

    let ApplyMLARopeResult {q, k_rope_roped} = apply_mla_rope(ApplyMLARopeArgs {
        q_raw: &q_raw, k_rope_shared: &k_rope_shared, qk_head_dim, position, config
    });

    let (full_k, full_v) = build_kv_for_caching(BuildKVForCachingArgs::<TConfig> {
        config, qk_head_dim, kv_b: &kv_b, k_rope_roped: &k_rope_roped
    });

    kv_cache.append(layer_no, full_k, full_v);

    let attn_out = mla_attention_scores(MLAAttentionScoresArgs {
        q: &q, k_cache: &kv_cache.k[layer_no], v_cache: &kv_cache.v[layer_no], v_head_dim: config.get_v_head_dim(), num_attention_heads: config.get_num_attention_heads(), qk_head_dim
    });

    let projected = matmul_transposed_f16(&attn_out, layer.get_o_proj_w(), config.get_hidden_size(), gpu);

    add_to_vector(hidden, &projected);
}

pub fn gqa_attention<TC: AttnConfig>(
    q: &Vecf32,
    k_cache: &PositionKVVectors,
    v_cache: &PositionKVVectors,
    tc: &TC,
    attn_dim: usize,
    repeats: usize,
) -> Vecf32 {
    let cache_len = k_cache.len();
    let head_dim = tc.get_head_dim();
    let num_attention_heads = tc.get_num_attention_heads();

    let mut attn_out = vec![0.0; attn_dim];

    for h in 0..num_attention_heads {
        let kv_h = h / repeats;
        let q_offset = h * head_dim;
        let kv_offset = kv_h * head_dim;

        let probs = compute_qk_scores(ComputeQKScoresArgs {
            cache_len,
            q_offset,
            head_dim,
            q,
            kv_offset,
            k_cache: &k_cache,
        });

        project_v_to_attn(&mut attn_out, &probs, &v_cache, cache_len, head_dim, kv_offset, q_offset);
    }

    attn_out
}