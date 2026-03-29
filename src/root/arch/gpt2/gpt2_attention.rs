use crate::root::activation::softmax;
use crate::root::arch::gpt2::gpt2_linear::{scale_and_add_multi, ScaleAndAddMultiArgs};
use crate::root::common_types::{Arrf32, Vecf32};
use crate::root::kv_cache::PositionKVVectors;

pub struct GPT2QKV {
    pub q: Vecf32,
    pub k: Vecf32,
    pub v: Vecf32,
}
pub fn gpt2_extract_qkv(qkv: &Arrf32) -> GPT2QKV {
    let dim = qkv.len() / 3;
    let q = qkv[0..dim].to_vec();
    let k = qkv[dim..dim * 2].to_vec();
    let v = qkv[dim * 2..dim * 3].to_vec();

    GPT2QKV {q, k, v}
}

struct GPT2ComputeScoreArgs<'a> {
    all_q: &'a Arrf32,
    q_offset: usize,
    all_k: &'a Arrf32,
    k_offset: usize,
    head_dim: usize,
}
fn gpt2_compute_score(args: GPT2ComputeScoreArgs) -> f32 {
    let GPT2ComputeScoreArgs {all_q, all_k, q_offset, k_offset, head_dim} = args;
    let mut dot = 0.0f32;
    for i in 0..head_dim {
        dot += all_q[q_offset + i] * all_k[k_offset + i];
    }

    let score = dot / (head_dim as f32).sqrt();

    score
}

pub struct GPT2CachedComputeScores<'a> {
    pub cache_len: usize,
    pub k_cache: &'a PositionKVVectors,
    pub head_dim: usize,
    pub h_offset: usize,
    pub q: &'a Arrf32,
}
fn gpt2_cached_compute_scores(args: GPT2CachedComputeScores) -> Vecf32 {
    let GPT2CachedComputeScores {cache_len, k_cache, head_dim, h_offset, q} = args;

    let mut scores = vec![0.0f32; cache_len];
    for prev in 0..cache_len {
        let all_k = &k_cache[prev];
        let score = gpt2_compute_score(GPT2ComputeScoreArgs {
            head_dim, all_q: q, all_k, q_offset: h_offset, k_offset: h_offset
        });
        scores[prev] = score;
    }

    scores
}


pub struct GPT2CachedAttentionArgs<'a> {
    pub q: &'a Arrf32,
    pub k_cache: &'a PositionKVVectors,
    pub v_cache: &'a PositionKVVectors,
    pub cache_len: usize,
    pub n_heads: usize,
    pub dim: usize,
    pub head_dim: usize,
}
pub fn gpt2_cached_attention(args: GPT2CachedAttentionArgs) -> Vecf32 {
    let GPT2CachedAttentionArgs {q, k_cache, v_cache, n_heads, dim, head_dim, cache_len} = args;

    let mut output = vec![0.0f32; dim];

    for h in 0..n_heads {
        let h_offset = h * head_dim;
        let scores = gpt2_cached_compute_scores(GPT2CachedComputeScores {
            cache_len, k_cache, head_dim, h_offset, q
        });

        let probs = softmax(&scores);

        scale_and_add_multi(ScaleAndAddMultiArgs {
            output: &mut output, sources: v_cache, num_iterations: head_dim, start_offset: h_offset, scales: &probs, seq_len: cache_len
        });
    }
    output
}
