use crate::activation::softmax;
use crate::common_types::{Arrf32, Vecf32};
use crate::kv_cache::{KVCache, PositionKVVectors};
use crate::linear::{scale_and_add, scale_and_add_multi, ScaleAndAddArgs, ScaleAndAddMultiArgs};

pub fn extract_q(qkv: &[f32]) -> Vec<f32> {
    let dim = qkv.len() / 3;
    qkv[0..dim].to_vec()
}

pub fn extract_k(qkv: &[f32]) -> Vec<f32> {
    let dim = qkv.len() / 3;
    qkv[dim..dim * 2].to_vec()
}

pub fn extract_v(qkv: &[f32]) -> Vec<f32> {
    let dim = qkv.len() / 3;
    qkv[dim * 2..dim * 3].to_vec()
}

pub struct QKV {
    pub q: Vecf32,
    pub k: Vecf32,
    pub v: Vecf32,
}
pub fn extract_qkv(qkv: &Arrf32) -> QKV {
    let dim = qkv.len() / 3;
    let q = qkv[0..dim].to_vec();
    let k = qkv[dim..dim * 2].to_vec();
    let v = qkv[dim * 2..dim * 3].to_vec();

    QKV {q, k, v}
}

pub struct ComputeScoreArgs<'a> {
    all_q: &'a Arrf32,
    q_offset: usize,
    all_k: &'a Arrf32,
    k_offset: usize,
    head_dim: usize,
}
pub fn compute_score(args: ComputeScoreArgs) -> f32 {
    let ComputeScoreArgs {all_q, all_k, q_offset, k_offset, head_dim} = args;
    let mut dot = 0.0f32;
    for i in 0..head_dim {
        dot += all_q[q_offset + i] * all_k[k_offset + i];
    }

    let score = dot / (head_dim as f32).sqrt();

    score
}

struct ComputeScoresArgs<'a> {
    all_qkv: &'a Arrf32,
    qkv_size: usize,
    k_start: usize,
    h_offset: usize,
    q_offset: usize,
    pos: usize,
    head_dim: usize
}
pub fn compute_scores(args: ComputeScoresArgs) -> Vec<f32> {
    let ComputeScoresArgs {pos, qkv_size, all_qkv, k_start, h_offset, q_offset, head_dim} = args;
    let mut scores = vec![0.0f32; pos + 1];

    for prev in 0..=pos {
        let prev_start = prev * qkv_size;
        let k_offset = prev_start + k_start + h_offset;
        let score = compute_score(ComputeScoreArgs {all_q: all_qkv, all_k: all_qkv, q_offset, k_offset, head_dim});
        scores[prev] = score;
    }

    scores
}

struct ComputeVArgs<'a> {
    prev: usize,
    qkv_size: usize,
    v_start: usize,
    h_offset: usize,
    token_offset: usize,
    head_dim: usize,
    output: &'a mut [f32],
    all_qkv: &'a [f32],
    prob: f32
}
pub fn compute_v(args: ComputeVArgs) {
    let ComputeVArgs {prev, qkv_size, v_start, h_offset, token_offset, head_dim, all_qkv, output, prob} = args;
    let prev_start = prev * qkv_size;
    let v_offset = prev_start + v_start + h_offset;
    let out_offset = token_offset + h_offset;
    for i in 0..head_dim {
        let v = all_qkv[v_offset + i];
        output[out_offset + i] += prob * v;
    }
}

struct MHASingleTokenArgs<'a> {
    pos: usize,
    qkv_size: usize,
    h_offset: usize,
    dim: usize,
    head_dim: usize,
    k_start: usize,
    v_start: usize,
    output: &'a mut [f32],
    all_qkv: &'a [f32],
}
pub fn multi_head_attention_single_token(args: MHASingleTokenArgs) {
    let MHASingleTokenArgs {pos, qkv_size, dim, h_offset, head_dim, v_start, mut output, all_qkv, k_start} = args;
    let pos_start = pos * qkv_size;
    let q_offset = pos_start + h_offset;
    let token_offset = pos * dim;
    let scores = compute_scores(ComputeScoresArgs { pos, q_offset, head_dim, all_qkv, k_start, qkv_size, h_offset});
    let probs = softmax(&scores);

    for prev in 0..=pos {
        compute_v(ComputeVArgs {prev, qkv_size, head_dim, all_qkv, output: &mut output, prob: probs[prev], v_start, h_offset, token_offset});
    }
}

pub fn multi_head_attention(all_qkv: &[f32], seq_len: usize, dim: usize, n_heads: usize) -> Vec<f32> {
    let head_dim = dim / n_heads;
    let qkv_size = dim * 3;
    let out_size = seq_len * dim;
    let mut output = vec![0.0f32; out_size];
    let v_start = dim + dim;
    let k_start = dim;

    for h in 0..n_heads {
        let h_offset = h * head_dim;
        for pos in 0..seq_len {
            multi_head_attention_single_token(MHASingleTokenArgs {dim, head_dim, qkv_size, v_start, k_start, h_offset, pos, output: &mut output, all_qkv});
        }
    }

    output
}


pub struct CachedComputeScores<'a> {
    pub cache_len: usize,
    pub k_cache: &'a PositionKVVectors,
    pub head_dim: usize,
    pub h_offset: usize,
    pub q: &'a Arrf32,
}
fn cached_compute_scores(args: CachedComputeScores) -> Vecf32 {
    let CachedComputeScores {cache_len, k_cache, head_dim, h_offset, q} = args;

    let mut scores = vec![0.0f32; cache_len];
    for prev in 0..cache_len {
        let all_k = &k_cache[prev];
        let score = compute_score(ComputeScoreArgs {
            head_dim, all_q: q, all_k, q_offset: h_offset, k_offset: h_offset
        });
        scores[prev] = score;
    }

    scores
}


pub struct CachedAttentionArgs<'a> {
    pub q: &'a Arrf32,
    pub k_cache: &'a PositionKVVectors,
    pub v_cache: &'a PositionKVVectors,
    pub cache_len: usize,
    pub n_heads: usize,
    pub dim: usize,
    pub head_dim: usize,
}
pub fn cached_attention(args: CachedAttentionArgs) -> Vecf32 {
    let CachedAttentionArgs {q, k_cache, v_cache, n_heads, dim, head_dim, cache_len} = args;

    let mut output = vec![0.0f32; dim];

    for h in 0..n_heads {
        let h_offset = h * head_dim;
        let scores = cached_compute_scores(CachedComputeScores {
            cache_len, k_cache, head_dim, h_offset, q
        });

        let probs = softmax(&scores);

        scale_and_add_multi(ScaleAndAddMultiArgs {
            output: &mut output, sources: v_cache, num_iterations: head_dim, start_offset: h_offset, scales: &probs, seq_len: cache_len
        });
    }
    output
}
