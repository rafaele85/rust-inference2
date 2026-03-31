use crate::root::activation::softmax;
use crate::root::common_types::Vecf32;
use crate::root::kv_cache::PositionKVVectors;

pub struct ComputeQKScoresArgs<'a> {
    pub(crate) cache_len: usize,
    pub(crate) q_offset: usize,
    pub(crate) head_dim: usize,
    pub(crate) q: &'a Vecf32,
    pub(crate) kv_offset: usize,
    pub(crate) k_cache: &'a PositionKVVectors
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

pub fn project_v_to_attn(attn_out: &mut Vecf32, probs: &Vecf32, cache:  &&PositionKVVectors, cache_len: usize, dim: usize, cache_offset: usize, attn_offset: usize) {
    for pos in 0..cache_len {
        for i in 0..dim {
            attn_out[attn_offset+i] += probs[pos] * cache[pos][cache_offset+i];
        }
    }
}