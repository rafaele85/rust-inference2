use crate::activation::argmax;
use crate::common_types::Weight;
use crate::gpu::Gpu;
use crate::matmul::matmul_transposed;
use crate::norm::layer_norm;

pub struct ComputeLogitsArgs<'a> {
    pub hidden: &'a [f32],
    pub ln_f_w: &'a [f32],
    pub ln_f_b: &'a [f32],
    pub wte: &'a Weight,
    pub last_pos: usize,
    pub dim: usize,
    pub vocab_size: usize,
    pub gpu : &'a Gpu,
}

pub fn compute_logits(args: ComputeLogitsArgs) -> usize {
    let ComputeLogitsArgs {hidden, ln_f_w, ln_f_b, wte, last_pos, dim, vocab_size, gpu} = args;
    let start = last_pos * dim;
    let end = start + dim;
    let last_h = &hidden[start..end];
    let ln_out = layer_norm(last_h, ln_f_w, ln_f_b);
    let logits = matmul_transposed(&ln_out, wte, vocab_size, gpu);
    let top_id = argmax(&logits);

    top_id
}

