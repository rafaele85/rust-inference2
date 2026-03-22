use crate::activation::argmax;
use crate::common_types::{Arrf32};
use crate::gpu::Gpu;
use crate::matmul::matmul_transposed;
use crate::model::gpt2::gpt2::GPT2Model;
use crate::norm::layer_norm;

pub struct ComputeLogitsArgs<'a> {
    pub hidden: &'a Arrf32,
    pub last_pos: usize,
    pub dim: usize,
    pub vocab_size: usize,
    pub gpu : &'a Gpu,
    pub gpt2_model: &'a GPT2Model
}

pub fn compute_logits(args: ComputeLogitsArgs) -> usize {
    let ComputeLogitsArgs {hidden, last_pos, dim, vocab_size, gpu, gpt2_model } = args;
    let start = last_pos * dim;
    let end = start + dim;
    let last_h = &hidden[start..end];
    let ln_out = layer_norm(last_h, &gpt2_model.ln_f_w, &gpt2_model.ln_f_b);
    let logits = matmul_transposed(&ln_out, &gpt2_model.wte_gpu, vocab_size, gpu);
    let top_id = argmax(&logits);

    top_id
}

