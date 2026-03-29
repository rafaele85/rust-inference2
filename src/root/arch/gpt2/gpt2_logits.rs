use crate::root::activation::argmax;
use crate::root::arch::gpt2::gpt2::GPT2Model;
use crate::root::arch::gpt2::gpt2_norm::layer_norm;
use crate::root::common_types::Arrf32;
use crate::root::gpu::Gpu;
use crate::root::matmul::matmul_transposed;

pub struct GPT2ComputeLogitsArgs<'a> {
    pub hidden: &'a Arrf32,
    pub last_pos: usize,
    pub dim: usize,
    pub vocab_size: usize,
    pub gpu : &'a mut Gpu,
    pub gpt2_model: &'a GPT2Model
}

pub fn gpt2_compute_logits(args: GPT2ComputeLogitsArgs) -> usize {
    let GPT2ComputeLogitsArgs {hidden, last_pos, dim, vocab_size, gpu, gpt2_model } = args;
    let start = last_pos * dim;
    let end = start + dim;
    let last_h = &hidden[start..end];
    let ln_out = layer_norm(last_h, &gpt2_model.ln_f_w, &gpt2_model.ln_f_b, gpt2_model.config.layer_norm_epsilon);
    let logits = matmul_transposed(&ln_out, &gpt2_model.wte_gpu, vocab_size, gpu);
    let top_id = argmax(&logits);

    top_id
}

