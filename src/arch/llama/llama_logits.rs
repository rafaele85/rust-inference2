use crate::activation::argmax;
use crate::common_types::{Vecf32, WeightF16};
use crate::gpu::Gpu;
use crate::matmul::{matmul_transposed_f16};
use crate::arch::llama::llama_norm::rms_norm;

pub fn llama_logits(hidden: Vecf32, final_norm_w: &Vecf32, lm_head_w: &WeightF16, vocab_size: usize, eps: f32, gpu: &mut Gpu) -> usize {
    let normed = rms_norm(&hidden, &final_norm_w, eps);
    let logits = matmul_transposed_f16(&normed, lm_head_w, vocab_size, gpu);
    let top_id = argmax(&logits);

    top_id
}