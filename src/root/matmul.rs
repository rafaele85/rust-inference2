use crate::root::common_types::{Arrf32, Vecf32, Weight, WeightF16};
use crate::root::gpu::Gpu;

pub fn matmul_transposed(input: &Arrf32, weight: &Weight, out_dim: usize, gpu: &mut Gpu) -> Vecf32 {
    gpu.matmul_transposed(input, weight, input.len(), out_dim)
}

pub fn matmul_transposed_f16(input: &Arrf32, weight: &WeightF16, out_dim: usize, gpu: &mut Gpu) -> Vecf32 {
    gpu.matmul_transposed_f16(input, weight, input.len(), out_dim)
}