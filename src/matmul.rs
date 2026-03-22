use crate::common_types::{Arrf32, Vecf32, Weight};
use crate::gpu::{Gpu};

pub fn matmul(input: &Arrf32, weight: &Weight, out_dim: usize, gpu: &Gpu) -> Vecf32 {
    gpu.matmul(input, weight, input.len(), out_dim)
}

pub fn matmul_transposed(input: &Arrf32, weight: &Weight, out_dim: usize, gpu: &Gpu) -> Vecf32 {
    gpu.matmul_transposed(input, weight, input.len(), out_dim)
}