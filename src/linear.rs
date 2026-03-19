use crate::matmul::matmul;

pub fn linear_projection(input: &[f32], weight: &[f32], bias: &[f32], out_dim: usize) -> Vec<f32> {
    let mut output = matmul(input, weight, out_dim);
    for i in 0..out_dim {
        output[i] += bias[i];
    }

    output
}