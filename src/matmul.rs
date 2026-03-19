pub fn matmul(input: &[f32], weight: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = input.len();
    let mut output = vec![0.0f32; out_dim];
    for j in 0..out_dim {
        let mut sum = 0.0f32;
        for i in 0..in_dim {
            sum += input[i] * weight[i * out_dim + j];
        }
        output[j] = sum;
    }

    output
}