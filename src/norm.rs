pub fn layer_norm(input: &[f32], weight: &[f32], bias: &[f32] ) -> Vec<f32> {
    let n = input.len() as f32;
    let mean: f32 = input.iter().sum::<f32>() / n;
    let variance: f32 = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let eps = 1e-5f32;
    let std = (variance + eps).sqrt();

    let result = input.iter().enumerate().map(|(i, &x)| {
        weight[i] * (x - mean) / std + bias[i]
    }).collect();

    result
}