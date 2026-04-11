use crate::root::common_types::{Arrf32, Vecf32};
use crate::root::linear::{scale_with_offset, sum_sq};

pub fn rms_norm(input: &Arrf32, weight: &Arrf32, eps: f32) -> Vecf32 {
    let n = input.len() as f32;
    let variance: f32 = input.iter().map(|x| x * x).sum::<f32>() / n;
    let scale = 1.0 / (variance + eps).sqrt();

    let result = input.iter().enumerate().map(|(i, x)| weight[i] * x * scale).collect();

    result
}


pub fn l2_norm(x: &Arrf32, dim: usize, num_heads: usize, eps: f32) -> Vecf32 {

    let mut out = x.to_vec();

    for h in 0..num_heads {
        let offset = h * dim;
        let sum_sq = sum_sq(&out, offset, dim);
        let scale = 1.0 / (sum_sq + eps).sqrt();
        scale_with_offset(&mut out, scale, offset, dim);
    }

    out
}


pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

pub fn rms_norm_zero_centered(input: &Arrf32, weight: &Arrf32, eps: f32) -> Vecf32 {
    let n = input.len() as f32;
    let variance = sum_sq(&input, 0, input.len()) / n;
    let scale = 1.0 / (variance + eps).sqrt();

    let res = input.iter().enumerate().map(|(i, x)| (1.0 + weight[i]) * x * scale).collect();

    res
}
