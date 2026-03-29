use crate::root::common_types::{Arrf32, Vecf32};

pub fn rms_norm(input: &Arrf32, weight: &Arrf32, eps: f32) -> Vecf32 {
    let n = input.len() as f32;
    let variance: f32 = input.iter().map(|x| x * x).sum::<f32>() / n;
    let scale = 1.0 / (variance + eps).sqrt();

    let result = input.iter().enumerate().map(|(i, x)| weight[i] * x * scale).collect();

    result
}