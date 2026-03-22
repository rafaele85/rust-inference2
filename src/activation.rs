use crate::common_types::{Arrf32, Vecf32};

const GELU_COEFF: f32 = 0.044715;
const HALF: f32 = 0.5;

const SQRT_2_OVER_PI: f32 = 0.7978845608;
pub fn get_gelu_scalar(x: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x.powi(3));
    let result = HALF * x * (1.0 + inner.tanh());

    result
}

pub fn gelu(input: &[f32]) -> Vec<f32> {
    input.iter()
        .map(|&x| get_gelu_scalar(x))
        .collect()
}

pub fn softmax(input: &[f32]) -> Vec<f32> {
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let result = exps.iter().map(|&x| x / sum).collect();

    result
}

pub fn argmax(values: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val: f32 = f32::NEG_INFINITY;

    for (i, &v) in values.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }

    max_idx
}

pub fn silu(input: &Arrf32) -> Vecf32 {
    let result = input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    result
}