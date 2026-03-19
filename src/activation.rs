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