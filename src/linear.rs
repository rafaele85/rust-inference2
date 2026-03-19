use crate::matmul::matmul;
use crate::norm::layer_norm;

pub fn linear_projection(input: &[f32], weight: &[f32], bias: &[f32], out_dim: usize) -> Vec<f32> {
    let mut output = matmul(input, weight, out_dim);
    for i in 0..out_dim {
        output[i] += bias[i];
    }

    output
}

pub fn add_to_vector(input: &mut Vec<f32>, vector_to_add: &[f32]) -> () {
    for i in 0..input.len() {
        input[i] += vector_to_add[i];
    }
}

pub struct ProjectAndResidualArgs<'a> {
    pub hidden: &'a mut Vec<f32>,
    pub activation_weights: &'a Vec<f32>,
    pub projection_weights: &'a Vec<f32>,
    pub projection_bias: &'a Vec<f32>,
    pub dim: usize,
}

pub fn project_and_residual(args: ProjectAndResidualArgs) -> () {
    let projected = linear_projection(&args.activation_weights, &args.projection_weights, &args.projection_bias, args.dim);
    add_to_vector(args.hidden, &projected);
}

pub struct NormalizeAndProject<'a> {
    pub hidden: &'a mut Vec<f32>,
    pub norm_weights: &'a Vec<f32>,
    pub norm_bias: &'a Vec<f32>,
    pub projection_weights: &'a Vec<f32>,
    pub projection_bias: &'a Vec<f32>,
    pub projection_dim: usize,
}

pub fn normalize_and_project(args: NormalizeAndProject) -> Vec<f32> {
    let mut normalized = layer_norm(args.hidden, args.norm_weights, args.norm_bias);
    let result = linear_projection(&mut normalized, args.projection_weights, args.projection_bias, args.projection_dim);

    result
}