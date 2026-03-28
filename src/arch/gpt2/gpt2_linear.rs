use crate::common_types::{Arrf32, Vecf32, Weight};
use crate::gpu::Gpu;
use crate::linear::{add_to_vector, scale_and_add, ScaleAndAddArgs};
use crate::arch::gpt2::gpt2_matmul::matmul;
use crate::arch::gpt2::gpt2_norm::layer_norm;

pub fn linear_projection(input: &[f32], weight: &Weight, bias: &[f32], out_dim: usize, gpu: &mut Gpu) -> Vec<f32> {
    let mut output = matmul(input, weight, out_dim, gpu);
    for i in 0..out_dim {
        output[i] += bias[i];
    }

    output
}


pub struct ProjectAndResidualArgs<'a> {
    pub activation_weights: &'a [f32],
    pub projection_weights: &'a Weight,
    pub projection_bias: &'a [f32],
    pub dim: usize,
    pub hidden: &'a mut [f32],
    pub gpu: &'a mut Gpu,
}

pub fn project_and_residual(args: ProjectAndResidualArgs) -> () {
    let projected = linear_projection(&args.activation_weights, &args.projection_weights, &args.projection_bias, args.dim, args.gpu);
    add_to_vector(args.hidden, &projected);
}

pub struct NormalizeAndProject<'a> {
    pub hidden: &'a Arrf32,
    pub norm_weights: &'a Vecf32,
    pub norm_bias: &'a Vecf32,
    pub projection_weights: &'a Weight,
    pub projection_bias: &'a Vecf32,
    pub projection_dim: usize,
    pub gpu: &'a mut Gpu,
    pub eps: f32
}

pub fn normalize_and_project(args: NormalizeAndProject) -> Vecf32 {
    let NormalizeAndProject {hidden, norm_weights, norm_bias, eps, projection_weights, projection_bias, projection_dim,  gpu} = args;
    let normalized = layer_norm(hidden, norm_weights, norm_bias, eps);
    let result = linear_projection(&normalized, projection_weights, projection_bias, projection_dim, gpu);

    result
}


pub struct ScaleAndAddMultiArgs<'a> {
    pub output: &'a mut Vecf32,
    pub sources: &'a Vec<Vecf32>,
    pub scales: &'a Vecf32,
    pub start_offset: usize,
    pub num_iterations: usize,
    pub seq_len: usize,
}
pub fn scale_and_add_multi(args: ScaleAndAddMultiArgs) {
    let ScaleAndAddMultiArgs {seq_len, scales, sources, output, num_iterations, start_offset} = args;
    for i in 0..seq_len {
        let scale = scales[i];
        let source = &sources[i];
        scale_and_add(ScaleAndAddArgs {output, source, num_iterations, start_offset, scale});
    }

}
