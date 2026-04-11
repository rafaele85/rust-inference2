use crate::root::common_types::{Arrf32, Vecf32};

pub fn add_to_vector(input: &mut Arrf32, vector_to_add: &Arrf32) -> () {
    for i in 0..input.len() {
        input[i] += vector_to_add[i];
    }
}

pub fn scale(v: Vecf32, scale: f32) -> Vecf32 {
    v.iter().map(|&x| scale * x).collect()
}

pub struct ScaleAndAddArgs<'a> {
    pub output: &'a mut Vecf32,
    pub source: &'a Vecf32,
    pub scale: f32,
    pub start_offset: usize,
    pub num_iterations: usize,
}
pub fn scale_and_add(args: ScaleAndAddArgs) {
    let ScaleAndAddArgs {output, source, scale, start_offset, num_iterations} = args;
    for i in 0..num_iterations {
        output[start_offset + i] += scale * source[start_offset + i];
    }
}


pub fn hadamard(a: Vecf32, b: Vecf32, len: usize) -> Vecf32 {
    let mut output = vec![0.0f32; len];
    for i in 0..len {
        output[i] = a[i] * b[i];
    }

    output
}

pub fn sum_products(a: &Vecf32, b: &Vecf32, len: usize) -> f32 {
    let mut out = 0.0f32;
    for i in 0..len {
        out += a[i] * b[i];
    }
    out
}

pub fn sum_products_with_offset(a: &Vecf32, a_offset: usize, b: &Vecf32, b_offset: usize, len: usize) -> f32 {
    let mut out = 0.0f32;
    for i in 0..len {
        out += a[a_offset + i] * b[b_offset + i];
    }
    out
}


pub fn sum_sq(a: &Vecf32, a_offset: usize, len: usize) -> f32 {
    let mut out = 0.0f32;
    for i in 0..len {
        out += a[a_offset + i] * a[a_offset + i];
    }
    out
}


pub fn scale_with_offset(out: &mut Vecf32, scale: f32, offset: usize, len: usize) {
    for i in 0..len {
        out[offset + i] *= scale;
    }
}

