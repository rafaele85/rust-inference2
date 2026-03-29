use crate::root::common_types::Vecf32;

pub fn hadamard(a: Vecf32, b: Vecf32, len: usize) -> Vecf32 {
    let mut output = vec![0.0f32; len];
    for i in 0..len {
        output[i] = a[i] * b[i];
    }

    output
}