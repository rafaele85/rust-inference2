use crate::common_types::{Arrf32, Vecf32};

pub fn silu(input: &Arrf32) -> Vecf32 {
    let result = input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    result
}