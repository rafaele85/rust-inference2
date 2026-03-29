use crate::root::common_types::{Arrf32, Vecf32};


pub fn apply_rope(x: &Arrf32, cos: &Arrf32, sin: &Arrf32, head_dim: usize) -> Vecf32 {
    let half = head_dim / 2;
    let mut out = vec![0.0f32; x.len()];

    let num_heads = x.len() / head_dim;
    for h in 0..num_heads {
        let offset = h * head_dim;
        for i in 0..half {
            let x1 = x[offset + i];
            let x2 = x[offset + half + i];
            let out1 = x1 * cos[i] - x2 * sin[i];
            let out2 = x2 * cos[i] + x1 * sin[i];
            out[offset + i] = out1;
            out[offset + half + i] = out2;
        }
    }

    out
}