use crate::common_types::{Arrf32, Vecf32};

pub fn compute_rope_freqs(head_dim: usize, pos: usize, theta: f32, half: usize) -> (Vecf32, Vecf32) {
    let mut cos_vals = vec![0.0f32; head_dim];
    let mut sin_vals = vec![0.0f32; head_dim];

    for i in 0..half {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        cos_vals[i] = angle.cos();
        cos_vals[i + half] = cos_vals[i];
        sin_vals[i] = angle.sin();
        sin_vals[i + half] = sin_vals[i];
    }

    (cos_vals, sin_vals)
}

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