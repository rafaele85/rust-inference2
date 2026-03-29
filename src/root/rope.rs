use crate::root::common_types::Vecf32;

pub fn compute_rope_freqs(head_dim: usize, pos: usize, theta: f32) -> (Vecf32, Vecf32) {
    let half= head_dim / 2;
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
