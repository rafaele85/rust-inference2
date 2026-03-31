use crate::root::common_types::{Arrf32, Vecf32};

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


pub struct ApplyRopeArgs<'a> {
    pub out: &'a mut Vecf32,
    pub x: &'a Arrf32,
    pub num_heads: usize,
    pub cos: &'a Arrf32,
    pub sin: &'a Arrf32,
    pub head_dim: usize,
    pub nope_dim: usize
}

pub fn apply_rope<'a>(args: ApplyRopeArgs<'a>) -> &'a mut Vecf32 {
    let ApplyRopeArgs {out, x, num_heads, cos, sin, head_dim, nope_dim} = args;
    let half = head_dim / 2;
    for h in 0..num_heads {
        let offset = h * head_dim + nope_dim;
        apply_rope_to_head(ApplyRopeToHeadArgs {out, x, offset, half, cos, sin});
    }

    out
}

pub struct ApplyRopeToHeadArgs<'a> {
    pub out: &'a mut Vecf32,
    pub x: &'a Arrf32,
    pub cos: &'a Arrf32,
    pub sin: &'a Arrf32,
    pub half: usize,
    pub offset: usize,
}

pub fn apply_rope_to_head(args: ApplyRopeToHeadArgs) {
    let ApplyRopeToHeadArgs {half, x, offset, cos, sin, out} = args;
    for i in 0..half {
        let x1 = x[offset + i];
        let x2 = x[offset + half + i];
        let out1 = x1 * cos[i] - x2 * sin[i];
        let out2 = x2 * cos[i] + x1 * sin[i];
        out[offset + i] = out1;
        out[offset + half + i] = out2;
    }
}